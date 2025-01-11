import copy
import warnings
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C

# from arguments import arg

adaptive_kernel = False  # Automatically tune the kernel hyperparameters
ADAPTIVE_TH = 0.5


def add_t(X, t: float):
    return np.concatenate((X, np.zeros((X.shape[0], 1)) + t), axis=1)


class GaussianProcess:
    def __init__(self, node_coords):
        if adaptive_kernel:
            self.kernel = Matern(
                length_scale=[0.1, 0.1, 3], length_scale_bounds=(1e-3, 1e3), nu=1.5
            )
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=10
            )
        else:
            self.kernel = Matern(length_scale=[0.25, 0.25, 15])
            # self.kernel = Matern(length_scale=[0.1, 0.1, 10])
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel, optimizer=None, n_restarts_optimizer=0
            )
        self.observed_points = []
        self.observed_value = []
        self.node_coords = node_coords
        self.y_pred_at_node, self.std_at_node = None, None
        self.y_pred_at_grid, self.std_at_grid = None, None
        self.env_size = 30
        self.grid = np.array(
            list(product(np.linspace(0, 1, 30), np.linspace(0, 1, 30)))
        )

    def add_observed_point(self, point_pos, value):
        self.observed_points.append(point_pos)
        self.observed_value.append(value)

    def update_gp(self):
        if not self.observed_points:
            return
        scale_t = (
            self.gp.kernel_.length_scale[-1]
            if hasattr(self.gp, "kernel_")
            else self.gp.kernel.length_scale[-1]
        )
        dt = 1.993 * scale_t  # Matern1.5: 2.817: 0.1%; 1.993: 1%; 1.376: 5%; 1.093: 10%
        curr_t = self.observed_points[-1][0][-1]
        mask_idx = []
        for i, ob in enumerate(self.observed_points):
            # print(curr_t, ob[0][-1], dt)
            if curr_t - ob[0][-1] < dt:
                mask_idx.append(i)
        if self.observed_points:
            X = np.take(self.observed_points, mask_idx, axis=0).reshape(-1, 3)
            y = np.take(self.observed_value, mask_idx, axis=0).reshape(-1, 1)
            self.gp.fit(X, y)

    def update_node(self, t):
        self.y_pred_at_node, self.std_at_node = self.gp.predict(
            add_t(self.node_coords, t), return_std=True
        )
        return self.y_pred_at_node, self.std_at_node

    def update_grid(self, t):
        self.y_pred_at_grid, self.std_at_grid = self.gp.predict(
            add_t(self.grid, t), return_std=True
        )
        return self.y_pred_at_grid, self.std_at_grid

    def evaluate_RMSE(self, y_true, t=None):
        if t is not None:
            self.update_grid(t)
        RMSE = np.sqrt(mean_squared_error(self.y_pred_at_grid, y_true))
        return RMSE

    def evaluate_F1score(self, y_true, t):
        score = self.gp.score(add_t(self.grid, t), y_true)
        return score

    def evaluate_cov_trace(self, idx=None, t=None):
        if t is not None:
            self.update_grid(t)
        if idx is not None:
            X = self.std_at_grid[idx]
            return np.sum(X * X)
        else:
            return np.sum(self.std_at_grid * self.std_at_grid)

    def evaluate_unc(self, idx=None, t=None):
        if t is not None:
            self.update_grid(t)
        if idx is not None:
            X = self.std_at_grid[idx]
            return np.mean(X)
        else:
            return np.mean(self.std_at_grid)

    def evaluate_mutual_info(self, t):
        n_sample = self.grid.shape[0]
        _, cov = self.gp.predict(add_t(self.grid, t), return_cov=True)
        mi = (1 / 2) * np.log(
            np.linalg.det(
                0.01 * cov.reshape(n_sample, n_sample) + np.identity(n_sample)
            )
        )
        return mi

    def get_high_info_area(self, curr_t=0, t=ADAPTIVE_TH, beta=1):
        x1 = np.linspace(0, 1, self.env_size)
        x2 = np.linspace(0, 1, self.env_size)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(add_t(x1x2, curr_t), return_std=True)

        high_measurement_area = []
        for i in range(900):
            if y_pred[i] + beta * std[i] >= t:
                high_measurement_area.append(x1x2[i])
        high_measurement_area = np.array(high_measurement_area)
        if high_measurement_area.shape[0] == 0:
            warnings.warn("No high information area found")
            # print('y_pred is ', y_pred)
        # print("----> number of elements in high_measurement_area: ", high_measurement_area.shape)
        return high_measurement_area

    def evaluate_KL_div(self, y_true, t=None, norm=True, base=None):
        if t is not None:
            self.update_grid(t)
        y_pred = copy.deepcopy(self.y_pred_at_grid)
        y_pred[y_pred < 0] = 0
        P = np.array(y_true) + 1e-8
        Q = np.array(y_pred).reshape(-1) + 1e-8
        if norm:
            P /= np.sum(P, axis=0, keepdims=True)
            Q /= np.sum(Q, axis=0, keepdims=True)
        vec = P * np.log(P / Q)
        S = np.sum(vec, axis=0)
        if base is not None:
            S /= np.log(base)
        return S

    def evaluate_JS_div(self, y_true, t=None, norm=True):
        if t is not None:
            self.update_grid(t)
        y_pred = copy.deepcopy(self.y_pred_at_grid)
        y_pred[y_pred < 0] = 0
        P = np.array(y_true) + 1e-8
        Q = np.array(y_pred).reshape(-1) + 1e-8
        if norm:
            P /= np.sum(P, axis=0, keepdims=True)
            Q /= np.sum(Q, axis=0, keepdims=True)
        M = 0.5 * (P + Q)
        vec_PM = P * np.log(P / M)
        vec_QM = Q * np.log(Q / M)
        KL_PM = np.sum(vec_PM, axis=0)
        KL_QM = np.sum(vec_QM, axis=0)
        JS = 0.5 * (KL_PM + KL_QM)
        return JS

    # def plot(self, y_true, target_id, target_num, target_loc=None, all_pred=None, high_idx=None, agent_loc=None):
    #     y_true = y_true.reshape(30, 30, -1)
    #     X0p, X1p = self.grid[:, 0].reshape(30, 30), self.grid[:, 1].reshape(30, 30)
    #     y_pred = self.y_pred_at_grid.reshape(30, 30)
    #     std = self.std_at_grid.reshape(30, 30)
    #     target_cmap = ['r', 'g', 'b', 'm', 'y', 'c', 'lightcoral', 'lightgreen', 'lightblue', 'orange', 'gold', 'pink']

    #     if target_id == 0:
    #         plt.subplot(2, target_num+1, target_num+3)  # ground truth
    #         plt.title('Ground truth')
    #         plt.xlim((0, 1)); plt.ylim((0, 1))
    #         plt.pcolormesh(X0p, X1p, y_true.max(axis=-1), shading='auto', vmin=0, vmax=1)
    #         if high_idx is not None:
    #             high_info_area = [self.grid[high_idx[i]] for i in range(target_num)]
    #             for i in range(target_num):
    #                 plt.scatter(high_info_area[i][:, 0], high_info_area[i][:, 1], s=0.5, c=target_cmap[i], alpha=0.5)
    #         plt.subplot(2, target_num+1, target_num+1)  # stddev
    #         plt.title(f'Target {target_id} std')
    #         plt.pcolormesh(X0p, X1p, std, shading='auto', vmin=0, vmax=1)
    #     plt.subplot(2, target_num+1, target_id+1)  # mean
    #     plt.title(f'Target {target_id} mean')
    #     plt.pcolormesh(X0p, X1p, y_pred, shading='auto', vmin=0, vmax=1)
    #     if target_loc[target_id] is not None:
    #         plt.scatter(*target_loc[target_id], c=target_cmap[target_id], s=10, marker='s')
    #     if target_id+1 == target_num:
    #         all_pred.append(y_pred)
    #         all_pred = np.asarray(all_pred).max(axis=0)
    #         plt.subplot(2, target_num+1, target_num+2)  # all mean
    #         plt.title('All targets')
    #         plt.xlim((0, 1)); plt.ylim((0, 1))
    #         plt.pcolormesh(X0p, X1p, all_pred, shading='auto', vmin=0, vmax=1)
    #         if (np.linalg.norm(target_loc-agent_loc, axis=1) < 0.1).any():
    #             fov = plt.Circle(agent_loc, 0.1, color='y', fill=False)
    #         else:
    #             fov = plt.Circle(agent_loc, 0.1, color='c', fill=False)
    #         plt.gca().add_patch(fov)
    #     return y_pred

    def plot(self, y_true, curr_t=0, attention_weights=None):
        y_pred, std = self.gp.predict(add_t(self.grid, curr_t), return_std=True)

        X0p, X1p = self.grid[:, 0].reshape(self.env_size, self.env_size), self.grid[
            :, 1
        ].reshape(self.env_size, self.env_size)
        y_pred = np.reshape(y_pred, (self.env_size, self.env_size))
        std = std.reshape(self.env_size, self.env_size)

        X = np.array(self.observed_points)

        fig = plt.figure(figsize=(6, 6))
        # if self.observed_points:
        #    plt.scatter(X[:, 0].reshape(1, -1), X[:, 1].reshape(1, -1), s=10, c='r')
        plt.subplot(2, 2, 2)  # ground truth
        plt.title("Ground truth")
        fig.colorbar(
            plt.pcolormesh(
                X0p,
                X1p,
                y_true.reshape(self.env_size, self.env_size),
                shading="auto",
                vmin=0,
                # vmax=y_true.max(),
                vmax=1,
            )
        )
        plt.subplot(2, 2, 3)  # stddev
        plt.title("Predict std")
        fig.colorbar(
            plt.pcolormesh(
                # X0p, X1p, std, shading="auto", vmin=std.min(), vmax=std.max()
                X0p,
                X1p,
                std,
                shading="auto",
                vmin=0.0,
                vmax=1.0,
            )
        )
        plt.subplot(2, 2, 1)  # mean
        plt.title("Predict mean")
        fig.colorbar(
            plt.pcolormesh(
                # X0p, X1p, y_pred, shading="auto", vmin=y_pred.min(), vmax=y_pred.max()
                X0p,
                X1p,
                y_pred,
                shading="auto",
                vmin=0.0,
                vmax=1.0,
            )
        )

        plt.subplot(2, 2, 4)  # absolute error
        plt.title("Absolute error")
        fig.colorbar(
            plt.pcolormesh(
                X0p,
                X1p,
                np.abs(y_true.reshape(self.env_size, self.env_size) - y_pred),
                shading="auto",
                vmin=0,
                vmax=1,
            )
        )

        plt.xlim((0, 1))
        plt.ylim((0, 1))
        # if self.observed_points:
        #     plt.scatter(X[:, 0].reshape(1, -1), X[:, 1].reshape(1, -1), s=2, c='r')
        # plt.show()
        return y_pred


class GaussianProcessWrapper:
    def __init__(self, num_gp, node_coords):
        self.num_gp = num_gp
        self.node_coords = node_coords
        self.GPs = [GaussianProcess(self.node_coords[i]) for i in range(self.num_gp)]
        self.curr_t = None
        self.env_size = 30

    def add_init_measures(self, all_point_pos):
        for i, gp in enumerate(self.GPs):
            gp.add_observed_point(all_point_pos[i].reshape(-1, 3), 1.0)

    def add_observed_points(self, point_pos, values):  # value: (1, n)
        try:
            for i, gp in enumerate(self.GPs):
                gp.add_observed_point(point_pos, values[0, i])
        except:
            for i, gp in enumerate(self.GPs):
                gp.add_observed_point(point_pos, values)

    def update_gps(self):
        for gp in self.GPs:
            gp.update_gp()

    def update_node_feature(self, t, agent_id=None):
        self._update_curr_t(t, agent_id)
        self.update_grids()
        node_pred, node_std = None, None
        node_info, node_info_future = [], []  # (target, node, 2)

        def process_node(gp: GaussianProcess, t):
            node_pred, node_std = gp.update_node(t)
            node_pred_future, node_std_future = gp.update_node(t + 2)
            node_info = [np.hstack((node_pred.reshape(-1, 1), node_std.reshape(-1, 1)))]
            node_info_future = [
                np.hstack(
                    (node_pred_future.reshape(-1, 1), node_std_future.reshape(-1, 1))
                )
            ]
            node_feature = np.concatenate(
                (np.asarray(node_info), np.asarray(node_info_future)), axis=-1
            )  # (target, node, features(4))
            node_feature = node_feature.transpose((1, 0, 2)).reshape(
                self.node_coords[0].shape[0], -1
            )  # (node, (targetxfeature))
            return node_feature

        if agent_id is not None:
            gp = self.GPs[agent_id]
            if type(t) == list:
                return process_node(gp, t[agent_id])
            else:
                return process_node(gp, t)
        else:
            all_node_features = []
            # print('self.GPs is ', len(self.GPs))
            for i, gp in enumerate(self.GPs):
                node_feature = process_node(gp, t[i])
                # print(">> node_feature is ", node_feature.shape)
                all_node_features.append(node_feature)
            return all_node_features

        # node_feature = np.concatenate(
        #     (np.asarray(node_info), np.asarray(node_info_future)), axis=-1
        # )  # (target, node, features(4))
        # node_feature = node_feature.transpose((1, 0, 2)).reshape(
        #     self.node_coords.shape[0], -1
        # )  # (node, (targetxfeature))

        # contiguous at feature level
        # og, og_future = node_feature[:, :2], node_feature[:, 2:]
        # og_pred, og_std = og[:, 0], og[:, 1]
        # print("testing-----------node feature")
        # print('Node_pred is ', node_pred)
        # print('og_pred is ', og_pred)
        # print('Node_std is ', node_std)
        # print('og_std is ', og_std)
        # print("................node feature is ", node_feature.shape)
        # quit()

        return node_feature

    def update_grids(self):
        for i, gp in enumerate(self.GPs):
            gp.update_grid(self.curr_t[i])

    def return_grid(self):
        env_grid_mean = []
        env_grid_std = []
        for gp in self.GPs:
            env_grid_mean.append(
                gp.y_pred_at_grid.reshape(self.env_size, self.env_size)
            )
            env_grid_std.append(gp.std_at_grid.reshape(self.env_size, self.env_size))
        return env_grid_mean[0], env_grid_std[0]

    def _update_curr_t(self, t, agent_id):
        """Helper method to update current time consistently"""
        if self.curr_t is None:
            if type(t) == list:
                self.curr_t = t.copy()
            else:
                self.curr_t = [t] * self.num_gp
        
        if agent_id is not None:
            if type(t) == list:
                self.curr_t[agent_id] = t[agent_id]
            else:
                self.curr_t[agent_id] = t
        else:
            if type(t) == list:
                self.curr_t = t.copy()
            else:
                self.curr_t = [t] * self.num_gp

    def _process_single_metric(self, metric_func, t, agent_id=None, **kwargs):
        """Generic helper for processing metrics"""
        self._update_curr_t(t, agent_id)
        self.update_grids()

        if agent_id is not None:
            result = [metric_func(self.GPs[agent_id], **kwargs)]
        else:
            result = [metric_func(gp, **kwargs) for gp in self.GPs]

        avg_result = (
            np.mean(result)
            if metric_func.__name__ != "evaluate_RMSE"
            else np.sqrt(np.mean(np.square(result)))
        )
        return (avg_result, result) if kwargs.get("return_all", False) else result

    def eval_avg_RMSE(self, y_true, t, agent_id=None, return_all=False):
        return self._process_single_metric(
            lambda gp, **kwargs: gp.evaluate_RMSE(y_true),
            t,
            agent_id,
            return_all=return_all,
        )

    def eval_avg_cov_trace(
        self, t, high_info_idx=None, agent_id=None, return_all=False
    ):
        def get_trace(gp, **kwargs):
            idx = None if high_info_idx is None else high_info_idx[self.GPs.index(gp)]
            return gp.evaluate_cov_trace(idx)

        return self._process_single_metric(
            get_trace, t, agent_id, return_all=return_all
        )

    def eval_avg_unc(self, t, high_info_idx=None, agent_id=None, return_all=False):
        def get_unc(gp, **kwargs):
            idx = None if high_info_idx is None else high_info_idx[self.GPs.index(gp)]
            return gp.evaluate_unc(idx)

        return self._process_single_metric(get_unc, t, agent_id, return_all=return_all)

    def eval_avg_unc_sum(
        self, unc, high_info_idx=None, agent_id=None, return_all=False
    ):
        if high_info_idx is None:
            num_high = [self.env_size * self.env_size] * len(self.GPs)
        else:
            num_high = list(map(len, high_info_idx))

        if agent_id is not None:
            std_sum = [unc[agent_id] * num_high[agent_id]]
        else:
            std_sum = [unc[i] * num_high[i] for i in range(len(self.GPs))]

        avg_std_sum = np.mean(std_sum)
        return (avg_std_sum, std_sum) if return_all else std_sum

    def eval_avg_KL(self, y_true, t, agent_id=None, return_all=False):
        return self._process_single_metric(
            lambda gp, **kwargs: gp.evaluate_KL_div(y_true),
            t,
            agent_id,
            return_all=return_all,
        )

    def eval_avg_JS(self, y_true, t, agent_id=None, return_all=False):
        return self._process_single_metric(
            lambda gp, **kwargs: gp.evaluate_JS_div(y_true),
            t,
            agent_id,
            return_all=return_all,
        )

    def eval_avg_F1(self, y_true, t, agent_id=None, return_all=False):
        return self._process_single_metric(
            lambda gp, **kwargs: gp.evaluate_F1score(y_true, self.curr_t),
            t,
            agent_id,
            return_all=return_all,
        )

    def eval_avg_MI(self, t, agent_id=None, return_all=False):
        return self._process_single_metric(
            lambda gp, **kwargs: gp.evaluate_mutual_info(self.curr_t),
            t,
            agent_id,
            return_all=return_all,
        )

    def get_high_info_area(self, curr_t, adaptive_t, return_all=False):
        high_info_area = []
        for gp in self.GPs:
            high_info_area += [gp.get_high_info_area(curr_t=curr_t, t=adaptive_t)]
        return high_info_area

    def plot(self, y_true, curr_t=0, attention_weights=None):
        for gp in self.GPs:
            gp.plot(y_true, curr_t, attention_weights)


if __name__ == "__main__":
    pass
