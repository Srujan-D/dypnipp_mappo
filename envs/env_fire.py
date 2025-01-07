import numpy as np
from numba import jit
from scipy.ndimage import gaussian_filter

from .fire_commander.Fire_2D import FireCommanderExtreme as Fire
from .gaussian_process.gp_st import GaussianProcessWrapper, add_t
from .classes import PRMController, Utils

import copy


class FireEnvCore(object):
    """
    # FireCommander Sim Wrapper
    """

    def __init__(
        self,
        sample_size=500,
        k_size=10,
        start=None,
        destination=None,
        obstacle=[],
        budget_range=None,
        save_image=False,
        seed=None,
        fixed_env=None,
        agent_num=1,
        fuel=None,
        env_size=30,
    ):

        self._graph_sample_size = sample_size
        self._graph_k_size = k_size

        self._obstacle = obstacle
        self._budget_range = budget_range
        self.budget = self.budget_init = np.random.uniform(*self._budget_range)
        self._save_image = save_image
        self._seed = seed

        self._fixed_env = fixed_env
        self.agent_num = agent_num
        self.fuel = fuel

        if start is None:
            self.start = np.random.rand(self.agent_num, 2)
        else:
            if len(start) == 1:
                self.start = np.array([start[0]] * self.agent_num)
            else:
                self.start = np.array(start)

        if destination is None:
            self.destination = np.random.rand(self.agent_num, 2)
        else:
            if len(destination) == 1:
                self.destination = np.array([destination[0]] * self.agent_num)
            else:
                self.destination = np.array(destination)
        
        self._world_size = env_size

        self._start_Fire()
        self._generate_Graph(
            agent_num=agent_num
        )  # TODO: Generate multiple graphs for multiple agents, OR change PRMController to support/return graphs for multiple agents

        # underlying distribution
        self.underlying_distribution = None
        self.ground_truth = None

        self.obs_dim = 502  # 2D coordinates
        self.action_dim = self._graph_k_size  # 1D action space

        # GP
        self.gp_wrapper = None
        self.node_feature = None
        # self.gp_ipp = None
        self.node_info, self.node_std = None, None
        self.node_info0, self.node_std0, self.budget0 = copy.deepcopy(
            (self.node_info, self.node_std, self.budget)
        )
        self.JS, self.JS_init, self.JS_list, self.KL, self.KL_init, self.KL_list = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.cov_trace, self.cov_trace_init = None, None
        self.unc, self.unc_list, self.unc_init, self.unc_sum, self.unc_sum_list = (
            None,
            None,
            None,
            None,
            None,
        )
        self.RMSE = None
        self.F1score = None
        self.MI = None
        self.MI0 = None

        # start point
        self.current_node_index = 1  # 1  # 0 used in STAMP
        self.sample = self.start
        self.dist_residual = 0
        self.route = []

        self.save_image = save_image
        self.frame_files = []

        self.env_sim_params = self.fire.fire_params

    def _start_Fire(self, seed=None):
        self.fire = Fire(
            world_size=self._world_size,
            start=self.start,
            seed=self._seed if seed is None else seed,
            fuel=self.fuel,
        )
        self.fire.env_init()

    def _generate_Graph(self, agent_num=1):
        # TODO: Generate multiple graphs for multiple agents, OR change PRMController to support/return graphs for multiple agents
        self.graph_PRM = [
            PRMController.PRMController(
                self._graph_sample_size,
                self._obstacle,
                self.start[i],
                self.destination[i],
                self._graph_k_size,
            )
            for i in range(agent_num)
        ]
        self.node_coords = []
        self.graph = []
        for prm in self.graph_PRM:
            node_coords, graph = prm.runPRM(saveImage=False, seed=self._seed)
            self.node_coords.append(node_coords)
            self.graph.append(graph)

    def reset(self, seed=None):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self._seed)

        self._start_Fire(seed)
        self.underlying_distribution = self.fire

        self.ground_truth = self.get_ground_truth()
        self.ground_truth = gaussian_filter(self.ground_truth, sigma=1.5)

        # initialize gp
        self.curr_t = 0.0
        self.gp_wrapper = GaussianProcessWrapper(self.agent_num, self.node_coords)
        self.node_feature = self.gp_wrapper.update_node_feature(self.curr_t)

        self.RMSE = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, self.curr_t)
        self.cov_trace = self.gp_wrapper.eval_avg_cov_trace(self.curr_t)
        self.unc, self.unc_list = self.gp_wrapper.eval_avg_unc(
            self.curr_t, return_all=True
        )
        self.JS, self.JS_list = self.gp_wrapper.eval_avg_JS(
            self.ground_truth, self.curr_t, return_all=True
        )
        self.KL, self.KL_list = self.gp_wrapper.eval_avg_KL(
            self.ground_truth, self.curr_t, return_all=True
        )
        self.unc_sum, self.unc_sum_list = self.gp_wrapper.eval_avg_unc_sum(
            self.unc_list, return_all=True
        )
        self.JS_init = self.JS
        self.KL_init = self.KL
        self.cov_trace_init = self.cov_trace
        self.unc_init = self.unc
        self.budget = self.budget_init

        # start point
        self.current_node_index = 1  # 1
        self.dist_residual = 0
        self.sample = self.start
        # self.random_speed_factor = np.random.rand()
        self.route = []

        # sub_agent_obs = []
        # for i in range(self.agent_num):
        #     # sub_obs = [
        #     #     self.node_coords[i],
        #     #     self.graph[i],
        #     #     self.node_feature[i],
        #     #     self.budget,
        #     # ]
        #     sub_obs = self.node_feature[i]
        #     # breakpoint()
        #     sub_agent_obs.append(sub_obs)
        return self.node_coords, self.graph, self.node_feature, self.budget

    def step(self, actions):
        """
        Perform a single global timestep where all agents act simultaneously.
        Actions: List[[next_node_index, sample_length, measurement]]
        """
        # Initialize outputs for all agents
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        # Update the fire environment for the current timestep (global update)
        self.update_fire()

        # Parallel processing of agents
        for i in range(self.agent_num):
            next_node_index, sample_length, measurement = actions[i]

            # Compute movement and sampling
            dist = np.linalg.norm(
                self.node_coords[self.current_node_index[i]]
                - self.node_coords[next_node_index]
            )
            remain_length = dist
            next_length = sample_length - self.dist_residual[i]
            reward = 0
            done = next_node_index == 0
            no_sample = True

            while remain_length > next_length:
                # Compute the sample position
                if no_sample:
                    self.sample[i] = (
                        self.node_coords[next_node_index]
                        - self.node_coords[self.current_node_index[i]]
                    ) * next_length / dist + self.node_coords[
                        self.current_node_index[i]
                    ]
                else:
                    self.sample[i] = (
                        self.node_coords[next_node_index]
                        - self.node_coords[self.current_node_index[i]]
                    ) * next_length / dist + self.sample[i]

                # Take measurements based on the latest fire state
                if measurement:
                    observed_value = (
                        self.underlying_distribution.return_fire_at_location(
                            self.sample[i].reshape(-1, 2)[0], fire_map=self.fire_map
                        )
                    )
                else:
                    observed_value = np.array([0])

                # Update time and distances
                self.curr_t[i] += next_length
                remain_length -= next_length
                next_length = sample_length
                no_sample = False

                self.underlying_distribution.single_agent_state_update(
                    self.sample[i].reshape(-1, 2)[0], agent_id=i
                )

                # Update the GP with the new observed value
                self.gp_wrapper.GPs[i].add_observed_point(
                    add_t(self.sample[i].reshape(-1, 2), self.curr_t[i]), observed_value
                )

            # Update GP and metrics when the agent reaches the next node
            self.gp_wrapper.GPs[i].update()
            self.dist_residual[i] = remain_length if no_sample else 0.0

            # Compute metrics
            actual_t = self.curr_t[i]
            actual_budget = self.budget[i] - dist
            self.node_feature[i] = self.gp_wrapper.update_node_feature(actual_t, agent_id=i)
            self.RMSE[i] = self.gp_wrapper.eval_avg_RMSE(self.ground_truth, actual_t)
            cov_trace = self.gp_wrapper.eval_avg_cov_trace(actual_t)
            unc, unc_list = self.gp_wrapper.eval_avg_unc(actual_t, return_all=True)
            JS, JS_list = self.gp_wrapper.eval_avg_JS(
                self.ground_truth, actual_t, return_all=True
            )
            KL, KL_list = self.gp_wrapper.eval_avg_KL(
                self.ground_truth, actual_t, return_all=True
            )

            # Reward logic
            if next_node_index in self.route[i][-2:]:
                reward += -0.1
            elif self.cov_trace[i] > cov_trace:
                reward += (self.cov_trace[i] - cov_trace) / self.cov_trace[i]
            self.cov_trace[i] = cov_trace
            if done:
                reward -= cov_trace / 900

            # Update agent state
            self.route[i].append(next_node_index)
            self.current_node_index[i] = next_node_index
            if not done and actual_budget <= 0.0005:
                done = True

            # Append outputs for the agent
            sub_agent_obs.append(self.node_feature[i])  # Only features go into observations
            sub_agent_reward.append([reward])
            sub_agent_done.append(done)
            sub_agent_info.append({"actual_budget": actual_budget})  # Store scalar in info

            # Return the updated structure
            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


    def update_fire(self):
        # Fire updates independently of agents' actions
        fire_state, _, _, _, _, interp_fire_intensity = self.fire.env_step(r_func="RF4")
        self.set_momentum_GT(fire_map=interp_fire_intensity)

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def calc_fire_intensity_in_field(fire_map, world_size=100):
        field_intensity = np.zeros((world_size, world_size))

        for i in range(fire_map.shape[0]):
            x, y, intensity = fire_map[i]
            m = min(int(x), field_intensity.shape[0] - 1)
            n = min(int(y), field_intensity.shape[1] - 1)

            if field_intensity[m, n] > 0:
                field_intensity[m, n] = max(field_intensity[m, n], intensity)
            else:
                field_intensity[m, n] = intensity

        not_on_fire = np.ones((world_size, world_size))

        for i in range(world_size):
            for j in range(world_size):
                if field_intensity[i, j] > 0:
                    not_on_fire[i, j] = 0

        on_fire_indices = np.nonzero(
            field_intensity
        )  # Get indices of points that are on fire

        for i in range(not_on_fire.shape[0]):
            for j in range(not_on_fire.shape[1]):
                if not_on_fire[i, j] == 1:  # If the point is not on fire
                    neighboring_points = []
                    for k in range(len(on_fire_indices[0])):
                        x_, y_ = on_fire_indices[0][k], on_fire_indices[1][k]
                        distance_squared = (x_ - i) ** 2 + (y_ - j) ** 2
                        if distance_squared <= 2:
                            neighboring_points.append(field_intensity[x_, y_])

                    if len(neighboring_points) != 0:
                        field_intensity[i, j] = np.mean(np.array(neighboring_points))

        return field_intensity

    def get_ground_truth(self, scale=1):
        scale = self.underlying_distribution.world_size

        ground_truth = Fire.calc_fire_intensity_in_field(
            self.underlying_distribution.fire_map,
            self.underlying_distribution.world_size,
        )
        # ground_truth = self.underlying_distribution.fire_map[:, 2]
        ground_truth = Utils.Utils.compress_and_average(
            array=ground_truth, new_shape=(30, 30)
        )
        ground_truth = ground_truth.reshape(-1)

        return ground_truth / ground_truth.max()

    def set_ground_truth(self, fire_map):
        # print(">>> Max fire intensity: ", np.max(fire_map))
        # time1 = time.time()
        ground_truth = Utils.Utils.compress_and_average(array=fire_map, new_shape=(30, 30))
        # time2 = time.time()
        # print(f">>> Time to compress and average: {time2 - time1:.4f}")
        # print(">>> Max fire intensity after compression: ", np.max(ground_truth))
        self.ground_truth = ground_truth.reshape(-1)
        del ground_truth

    def set_momentum_GT(self, fire_map):
        ground_truth = Utils.Utils.compress_and_average(array=fire_map, new_shape=(30, 30))
        # new GT = 0.9 * old GT + 0.1 * new GT
        self.ground_truth = 0.9 * self.ground_truth + 0.1 * ground_truth.reshape(-1)
        del ground_truth
