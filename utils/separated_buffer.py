import torch
import numpy as np


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer:
    def __init__(self, args, obs_shape, share_obs_shape, act_space, agent_ID):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        self.sample_size = args.sample_size
        self.k_size = args.k_size
        self.embedding_dim = args.embedding_dim

        self.agent_num = args.agent_num

        self.agent_ID = agent_ID

        # Actor-specific storage
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )
        self.node_inputs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.sample_size + 2, 4),
            dtype=np.float32,
        )
        self.edge_inputs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.sample_size + 2,
                self.k_size,
            ),
            dtype=np.int32,
        )
        self.budget_inputs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.sample_size + 2, 1),
            dtype=np.float32,
        )
        self.current_index = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1, 1), dtype=np.int32
        )
        self.pos_encoding = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.sample_size + 2, 32),
            dtype=np.float32,
        )
        self.rnn_states_actor = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                2,  # 2 for LSTM hidden state (h) and cell state (c)
                1,
                self.embedding_dim,
            ),
            dtype=np.float32,
        )

        # Critic-specific storage
        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *share_obs_shape),
            dtype=np.float32,
        )

        self.global_node_inputs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.agent_num,
                self.sample_size + 2,
                4,
            ),
            dtype=np.float32,
        )
        self.global_edge_inputs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.agent_num,
                self.sample_size + 2,
                self.k_size,
            ),
            dtype=np.int32,
        )
        self.global_budget_inputs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.agent_num,
                self.sample_size + 2,
                1,
            ),
            dtype=np.float32,
        )
        self.global_current_index = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.agent_num, 1, 1),
            dtype=np.int32,
        )
        self.global_pos_encoding = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.agent_num,
                self.sample_size + 2,
                32,
            ),
            dtype=np.float32,
        )
        self.global_masks = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.agent_num,
                self.sample_size + 2,
                self.k_size,
            ),
            dtype=np.float32,
        )

        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        # Rewards, masks, and other attributes
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.masks = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.sample_size + 2,
                self.k_size,
            ),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),  # act_space["n"]),
            dtype=np.float32,
        )
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), #act_space["n"]),
            dtype=np.float32,
        )

        self.ppo_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.bad_masks = np.ones_like(self.ppo_masks)
        self.active_masks = np.ones_like(self.ppo_masks)

        if act_space["type"] == "Discrete":
            self.available_actions = np.zeros(
                (self.episode_length + 1, self.n_rollout_threads, act_space["n"]),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        self.step = 0

    def insert(
        self,
        obs,
        share_obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        ppo_masks=None,
        bad_masks=None,
        active_masks=None,
        available_actions=None,
    ):
        self.obs[self.step + 1] = obs.copy()
        self.share_obs[self.step + 1] = share_obs.copy()
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        if ppo_masks is not None:
            self.ppo_masks[self.step + 1] = ppo_masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states_actor[0] = self.rnn_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.ppo_masks[0] = self.ppo_masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    # def compute_returns(self, next_value):
    #     self.value_preds[-1] = next_value
    #     gae = 0
    #     for step in reversed(range(self.rewards.shape[0])):
    #         delta = (
    #             self.rewards[step]
    #             + self.gamma * self.value_preds[step + 1] * self.masks[step + 1]
    #             - self.value_preds[step]
    #         )
    #         gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
    #         self.returns[step] = gae + self.value_preds[step]
    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.ppo_masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * self.ppo_masks[step + 1]
                            * gae
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.ppo_masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * self.ppo_masks[step + 1]
                            * gae
                        )
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart:
                        self.returns[step] = (
                            self.returns[step + 1]
                            * self.gamma
                            * self.ppo_masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        self.returns[step] = (
                            self.returns[step + 1]
                            * self.gamma
                            * self.ppo_masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.ppo_masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * self.ppo_masks[step + 1]
                            * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.ppo_masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * self.ppo_masks[step + 1]
                            * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.ppo_masks[step + 1]
                        + self.rewards[step]
                    )

    def calculate_position_embedding(self, edge_inputs):
        A_matrix = np.zeros((self.sample_size + 2, self.sample_size + 2))
        D_matrix = np.zeros((self.sample_size + 2, self.sample_size + 2))
        for i in range(self.sample_size + 2):
            for j in range(self.sample_size + 2):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        for i in range(self.sample_size + 2):
            D_matrix[i][i] = 1 / np.sqrt(len(edge_inputs[i]) - 1)
        L = np.eye(self.sample_size + 2) - np.matmul(D_matrix, A_matrix, D_matrix)
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_vector = np.real(eigen_vector[:, idx])
        return eigen_vector[:, 1 : 32 + 1]

    def feed_forward_generator(
        self, advantages, num_mini_batch=None, mini_batch_size=None
    ):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(
                    n_rollout_threads,
                    episode_length,
                    n_rollout_threads * episode_length,
                    num_mini_batch,
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states_actor = self.rnn_states_actor[:-1].reshape(
            -1, *self.rnn_states_actor.shape[2:]
        )
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                -1, self.available_actions.shape[-1]
            )
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        ppo_masks = self.ppo_masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, self.action_log_probs.shape[-1]
        )
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states_actor[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            ppo_masks_batch = ppo_masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, ppo_masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch)
        )
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind])
                obs_batch.append(self.obs[:-1, ind])
                rnn_states_batch.append(self.rnn_states_actor[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch, 1).reshape(
                N, *self.rnn_states_actor.shape[2:]
            )
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch, 1).reshape(
                N, *self.rnn_states_critic.shape[2:]
            )

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(
                n_rollout_threads, episode_length, data_chunk_length
            )
        )
        assert data_chunks >= 2, "need larger batch size"

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        # Transform the input data
        node_inputs = (
            self.node_inputs[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.node_inputs.shape[2:])
        )
        edge_inputs = (
            self.edge_inputs[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.edge_inputs.shape[2:])
        )
        budget_inputs = (
            self.budget_inputs[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.budget_inputs.shape[2:])
        )
        current_index = (
            self.current_index[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.current_index.shape[2:])
        )
        pos_encoding = (
            self.pos_encoding[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.pos_encoding.shape[2:])
        )

        global_node_inputs = (
            self.global_node_inputs[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.global_node_inputs.shape[2:])
        )
        global_edge_inputs = (
            self.global_edge_inputs[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.global_edge_inputs.shape[2:])
        )
        global_budget_inputs = (
            self.global_budget_inputs[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.global_budget_inputs.shape[2:])
        )
        global_current_index = (
            self.global_current_index[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.global_current_index.shape[2:])
        )
        global_pos_encoding = (
            self.global_pos_encoding[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.global_pos_encoding.shape[2:])
        )

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = self.masks[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.masks.shape[2:])
        global_masks = (
            self.global_masks[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.masks.shape[2:])
        )
        ppo_masks = _cast(self.ppo_masks[:-1])
        active_masks = _cast(self.active_masks[:-1])

        rnn_states_actor = (
            self.rnn_states_actor[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.rnn_states_actor.shape[2:])
        )
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            node_inputs_batch = []
            edge_inputs_batch = []
            budget_inputs_batch = []
            current_index_batch = []
            pos_encoding_batch = []
            global_node_inputs_batch = []
            global_edge_inputs_batch = []
            global_budget_inputs_batch = []
            global_current_index_batch = []
            global_pos_encoding_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            global_masks_batch = []
            ppo_masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                node_inputs_batch.append(node_inputs[ind : ind + data_chunk_length])
                edge_inputs_batch.append(edge_inputs[ind : ind + data_chunk_length])
                budget_inputs_batch.append(budget_inputs[ind : ind + data_chunk_length])
                current_index_batch.append(current_index[ind : ind + data_chunk_length])
                pos_encoding_batch.append(pos_encoding[ind : ind + data_chunk_length])

                global_node_inputs_batch.append(
                    global_node_inputs[ind : ind + data_chunk_length]
                )
                global_edge_inputs_batch.append(
                    global_edge_inputs[ind : ind + data_chunk_length]
                )
                global_budget_inputs_batch.append(
                    global_budget_inputs[ind : ind + data_chunk_length]
                )
                global_current_index_batch.append(
                    global_current_index[ind : ind + data_chunk_length]
                )
                global_pos_encoding_batch.append(
                    global_pos_encoding[ind : ind + data_chunk_length]
                )

                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(
                        available_actions[ind : ind + data_chunk_length]
                    )
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                global_masks_batch.append(global_masks[ind : ind + data_chunk_length])
                ppo_masks_batch.append(ppo_masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(
                    action_log_probs[ind : ind + data_chunk_length]
                )
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                rnn_states_batch.append(rnn_states_actor[ind : ind + data_chunk_length])
                rnn_states_critic_batch.append(
                    rnn_states_critic[ind : ind + data_chunk_length]
                )

            L, N = data_chunk_length, mini_batch_size

            # Stack all batches
            node_inputs_batch = np.stack(node_inputs_batch)
            edge_inputs_batch = np.stack(edge_inputs_batch)
            budget_inputs_batch = np.stack(budget_inputs_batch)
            current_index_batch = np.stack(current_index_batch)
            pos_encoding_batch = np.stack(pos_encoding_batch)

            global_node_inputs_batch = np.stack(global_node_inputs_batch)
            global_edge_inputs_batch = np.stack(global_edge_inputs_batch)
            global_budget_inputs_batch = np.stack(global_budget_inputs_batch)
            global_current_index_batch = np.stack(global_current_index_batch)
            global_pos_encoding_batch = np.stack(global_pos_encoding_batch)

            actions_batch = np.stack(actions_batch)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch)
            value_preds_batch = np.stack(value_preds_batch)
            return_batch = np.stack(return_batch)
            ppo_masks_batch = np.stack(ppo_masks_batch)
            masks_batch = np.stack(masks_batch)
            global_masks_batch = np.stack(global_masks_batch)
            active_masks_batch = np.stack(active_masks_batch)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch)
            adv_targ = np.stack(adv_targ)

            rnn_states_batch = np.stack(rnn_states_batch)
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch)

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            node_inputs_batch = _flatten(L, N, node_inputs_batch)
            edge_inputs_batch = _flatten(L, N, edge_inputs_batch)
            budget_inputs_batch = _flatten(L, N, budget_inputs_batch)
            current_index_batch = _flatten(L, N, current_index_batch)
            pos_encoding_batch = _flatten(L, N, pos_encoding_batch)

            global_node_inputs_batch = _flatten(L, N, global_node_inputs_batch)
            global_edge_inputs_batch = _flatten(L, N, global_edge_inputs_batch)
            global_budget_inputs_batch = _flatten(L, N, global_budget_inputs_batch)
            global_current_index_batch = _flatten(L, N, global_current_index_batch)
            global_pos_encoding_batch = _flatten(L, N, global_pos_encoding_batch)

            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            ppo_masks_batch = _flatten(L, N, ppo_masks_batch)
            masks_batch = _flatten(L, N, masks_batch)
            global_masks_batch = _flatten(L, N, global_masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            rnn_states_batch = _flatten(L, N, rnn_states_batch)
            rnn_states_critic_batch = _flatten(L, N, rnn_states_critic_batch)

            yield node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_index_batch, pos_encoding_batch, global_node_inputs_batch, global_edge_inputs_batch, global_budget_inputs_batch, global_current_index_batch, global_pos_encoding_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, global_masks_batch, ppo_masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
