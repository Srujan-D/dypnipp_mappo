import time
import os
import numpy as np
import torch
from itertools import chain
from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner


def _t2n(x):
    if type(x) == list:
        return [i.detach().cpu().numpy() for i in x]
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def compute(self):
        # TODO: Check if _combine_inputs is necessary -- storing global_* vars in buffer
        (
            global_node_inputs,
            global_edge_inputs,
            global_budget_inputs,
            global_pos_encodings,
            global_current_index,
            global_masks,
        ) = self._combine_inputs(self.buffer, -1)
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                torch.FloatTensor(global_node_inputs).to(self.device),
                torch.FloatTensor(global_edge_inputs).to(self.device),
                torch.FloatTensor(global_budget_inputs).to(self.device),
                torch.FloatTensor(global_current_index).to(self.device),
                torch.tensor(
                    self.buffer[agent_id].rnn_states_critic[-1].transpose(1, 0, 2, 3)
                ).to(self.device),
                torch.tensor(global_pos_encodings).to(self.device),
                torch.tensor(global_masks).to(self.device),
                # global_node_inputs[-1],
                # global_edge_inputs[-1],
                # global_budget_inputs[-1],
                # global_current_index[-1],
                # self.buffer[agent_id].rnn_states_critic[-1],
                # global_pos_encodings[-1],
                # self.buffer[agent_id].masks[-1],
                # self.buffer[agent_id].next_belief[-1],
            )
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(
                next_value, self.trainer[agent_id].value_normalizer
            )

    def run(self):
        self.warmup()
        start = time.time()
        # breakpoint()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Collect actions and values
                # print(">>> step", step)
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)
                # breakpoint()
                # Step the environment
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # Process new observations
                for thread_id in range(self.n_rollout_threads):
                    for agent_id in range(self.num_agents):
                        self.process_obs(
                            obs[thread_id][agent_id], obs, thread_id, agent_id, step + 1
                        )

                # Insert data into buffer
                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                )
                self.insert(data)

            print("Running episode", episode, " w rewards", rewards)
            # Compute returns and train the model
            self.compute()
            train_infos = self.train()

            # Log and save periodically
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    f"\n Scenario {self.all_args.scenario_name} "
                    f"Algo {self.algorithm_name} Exp {self.experiment_name} "
                    f"updates {episode}/{episodes} episodes, total num timesteps "
                    f"{total_num_steps}/{self.num_env_steps}, FPS {int(total_num_steps / (end - start))}.\n"
                )
                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        """
        Initialize buffers and process the first observation.
        """
        node_coords, graph, node_feature, budget = self.envs.reset()
        obs = node_feature  # obs is [n_rollout_threads][n_agents][...features...]

        # Process initial observations
        for thread_id in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                self.process_obs(obs[thread_id][agent_id], obs, thread_id, agent_id, 0)

    def process_obs(self, obs, share_obs, thread_id, agent_id, step):
        """
        Process node feature and graph data for a specific agent and thread, and update the buffer.
        """

        node_feature = obs  # For this specific thread and agent
        node_coords = self.envs.env.node_coords[agent_id]
        graph = self.envs.env.graph[agent_id]
        budget = self.envs.env.budget[agent_id]

        node_info = node_feature[:, 0]  # Predictions
        node_std = node_feature[:, 1]  # Uncertainty

        # Node and budget inputs
        n_nodes = node_coords.shape[0]
        node_info_inputs = node_info.reshape((n_nodes, 1))
        node_std_inputs = node_std.reshape((n_nodes, 1))
        node_inputs = np.concatenate(
            (node_coords, node_info_inputs, node_std_inputs), axis=1
        )
        budget_inputs = self.envs.env.env.calc_estimate_budget(agent_id, budget, 1)

        # Graph edges
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, graph[node]))
            edge_inputs.append(node_edges)

        # Position encodings
        pos_encoding = self.calculate_position_embedding(edge_inputs)

        # Share observations (combine for critic)
        # share_obs = list(chain(*all_obs))
        # share_obs = all_obs

        # Update buffer
        self.buffer[agent_id].node_inputs[step, thread_id] = node_inputs
        self.buffer[agent_id].edge_inputs[step, thread_id] = edge_inputs
        self.buffer[agent_id].budget_inputs[step, thread_id] = budget_inputs
        self.buffer[agent_id].pos_encoding[step, thread_id] = pos_encoding

        share_obs = np.concatenate(share_obs[thread_id])
        self.buffer[agent_id].share_obs[step, thread_id] = share_obs
        self.buffer[agent_id].obs[step, thread_id] = obs
        # TODO: Figure out what current_index should be stored as by referring to dypnipp_sarl code
        self.buffer[agent_id].current_index[step, thread_id] = np.zeros(
            (1,), dtype=np.int32
        )

        self.buffer[agent_id].global_node_inputs[
            step, thread_id, agent_id
        ] = node_inputs
        self.buffer[agent_id].global_edge_inputs[
            step, thread_id, agent_id
        ] = edge_inputs
        self.buffer[agent_id].global_budget_inputs[
            step, thread_id, agent_id
        ] = budget_inputs
        self.buffer[agent_id].global_pos_encoding[
            step, thread_id, agent_id
        ] = pos_encoding
        # TODO: Figure out what current_index should be stored as by referring to dypnipp_sarl code
        self.buffer[agent_id].global_current_index[step, thread_id, agent_id] = (
            np.zeros((1,), dtype=np.int32)
        )

    # def insert(self, data):
    #     """
    #     Inserts the data for the current step into the buffer.
    #     """
    #     (
    #         obs,
    #         rewards,
    #         dones,
    #         infos,
    #         values,
    #         actions,
    #         action_log_probs,
    #         rnn_states_actor,
    #         rnn_states_critic,
    #     ) = data
    #     print(">>> Inserting data")
    #     try:
    #         print("rnn_states_actor", rnn_states_actor.shape)
    #         print("rnn_states_critic", rnn_states_critic.shape)
    #     except:
    #         breakpoint()
    #     # Reset RNN states for done episodes
    #     for thread_id in range(self.n_rollout_threads):
    #         if any(dones[thread_id]):
    #             rnn_states_actor[thread_id] = np.zeros(
    #                 (self.recurrent_N, self.embedding_dim), dtype=np.float32
    #             )
    #             rnn_states_critic[thread_id] = np.zeros(
    #                 (self.recurrent_N, self.embedding_dim), dtype=np.float32
    #             )

    #     masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    #     masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

    #     for thread_id in range(self.n_rollout_threads):
    #         for agent_id in range(self.num_agents):
    #             self.buffer[agent_id].insert(
    #                 obs[thread_id][agent_id],
    #                 list(chain(*obs[thread_id])),
    #                 self.buffer[agent_id].node_inputs[self.buffer[agent_id].step, thread_id],
    #                 self.buffer[agent_id].edge_inputs[self.buffer[agent_id].step, thread_id],
    #                 self.buffer[agent_id].budget_inputs[self.buffer[agent_id].step, thread_id],
    #                 self.buffer[agent_id].current_index[self.buffer[agent_id].step, thread_id],
    #                 self.buffer[agent_id].pos_encoding[self.buffer[agent_id].step, thread_id],
    #                 rnn_states_actor[thread_id][agent_id],
    #                 rnn_states_critic[thread_id][agent_id],
    #                 actions[thread_id][agent_id],
    #                 action_log_probs[thread_id][agent_id],
    #                 values[thread_id][agent_id],
    #                 rewards[thread_id][agent_id],
    #                 masks[thread_id][agent_id],
    #             )

    def insert(self, data):
        """
        Inserts the data for the current step into the buffer.
        """
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states_actor,
            rnn_states_critic,
        ) = data
        # print(">>> Inserting data")

        # Reshape/reset RNN states for done episodes
        # breakpoint()
        # Reset RNN states for done episodes
        # # Reshape if needed since shape should be [agent_id, thread_id, 2, 1, embedding_dim]
        # if not isinstance(rnn_states_actor, np.ndarray):
        #     rnn_states_actor = np.array(rnn_states_actor)
        #     rnn_states_critic = np.array(rnn_states_critic)

        for thread_id in range(self.n_rollout_threads):
            if any(dones[thread_id]):
                for agent_id in range(self.num_agents):
                    rnn_states_actor[agent_id][thread_id] = np.zeros(
                        (2, 1, self.embedding_dim), dtype=np.float32
                    )
                    rnn_states_critic[agent_id][thread_id] = np.zeros(
                        (2, 1, self.embedding_dim), dtype=np.float32
                    )

        # Prepare masks
        # TODO: Check if masks are initialized correctly by referring to dypnipp_sarl code
        masks = np.ones(
            (
                self.n_rollout_threads,
                self.all_args.sample_size + 2,
                self.all_args.k_size,
            ),
            dtype=np.float32,
        )
        for thread_id in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                if dones[thread_id][agent_id]:
                    masks[thread_id, :, :] = 0.0

        # Insert data for each agent, passing all threads at once
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                obs=obs[:, agent_id],  # [n_threads, obs_dim]
                share_obs=np.array(
                    [np.concatenate(o) for o in obs]
                ),  # [n_threads, total_obs_dim]
                rnn_states_actor=rnn_states_actor[agent_id],  # [n_threads, ...]
                rnn_states_critic=rnn_states_critic[agent_id],  # [n_threads, ...]
                actions=actions[:, agent_id],  # [n_threads, action_dim]
                action_log_probs=action_log_probs[:, agent_id],  # [n_threads, 1]
                value_preds=values[:, agent_id],  # [n_threads, 1]
                rewards=rewards[:, agent_id],  # [n_threads, 1]
                masks=masks,  # [n_threads, mask_dim]
            )

    @torch.no_grad()
    def collect(self, step):
        values, actions, temp_actions_env, action_log_probs = [], [], [], []
        rnn_states_actor, rnn_states_critic = [], []

        (
            global_node_inputs,
            global_edge_inputs,
            global_budget_inputs,
            global_pos_encodings,
            global_current_index,
            global_masks,
        ) = self._combine_inputs(self.buffer, step)

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()

            # Get actions and values from policy
            (
                value,
                action,
                action_log_prob,
                rnn_actor_state,
                rnn_critic_state,
            ) = self.trainer[agent_id].policy.get_actions(
                torch.FloatTensor(self.buffer[agent_id].node_inputs[step]).to(
                    self.device
                ),
                torch.FloatTensor(self.buffer[agent_id].edge_inputs[step]).to(
                    self.device
                ),
                torch.FloatTensor(self.buffer[agent_id].budget_inputs[step]).to(
                    self.device
                ),
                torch.tensor(
                    self.buffer[agent_id].current_index[step], dtype=torch.int64
                ).to(self.device),
                torch.tensor(self.buffer[agent_id].rnn_states_actor[step]).to(
                    self.device
                ),
                torch.tensor(self.buffer[agent_id].rnn_states_critic[step]).to(
                    self.device
                ),
                torch.from_numpy(self.buffer[agent_id].pos_encoding[step])
                .float()
                .to(self.device),
                torch.FloatTensor(global_node_inputs).to(self.device),
                torch.FloatTensor(global_edge_inputs).to(self.device),
                torch.FloatTensor(global_budget_inputs).to(self.device),
                torch.FloatTensor(global_current_index).to(self.device),
                torch.tensor(global_pos_encodings).to(self.device),
                torch.tensor(self.buffer[agent_id].masks[step]).to(self.device),
                torch.tensor(global_masks).to(self.device),
                # self.buffer[agent_id].next_belief[step],
            )

            # breakpoint()

            values.append(_t2n(value))
            actions.append(_t2n(action))
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states_actor.append(_t2n(rnn_actor_state))
            rnn_states_critic.append(_t2n(rnn_critic_state))

            # Format actions for the environment
            action_env = self.format_action(action, agent_id)
            temp_actions_env.append(action_env)

        # Combine actions for all threads
        actions_env = [list(a) for a in zip(*temp_actions_env)]

        # values = np.array(values)

        actions = np.array(actions)
        actions = actions[..., None]
        # action_log_probs = np.array(action_log_probs)
        # rnn_states_actor = np.array(rnn_states_actor)
        # rnn_states_critic = np.array(rnn_states_critic)

        return (
            np.array(values).transpose(1, 0, 2),
            actions.transpose(1, 0, 2),
            np.array(action_log_probs).transpose(1, 0, 2),
            np.array(rnn_states_actor).transpose(0, 2, 1, 3, 4),
            np.array(rnn_states_critic).transpose(0, 2, 1, 3, 4),
            actions_env,
        )

    def format_action(self, action, agent_id):
        # Format actions based on action space type
        if self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
            return np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
        elif self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
            formatted_action = []
            for i in range(self.envs.action_space[agent_id].shape):
                formatted_action.append(
                    np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                )
            return np.concatenate(formatted_action, axis=1)
        else:
            # Continuous action space or custom actions
            return action

    def calculate_position_embedding(self, edge_inputs):
        """
        Calculate positional encodings for the nodes in the graph.
        """
        sample_size = len(edge_inputs)
        A_matrix = np.zeros((sample_size, sample_size))
        D_matrix = np.zeros((sample_size, sample_size))
        for i, edges in enumerate(edge_inputs):
            for j in edges:
                A_matrix[i, j] = 1
        for i in range(sample_size):
            D_matrix[i, i] = 1 / np.sqrt(len(edge_inputs[i]))
        L = np.eye(sample_size) - np.matmul(D_matrix, A_matrix)
        _, eigen_vector = np.linalg.eig(L)
        eigen_vector = np.real(eigen_vector[:, :32])
        return eigen_vector

    def _combine_inputs(self, buffer, step):
        """
        Combine inputs for the critic.
        """
        try:
            # Combine node inputs across agents
            global_node_inputs = np.concatenate(
                [
                    buffer[agent_id].node_inputs[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, node_features)

            # Combine edge inputs across agents
            global_edge_inputs = np.concatenate(
                [
                    buffer[agent_id].edge_inputs[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, k_size)

            # Combine budget inputs across agents
            global_budget_inputs = np.concatenate(
                [
                    buffer[agent_id].budget_inputs[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, 1)

            # Combine position encodings across agents
            global_pos_encodings = np.concatenate(
                [
                    buffer[agent_id].pos_encoding[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, pos_encoding_dim)

            global_current_index = np.concatenate(
                [
                    buffer[agent_id].current_index[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )

            global_masks = np.concatenate(
                [buffer[agent_id].masks[step] for agent_id in range(self.num_agents)],
                axis=1,
            )
        except:
            breakpoint()
            # Combine node inputs across agents
            global_node_inputs = np.concatenate(
                [
                    buffer[agent_id].node_inputs[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, node_features)

            # Combine edge inputs across agents
            global_edge_inputs = np.concatenate(
                [
                    buffer[agent_id].edge_inputs[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, k_size)

            # Combine budget inputs across agents
            global_budget_inputs = np.concatenate(
                [
                    buffer[agent_id].budget_inputs[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, 1)

            # Combine position encodings across agents
            global_pos_encodings = np.concatenate(
                [
                    buffer[agent_id].pos_encoding[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )  # Shape: (n_rollout_threads, sample_size + 2 * num_agents, pos_encoding_dim)

            global_current_index = np.concatenate(
                [
                    buffer[agent_id].current_index[step]
                    for agent_id in range(self.num_agents)
                ],
                axis=1,
            )

            global_masks = np.concatenate(
                [buffer[agent_id].masks[step] for agent_id in range(self.num_agents)],
                axis=1,
            )

        return (
            global_node_inputs,
            global_edge_inputs,
            global_budget_inputs,
            global_pos_encodings,
            global_current_index,
            global_masks,
        )
