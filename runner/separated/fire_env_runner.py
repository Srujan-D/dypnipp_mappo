import time
import os
import numpy as np
import torch
from itertools import chain
from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        self.warmup()
        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Collect actions and values
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Step the environment
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # Insert data into buffer
                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.insert(data)

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
        obs = self.envs.reset()
        share_obs = np.array([list(chain(*o)) for o in obs])

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values, actions, temp_actions_env, action_log_probs = [], [], [], []
        rnn_states, rnn_states_critic = [], []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            breakpoint()
            # Get actions and values from policy
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            values.append(_t2n(value))
            actions.append(_t2n(action))
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

            # Format actions for the environment
            action_env = self.format_action(action, agent_id)
            temp_actions_env.append(action_env)

        # Combine actions for all threads
        actions_env = [list(a) for a in zip(*temp_actions_env)]
        return (
            np.array(values).transpose(1, 0, 2),
            np.array(actions).transpose(1, 0, 2),
            np.array(action_log_probs).transpose(1, 0, 2),
            np.array(rnn_states).transpose(1, 0, 2, 3),
            np.array(rnn_states_critic).transpose(1, 0, 2, 3),
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

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        # Reset RNN states for done episodes
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # Share observations
        share_obs = np.array([list(chain(*o)) for o in obs])

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )
