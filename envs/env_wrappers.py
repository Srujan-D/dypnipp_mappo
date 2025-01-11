"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np

# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.share_observation_space
        self.action_space = self.env.action_space
        self.actions = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        
        obs, rews, dones, infos = [], [], [], []
        for result in results:
            obs.append(result[0])
            rews.append(result[1])
            dones.append(result[2])
            infos.append(result[3])
        
        # breakpoint()
        obs = np.array(obs)
        rews = np.array(rews)
        dones = np.array(dones)

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()[2]
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()[2]

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        node_coords, graph, node_feature, budget = [], [], [], []
        for env in self.envs:
            node_coord, g, node_feat, b = env.reset()
            node_coords.append(node_coord)
            graph.append(g)
            node_feature.append(node_feat)
            budget.append(b)
        #  [env.reset() for env in self.envs] # [env_num, agent_num, obs_dim]
        return node_coords, graph, node_feature, budget

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError