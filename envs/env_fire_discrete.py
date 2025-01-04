import numpy as np
from gym import spaces
from envs.env_fire import EnvCore


class DiscreteActionEnv:
    """
    Wrapper for discrete action environments integrated with EnvCore and FireCommander.
    """

    def __init__(self, config):
        """
        Initialize the environment.
        :param config: Configuration dictionary for EnvCore initialization.
        """
        self.env = EnvCore(
            sample_size=config.get("sample_size", 500),
            k_size=config.get("k_size", 10),
            start=config.get("start", None),
            destination=config.get("destination", None),
            obstacle=config.get("obstacle", []),
            budget_range=config.get("budget_range", (7, 9)),
            save_image=config.get("save_image", False),
            seed=config.get("seed", None),
            fixed_env=config.get("fixed_env", None),
            agent_num=config.get("agent_num", 3),
            fuel=config.get("fuel", None),
            env_size=config.get("env_size", 30),
        )

        self.num_agents = self.env.agent_num
        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # Define observation and action spaces as simple dictionaries
        self.observation_space = [
            {"shape": (self.signal_obs_dim, 4), "dtype": np.float32}
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            {"shape": (self.num_agents * self.signal_obs_dim, 4), "dtype": np.float32}
            for _ in range(self.num_agents)
        ]
        self.action_space = [
            {"type": "Discrete", "n": self.signal_action_dim}
            for _ in range(self.num_agents)
        ]
        
    def step(self, actions):
        """
        Perform a step in the environment for all agents.
        :param actions: List of actions for each agent.
        :return: Tuple (obs, rews, dones, infos) with batched results.
        """
        if not isinstance(actions, (list, np.ndarray)):
            raise ValueError("Actions must be a list or numpy array.")

        # Perform the step in EnvCore
        obs, rews, dones, infos = self.env.step(actions)

        # Ensure compatibility with gym-like environments
        obs = np.stack(obs)
        rews = np.stack(rews)
        dones = np.stack(dones)

        return obs, rews, dones, infos

    def reset(self):
        """
        Reset the environment.
        :return: Initial observations for all agents.
        """
        obs = self.env.reset()[2]
        # breakpoint()
        # print("obs------------------------", len(obs), np.stack(obs).shape)
        return np.stack(obs)

    def close(self):
        """Close the environment."""
        pass

    def render(self, mode="rgb_array"):
        """
        Render the environment. Placeholder implementation.
        :param mode: Rendering mode, e.g., 'rgb_array' or 'human'.
        """
        if mode == "rgb_array":
            print("Rendering not implemented.")
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported.")

    def seed(self, seed):
        """
        Set the random seed for reproducibility.
        :param seed: Seed value.
        """
        self.env.reset(seed=seed)


if __name__ == "__main__":
    # Example configuration
    config = {
        "sample_size": 500,
        "k_size": 10,
        "start": None,
        "destination": None,
        "obstacle": [],
        "budget_range": None,
        "save_image": False,
        "seed": 42,
        "fixed_env": None,
        "agent_num": 2,
        "fuel": None,
        "env_size": 30,
    }

    # Initialize environment
    env = DiscreteActionEnv(config)
    obs = env.reset()
    print("Initial observations:", obs)

    # Take a step
    actions = [env.action_space[i].sample() for i in range(env.num_agents)]
    print("Sampled actions:", actions)

    obs, rews, dones, infos = env.step(actions)
    print("Step results:")
    print("Observations:", obs)
    print("Rewards:", rews)
    print("Dones:", dones)
    print("Infos:", infos)
