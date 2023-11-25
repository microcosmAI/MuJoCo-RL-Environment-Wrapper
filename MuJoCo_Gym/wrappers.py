try:
    from mujoco_rl import MuJoCoRL  # Used during development
except:
    from MuJoCo_Gym.mujoco_rl import MuJoCoRL # Used as a pip package

import gym
import gym.spaces.box as gymBox
import gymnasium
from gymnasium.spaces.box import Box


class GymnasiumWrapper(gymnasium.Env):
    """
    A wrapper class for integrating a MuJoCoRL environment into the Gymnasium framework.

    Args:
        environment (MuJoCoRL): The MuJoCoRL environment to be wrapped.
        agent (str): The name of the agent in the environment.
        render_mode (str, optional): The rendering mode for the environment. Defaults to "none".

    Attributes:
        metadata (dict): Metadata for the environment, including render modes and render fps.
        observation_space (gym.Space): The observation space of the environment.
        action_space (gym.Space): The action space of the environment.

    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 4}

    def __init__(self, environment: MuJoCoRL, agent: str, render_mode="none") -> None:
        super().__init__()
        self.environment = environment
        self.agent = agent
        self.render_mode = render_mode

        if len(self.environment.agents) > 1:
            raise Exception("Environment has too many agents. Only one agent is allowed in a gym environment.")

        self.observation_space = Box(high=environment.observation_space.high, low=environment.observation_space.low)
        self.action_space = Box(high=environment.action_space.high, low=environment.action_space.low)

    def step(self, action):
        """
        Perform a step in the environment.

        Args:
            action: The action to be taken in the environment.

        Returns:
            observation (np.ndarray): The observation after taking the action.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode is done after taking the action.
            info (dict): Additional information about the step.

        """
        action = {self.agent: action}
        observations, rewards, terminations, truncations, infos = self.environment.step(action)
        termination = terminations["__all__"]
        truncation = truncations["__all__"]
        
        return observations[self.agent], rewards[self.agent], termination, truncation, infos[self.agent]

    def reset(self, *, seed=1, options={}):
        """
        Reset the environment.

        Args:
            seed (int, optional): The seed for the environment's random number generator. Defaults to 1.
            options (dict, optional): Additional options for resetting the environment. Defaults to {}.

        Returns:
            observation (np.ndarray): The initial observation after resetting the environment.

        """
        observations, infos = self.environment.reset()
        return observations[self.agent]
    
    def render(self):
        """
        Render the environment.

        """
        pass

class GymWrapper(gym.Env):
    """
    A Gym environment wrapper for MuJoCoRL.

    Args:
        environment (MuJoCoRL): The MuJoCoRL environment.
        agent (str): The agent name.
        render_mode (str, optional): The render mode. Defaults to "none".

    Attributes:
        metadata (dict): Metadata for the environment.
        observation_space (gym.spaces.Box): The observation space.
        action_space (gym.spaces.Box): The action space.

    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 4}

    def __init__(self, environment: MuJoCoRL, agent: str, render_mode="none") -> None:
        super().__init__()
        self.environment = environment
        self.agent = agent
        self.render_mode = render_mode

        if len(self.environment.agents) > 1:
            raise Exception("Environment has too many agents. Only one agent is allowed in a gym environment.")

        self.observation_space = gymBox.Box(high=environment.observation_space.high, low=environment.observation_space.low)
        self.action_space = gymBox.Box(high=environment.action_space.high, low=environment.action_space.low)

    def step(self, action):
        """
        Perform a step in the environment.

        Args:
            action: The action to take.

        Returns:
            tuple: A tuple containing the observation, reward, done flag, and info.

        """
        action = {self.agent: action}
        observations, rewards, terminations, truncations, infos = self.environment.step(action)
        if terminations["__all__"] or truncations["__all__"]:
            done = True
        else:
            done = False
        
        return observations[self.agent], rewards[self.agent], done, infos[self.agent]
    
    def reset(self):
        """
        Reset the environment.

        Returns:
            object: The initial observation.

        """
        observations, infos = self.environment.reset()
        return observations[self.agent]