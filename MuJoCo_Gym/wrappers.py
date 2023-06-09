try:
    from mujoco_rl import MuJoCo_RL # Used during development
except:
    from MuJoCo_Gym.mujoco_rl import MuJoCo_RL # Used as a pip package

import gym
import gym.spaces.box as gymBox
from gymnasium.spaces.box import Box
import gymnasium

class GymnasiumWrapper(gymnasium.Env):

    metadata = {"render_modes": ["human", "none"], "render_fps": 4}

    def __init__(self, environment: MuJoCo_RL, agent: str, render_mode="none") -> None:
        super().__init__()
        self.environment = environment
        self.agent = agent
        self.render_mode = render_mode

        if len(self.environment.agents) > 1:
            raise Exception("Environment has too many agents. Only one agent is allowed in a gym environment.")

        self.observation_space = Box(high=environment.observation_space.high, low=environment.observation_space.low)
        self.action_space = Box(high=environment.action_space.high, low=environment.action_space.low)

    def step(self, action):
        action = {self.agent: action}
        observations, rewards, terminations, truncations, infos = self.environment.step(action)
        termination = terminations["__all__"]
        truncation = truncations["__all__"]
        
        return observations[self.agent], rewards[self.agent], termination, truncation, infos[self.agent]

    def reset(self, *, seed=1, options={}):
        observations, infos = self.environment.reset()
        return observations[self.agent]
    
    def render():
        pass

class GymWrapper(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 4}

    def __init__(self, environment: MuJoCo_RL, agent: str, render_mode="none") -> None:
        super().__init__()
        self.environment = environment
        self.agent = agent
        self.render_mode = render_mode

        if len(self.environment.agents) > 1:
            raise Exception("Environment has too many agents. Only one agent is allowed in a gym environment.")

        self.observation_space = gymBox.Box(high=environment.observation_space.high, low=environment.observation_space.low)
        self.action_space = gymBox.Box(high=environment.action_space.high, low=environment.action_space.low)

    def step(self, action):
        action = {self.agent: action}
        observations, rewards, terminations, truncations, infos = self.environment.step(action)
        if terminations["__all__"] or truncations["__all__"]:
            done = True
        else:
            done = False
        
        return observations[self.agent], rewards[self.agent], done, infos[self.agent]