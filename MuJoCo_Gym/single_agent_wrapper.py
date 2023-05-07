try:
    from mujoco_rl import MuJoCo_RL # Used during development
except:
    from MuJoCo_Gym.mujoco_rl import MuJoCo_RL # Used as a pip package
import gym
from gym.spaces.box import Box

class Single_Agent_Wrapper(gym.Env):

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
        if terminations["__all__"] or truncations["__all__"]:
            done = True
        else:
            done = False
        return observations[self.agent], rewards[self.agent], done, infos[self.agent]

    def reset(self):
        observations, infos = self.environment.reset()
        return observations[self.agent]