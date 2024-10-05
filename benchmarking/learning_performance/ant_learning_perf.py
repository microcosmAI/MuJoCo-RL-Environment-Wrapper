import numpy as np
from MuJoCo_Gym.mujoco_rl import MuJoCoRL


def ant_reward_function(env, agent):
    """
    Custom reward function that mimics the reward function of Gym's Ant environment.
    Args:
        env: The environment instance.
        agent: The agent ID.
    Returns:
        float: The reward for the agent.
    """
    # Extract the position and velocity of the agent
    xpos_before = env.data_store[agent].get('xpos_before', None)
    xpos_after = env.get_data(agent)['position'][0]

    if xpos_before is None:
        env.data_store[agent]['xpos_before'] = xpos_after
        return 0

    # Extract time step duration from the environment's MuJoCo model
    dt = env.model.opt.timestep

    # Calculate the forward reward
    forward_reward = (xpos_after - xpos_before) / dt

    # Calculate the control cost
    control_cost = 0.5 * np.square(env.data.ctrl).sum()

    # Calculate the contact cost
    contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(env.data.cfrc_ext, -1, 1)))

    # Calculate the total reward
    reward = forward_reward - control_cost - contact_cost

    # Update the position for the next step
    env.data_store[agent]['xpos_before'] = xpos_after

    return reward


def get_config(xml_file, agents=None):
    if agents is None:
        agents = ["torso"]
    return {
        "xmlPath": xml_file,
        "agents": agents,
        "rewardFunctions": [ant_reward_function],
        "doneFunctions": [],
        "skipFrames": 5,
        "environmentDynamics": [],
        "freeJoint": False,
        "renderMode": False,
        "maxSteps": 1024,
        "agentCameras": True
    }


# Initialize the environment with the custom reward function
env = MuJoCoRL(config_dict=get_config('/Users/cowolff/Documents/GitHub/s.mujoco_environment/benchmarking/levels/Ant.xml', ['torso']))

# Print environment details
print("Environment details:")
print(f"Action space: {env.action_space('torso')}")
print(f"Observation space: {env.observation_space('torso')}")

# Define a basic training loop
num_episodes = 50
num_steps = 1024

for episode in range(num_episodes):
    observations, infos = env.reset()
    total_reward = 0
    for step in range(num_steps):
        action = env.action_space('torso').sample()  # Replace with a learned policy for actual training
        observations, rewards, terminations, truncations, infos = env.step({"torso": action})

        # print observation shape
        print(observations["torso"])
        exit()
        total_reward += rewards['torso']
        if terminations['torso']:
            break
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")


'''

import numpy as np
from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces


def ant_reward_function(env, agent):
    """
    Custom reward function that mimics the reward function of Gym's Ant environment.
    Args:
        env: The environment instance.
        agent: The agent ID.
    Returns:
        float: The reward for the agent.
    """
    xpos_before = env.data_store[agent].get('xpos_before', None)
    xpos_after = env.get_data(agent)['position'][0]

    if xpos_before is None:
        env.data_store[agent]['xpos_before'] = xpos_after
        return 0

    dt = env.model.opt.timestep
    forward_reward = (xpos_after - xpos_before) / dt
    control_cost = 0.5 * np.square(env.data.ctrl).sum()
    contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(env.data.cfrc_ext, -1, 1)))
    reward = forward_reward - control_cost - contact_cost
    env.data_store[agent]['xpos_before'] = xpos_after

    return reward


def get_config(xml_file, agents=None):
    if agents is None:
        agents = ["torso"]
    return {
        "xmlPath": xml_file,
        "agents": agents,
        "rewardFunctions": [ant_reward_function],
        "doneFunctions": [],
        "skipFrames": 5,
        "environmentDynamics": [],
        "freeJoint": False,
        "renderMode": False,
        "maxSteps": 1024,
        "agentCameras": True
    }


# Initialize the environment with the custom reward function
env = MuJoCoRL(config_dict=get_config('../levels/Ant.xml', ['torso']))


class CustomEnvWrapper(gym.Env):
    """
    Custom Environment that wraps the MuJoCoRL environment to be compatible with stable-baselines3
    """

    def __init__(self, env):
        super(CustomEnvWrapper, self).__init__()
        self.env = env
        self.action_space = self.env.action_space('torso')
        # print(self.action_space)
        # Define the correct observation space
        obs_shape = self.env.observation_space('torso').shape
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    def reset(self, seed=None):
        obs, infos = self.env.reset()
        obs = np.array(obs['torso'], dtype=np.float32)
        return obs, infos

    def step(self, action):
        obs, rewards, done, truncations, infos = self.env.step({"torso": action})
        # print(truncations)
        obs = np.array(obs['torso'], dtype=np.float32)
        return obs, rewards['torso'], done['torso'], truncations['torso'], infos['torso']


# Wrap the environment
custom_env = CustomEnvWrapper(env)
vec_env = DummyVecEnv([lambda: custom_env])

# Optional: Check the environment
check_env(custom_env, warn=True)

# Ensure the observation space is correctly defined
print("Observation space:", custom_env.observation_space)
print("Sample observation:", custom_env.reset()[0])

# Initialize the RL algorithm
model = PPO('MlpPolicy', vec_env, verbose=1)

# Train the agent
model.learn(total_timesteps=1000000)

# Save the model
model.save("ppo_ant")

# Load the model
model = PPO.load("ppo_ant")

# Evaluate the agent
obs = vec_env.reset()
for i in range(1024):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = vec_env.step(action)
    vec_env.render()

'''