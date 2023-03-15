from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import SAC
from pettingzoo.test import api_test, parallel_api_test
from pettingzoo.utils import parallel_to_aec, wrappers
import time
import copy
import sys
DIRECTORY = "/home/lisa/Mount/Dateien/StudyProject"
sys.path.insert(0, f"{DIRECTORY}/s.mujoco_environment/Gym")
'''
@ToDo: Peter said something about this being more flexible. may also work with the DIRECTORY setting???? 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
'''
from single_agent import SingleAgent
import torch as th
from ray import tune
from ray.rllib.algorithms.a3c import A3CConfig
from Pick_Up_Dynamic import Pick_Up_Dynamic

"""
WARNING!!!!
This file is used only for testing the purpose of the environment.
"""

def test_reward(mujoco_gym, model, data) -> float:
    """
    Implementation of the test reward function.
    It contains two parts:
    1. The agent gets a reward for moving towards the target
    2. The agent gets a reward for moving at all
    Both rewards are equally weighted.

    Parameters:
        mujoco_gym (SingleAgent): instance of single agent environment

    Returns:
        reward (float): reward for the agent
    """
    distance = mujoco_gym.calculate_distance("torso", mujoco_gym.data_store["current_target"])
    if "distance" not in mujoco_gym.data_store.keys():
        mujoco_gym.data_store["distance"] = distance
        new_reward = 0
    else:
        new_reward = mujoco_gym.data_store["distance"] - distance
        mujoco_gym.data_store["distance"] = distance
    reward = new_reward

    if "last_position" not in mujoco_gym.data_store.keys():
        mujoco_gym.data_store["last_position"] = copy.deepcopy(mujoco_gym.data.body("torso").xipos)
        new_reward = 0
    else:
        new_reward = mujoco_gym.calculate_distance("torso", mujoco_gym.data_store["last_position"])
        mujoco_gym.data_store["last_position"] = copy.deepcopy(mujoco_gym.data.body("torso").xipos)
        if new_reward < 0.08:
            new_reward = new_reward * -1
        new_reward = new_reward * 6
    reward = reward + new_reward
    return reward

def train():
    """
    Train a single agent with soft actor critic (SAC) to pick up a target.
    """
    env = SingleAgent(f"{DIRECTORY}/s.mujoco_environment/Environments/single_agent/ModelVis.xml", infoJson=f"{DIRECTORY}/s.mujoco_environment/Environments/single_agent/info_example.json", render=False, print_camera_config=False, add_target_coordinates=False, add_agent_coordinates=True, end_epoch_on_turn=True, env_dynamics=[Pick_Up_Dynamic], reward_function=test_reward, max_step=8192, use_ctrl_cost=False)
    print("env created")
    layer = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[1024, 512, 256], vf=[1024, 512, 256])])
    policy_kwargs = dict(net_arch=dict(pi=[4096, 2048, 1024], qf=[4096, 2048, 1024]))
    model = SAC("MlpPolicy", env, verbose=1, train_freq=(128, "step"), batch_size=128, learning_starts=100000, learning_rate=0.0015, buffer_size=1500000, policy_kwargs=policy_kwargs)
    print("model created")
    model.learn(total_timesteps=1, progress_bar=True)
    print("model trained")
    model.save("models/sac_model")

def train_ray():
    config = A3CConfig()
    env = SingleAgent(f"{DIRECTORY}/s.mujoco_environment/Environments/single_agent/ModelVis.xml", infoJson=f"{DIRECTORY}/s.mujoco_environment/Environments/single_agent/info_example.json", render=False, print_camera_config=False, add_target_coordinates=False, add_agent_coordinates=True, end_epoch_on_turn=True, env_dynamics=[pick_up_dynamic], reward_function=test_reward, max_step=8192, use_ctrl_cost=False)
    config = config.training(gamma=0.9, lr=0.01)
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=4)
    print(config.to_dict())
    algo = config.build(env=env)
    algo.train()

def infer():
    model = SAC.load("models/sac_model")
    env = SingleAgent("envs/ModelVis.xml", infoJson="envs/info_example.json", render=True, print_camera_config=False, add_target_coordinates=False, add_agent_coordinates=True, end_epoch_on_turn=True, env_dynamics=[pick_up_dynamic], reward_function=test_reward, max_step=8192, use_ctrl_cost=False)
    obs = env.reset()
    reward = 0
    for i in range(512):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action, False)
        if dones:
            print(reward)
            break
        reward += rewards
        time.sleep(0.1)
        env.render()
    env.reset()
    env.end()

if __name__ == "__main__":
    train()
    