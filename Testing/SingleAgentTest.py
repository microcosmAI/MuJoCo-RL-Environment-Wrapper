from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import SAC
import supersuit as ss
from pettingzoo.test import api_test, parallel_api_test
from pettingzoo.utils import parallel_to_aec, wrappers
import random
import time
import math
import copy
import numpy as np
import sys
sys.path.insert(0, "/Users/cowolff/Documents/GitHub/s.mujoco_environment/Gym")
from single_agent import SingleAgent
import torch as th
from ray import tune
from ray.rllib.algorithms.a3c import A3CConfig

"""
WARNING!!!!
This file is used only for testing the purpose of the environment.
"""

def reward(mujoco_gym, data, model) -> float:
    """
    Calculates the reward based only on the agent's distance to the target
    Parameters:
        agent (str): name of the agent
        target (str): name of the target
    Returns:
        reward (float): reward for the agent
    """
    distance = math.dist(data.body("torso").xipos, data.body("target").xipos)
    if mujoco_gym.lastDistance is None:
        mujoco_gym.lastDistance = distance
        reward = 0
    else:
        reward = mujoco_gym.lastDistance - distance
        mujoco_gym.lastDistance = distance
    if mujoco_gym.use_head_sensor:
        if mujoco_gym.data.sensordata.flat[4] < 0.15:
            reward = reward - 0.01
    return reward * 10

def test_reward(mujoco_gym, data, model) -> float:
    """
    Implementation of the test reward function.
    It contains two parts:
    1. The agent gets a reward for moving towards the target
    2. The agent gets a reward for moving at all
    Both rewards are equally weighted.
    Parameters:
        agent (str): name of the agent
        target (str): name of the target
    Returns:
        reward (float): reward for the agent
    """
    reward = 0
    distance = math.dist(data.body("torso").xipos, data.body(mujoco_gym.data_store["current_target"]).xipos)
    if "distance" not in mujoco_gym.data_store.keys():
        mujoco_gym.data_store["distance"] = distance
        new_reward = 0
    else:
        new_reward = mujoco_gym.data_store["distance"] - distance
        mujoco_gym.data_store["distance"] = distance
        new_reward * 10
    reward = reward + new_reward

    if "last_position" not in mujoco_gym.data_store.keys():
        mujoco_gym.data_store["last_position"] = copy.deepcopy(data.body("torso").xipos)
        new_reward = 0
    else:
        new_reward = math.dist(mujoco_gym.data_store["last_position"], data.body("torso").xipos)
        mujoco_gym.data_store["last_position"] = copy.deepcopy(data.body("torso").xipos)
        new_reward = new_reward * 10
        if new_reward < 0.8:
            new_reward = new_reward * -1
    reward = reward + (new_reward * 0.6)
    return reward

def environment_dynamic(mujoco_gym, data, model):
    """
    Update Target position if the agent is close enough
    """
    if "targets" not in mujoco_gym.data_store.keys():
        mujoco_gym.data_store["targets"] = []
        for body_index in range(model.nbody):
            if "target" in model.body(body_index).name:
                mujoco_gym.data_store["targets"].append(model.body(body_index).name)
        mujoco_gym.data_store["current_target"] = mujoco_gym.data_store["targets"][random.randint(0, len(mujoco_gym.data_store["targets"]) - 1)]
    distance = math.dist(data.body("torso").xipos, data.body(mujoco_gym.data_store["current_target"]).xipos)
    if distance < 1:
        mujoco_gym.data_store["current_target"] = mujoco_gym.data_store["targets"][random.randint(0, len(mujoco_gym.data_store["targets"]) - 1)]
        mujoco_gym.data_store["distance"] = math.dist(data.body("torso").xipos, data.body(mujoco_gym.data_store["current_target"]).xipos)
    return 0, data.body(mujoco_gym.data_store["current_target"]).xipos

def pick_up_dynamic(mujoco_gym, data, model):
    """
    Update Target position and adds the inventory to the agent as an observation
    """
    reward = 0
    if "inventory" not in mujoco_gym.data_store.keys():
        mujoco_gym.data_store["inventory"] = []
        mujoco_gym.data_store["inventory"].append(0)
    if "targets" not in mujoco_gym.data_store.keys():
        mujoco_gym.data_store["targets"] = mujoco_gym.filterByTag("target")
        mujoco_gym.data_store["current_target"] = mujoco_gym.data_store["targets"][random.randint(0, len(mujoco_gym.data_store["targets"]) - 1)]["name"]
    distance = mujoco_gym.calculate_distance("torso", mujoco_gym.data_store["current_target"])
    if distance < 2:
        print("target reached")
        reward = 0
        if mujoco_gym.data_store["inventory"][0] == 0:
            mujoco_gym.data_store["inventory"][0] = 1
            reward = 1
        elif mujoco_gym.data_store["inventory"][0] == 1:
            mujoco_gym.data_store["inventory"][0] = 0
            reward = 1
        mujoco_gym.data_store["current_target"] = mujoco_gym.data_store["targets"][random.randint(0, len(mujoco_gym.data_store["targets"]) - 1)]["name"]
        mujoco_gym.data_store["distance"] = math.dist(data.body("torso").xipos, data.body(mujoco_gym.data_store["current_target"]).xipos)
    return reward, np.concatenate((data.body(mujoco_gym.data_store["current_target"]).xipos, mujoco_gym.data_store["inventory"]))


def test():
    env = SingleAgent("/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/single_agent/ModelVis.xml", infoJson="/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/single_agent/info_example.json", render=False, print_camera_config=False, add_target_coordinates=False, add_agent_coordinates=True, end_epoch_on_turn=True, env_dynamics=[pick_up_dynamic], reward_function=test_reward, max_step=8192, use_ctrl_cost=False)
    print("env created")
    obs = env.reset()
    print(env.filterByTag("target"))
    print(env.get_data("target_3"))

def train():
    env = SingleAgent("/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/single_agent/ModelVis.xml", info_json="/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/single_agent/info_example.json", render=False, print_camera_config=False, add_target_coordinates=False, add_agent_coordinates=True, end_epoch_on_turn=True, env_dynamics=[pick_up_dynamic], reward_function=test_reward, max_step=8192, use_ctrl_cost=False)
    print("env created")
    layer = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[1024, 512, 256], vf=[1024, 512, 256])])
    policy_kwargs = dict(net_arch=dict(pi=[4096, 2048, 1024], qf=[4096, 2048, 1024]))
    model = SAC("MlpPolicy", env, verbose=1, train_freq=(128, "step"), batch_size=128, learning_starts=100000, learning_rate=0.0015, buffer_size=1500000, policy_kwargs=policy_kwargs)
    print("model created")
    model.learn(total_timesteps=2000000, progress_bar=True)
    print("model trained")
    model.save("models/sac_model")
    env2 = MyEnv("envs/ModelVis.xml", render=True, print_camera_config=False, add_target_coordinates=True, add_agent_coordinates=True, end_epoch_on_turn=True, max_step=8192)
    obs = env2.reset()
    for i in range(8192):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env2.step(action, False)
        if dones:
            print(rewards)
            obs = env2.reset()
        time.sleep(0.1)
        env2.render()

def train_ray():
    config = A3CConfig()
    env = SingleAgent("/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/single_agent/ModelVis.xml", info_json="/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/single_agent/info_example.json", render=False, print_camera_config=False, add_target_coordinates=False, add_agent_coordinates=True, end_epoch_on_turn=True, env_dynamics=[pick_up_dynamic], reward_function=test_reward, max_step=8192, use_ctrl_cost=False)
    config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=4)
    print(config.to_dict())
    algo = config.build(env=env)
    algo.train()

def infer():
    model = SAC.load("models/sac_model")
    env = SingleAgent("envs/ModelVis.xml", info_json="envs/info_example.json", render=True, print_camera_config=False, add_target_coordinates=False, add_agent_coordinates=True, end_epoch_on_turn=True, env_dynamics=[pick_up_dynamic], reward_function=test_reward, max_step=8192, use_ctrl_cost=False)
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
    train_ray()