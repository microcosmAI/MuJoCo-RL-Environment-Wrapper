from MuJoCo_Gym.mujoco_rl import MuJoCo_RL
from MuJoCo_Gym.single_agent_wrapper import Single_Agent_Wrapper
import numpy as np
import random
from PIL import Image
import cv2

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time


class Image:
    def __init__(self, environment):
        self.environment = environment
        shape = 64 * 64 * 3
        self.observation_space = {"low": [0 for _ in range(shape)] + [0, 0, 0], "high": [257 for _ in range(shape)]+ [255, 255, 255]}
        self.action_space = {"low": [], "high": []}
        self.dataStore = {}
        self.targets = {"red":["target_2_geom0"], "blue":["target_1_geom0"]}

    def dynamic(self, agent, actions):
        if not "target" in self.environment.dataStore.keys():
            keys = [key for key in self.targets.keys()]
            self.environment.dataStore["target"] = random.choice(keys)

        color = self.environment.dataStore["target"]
        if color == "red":
            obs = np.array([255, 0, 0])
        elif color == "green":
            obs = np.array([0, 255, 0])
        elif color == "blue":
            obs = np.array([0, 0, 255])
        image = self.environment.getCameraData(agent)
        image = th.from_numpy(image)
        image = th.flatten(image[0])
        image = image.cpu().detach().numpy()
        observation = np.concatenate((image, obs))
        return 0, observation
    
def reward(mujoco_gym, agent):
    possible_targets = {"red":["target_2_geom0"], "blue":["target_1_geom0"]}
    targets = possible_targets[mujoco_gym.dataStore["target"]]
    reward = 0
    for target in targets:
        if(mujoco_gym.collision("agent_geom0", target)):
            reward = 1
    if reward == 0:
        for obstacles in possible_targets.keys():
            if obstacles != mujoco_gym.dataStore["target"]:
                for obstacle in possible_targets[obstacles]:
                    if(mujoco_gym.collision("agent_geom0", obstacle)):
                        reward = -1
    return reward

def collision_reward(mujoco_gym, agent):
    borders = ["border1", "border2", "border3", "border4"]
    reward = 0
    for border in borders:
        if(mujoco_gym.collision("agent_geom0", border)):
            reward = -1
    return reward

def done(mujoco_gym, agent):
    borders = ["border1", "border2", "border3", "border4"]
    for border in borders:
        if(mujoco_gym.collision("agent_geom0", border)):
            print("Collision")
            return True
    return False
    
environment_path = "Examples/Environment/SimpleTest.xml"
# info_path = "Environment/info_example.json"
agents = ["agent"]
# config_dict = {"xmlPath":environment_path, "infoJson":info_path, "agents":agents, "rewardFunctions":[reward_function], "doneFunctions":[done_function], "environmentDynamics":[Language], "freeJoint":True, "renderMode":True, "maxSteps":4096}
config_dict = {"xmlPath":environment_path, "agents":agents, "rewardFunctions":[reward, collision_reward], "doneFunctions":[done], "skipFrames":30, "environmentDynamics":[Image], "freeJoint":True, "renderMode":False, "maxSteps":4096, "agentCameras":True}

environment = MuJoCo_RL(config_dict, )
environment = Single_Agent_Wrapper(environment, "agent")

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.cnn = nn.Sequential(
            nn.Unflatten(1, (3, 64, 64)),
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[:-3][None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim-3), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        image = observations[:, :-3]
        result = self.linear(self.cnn(image))
        return th.cat((result, observations[:, -3:]), dim=1)
    



policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=32),
    net_arch=dict(pi=[128, 128], vf=[128, 128])
)

learning_rate = [1e-6, 1e-5, 1e-4]
network_sizes = [[64, 64], [128, 128]]
features_dims = [32, 64]

for i in range(3):
    for j in range(2):
        for k in range(2):
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=features_dims[k]),
                net_arch=dict(pi=network_sizes[j], vf=network_sizes[j])
            )
            model = PPO("CnnPolicy", environment, policy_kwargs=policy_kwargs, verbose=1, device="mps", tensorboard_log="./results/", learning_rate=learning_rate[i])
            model.learn(300000, tb_log_name="testCNNRun_" + str(learning_rate[i]) + "_" + str(network_sizes[j][0]) + "_" + str(features_dims[k]), progress_bar=True)
            model.save("models/CNN_Policy")