from collections import deque
from gym import spaces
import numpy as np
import gym
import os
import mujoco as mj
from mujoco.glfw import glfw
import time
from stable_baselines3 import PPO
import math
import torch as th
import json
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation  
from mujoco_py import MuJoCo_RL

class MuJoCo_RL(MuJoCo_RL. gym.env):

    def __init__(self, configDict: dict):
        self.xmlPath = configDict.get("xmlPath")
        self.agentName = configDict.get("agentName")
        self.infoJson = configDict.get("infoJson", "")
        self.render = configDict.get("render", False)
        self.exportPath = configDict.get("exportPath")
        self.freeJoint = configDict.get("freeJoint", False)
        self.skipFrames = configDict.get("skipFrames", 1)
        self.maxSteps = configDict.get("maxSteps", 1024)
        self.rewardFunctions = configDict.get("rewardFunctions", [])
        self.observationFunctions = configDict.get("observationFunctions", [])
        self.doneFunctions = configDict.get("doneFunctions", [])

        self.actionSpace = self.__createActionSpace()
        self.observationSpace = self.__createObservationSpace()

    def __createActionSpace(self) -> spaces.Box:
        action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        return action_space

    def __createObservationSpace(self) -> spaces.Box:
        observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        return observation_space

    def step(self, action: np.array):
        obs = np.array()
        reward = 0
        done = 0
        info = {}
        return obs, reward, done, info

    def reset(self) -> np.array:
        obs = np.array()
        return obs