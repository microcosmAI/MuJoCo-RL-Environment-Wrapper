import functools
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box
from pettingzoo.test import parallel_api_test
from pettingzoo import ParallelEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils import parallel_to_aec, wrappers
import mujoco as mj
import random
import math
import xml.etree.ElementTree as ET
import os
from stable_baselines3 import PPO
from mujoco.glfw import glfw
import functools
import json
from scipy.spatial.transform import Rotation 
from mujoco_parent import MuJoCoParent

class MuJoCo_RL(ParallelEnv, MuJoCoParent):

    def __init__(self, configDict: dict):
        self.agents = configDict.get("agents", [])
        self.xmlPath = configDict.get("xmlPath")
        self.infoJson = configDict.get("infoJson", "")
        self.renderMode = configDict.get("renderMode", False)
        self.exportPath = configDict.get("exportPath")
        self.freeJoint = configDict.get("freeJoint", False)
        self.skipFrames = configDict.get("skipFrames", 1)
        self.maxSteps = configDict.get("maxSteps", 1024)
        self.rewardFunctions = configDict.get("rewardFunctions", [])
        self.observationFunctions = configDict.get("observationFunctions", [])
        self.doneFunctions = configDict.get("doneFunctions", [])

        self.dataStore = {}

        self.observationSpace = self.__createObservationSpace()
        self.actionSpace = self.__createActionSpace()

    def __createActionSpace(self):
        pass

    def __createObservationSpace(self):
        pass

    def step(self, action: dict):
        pass

    def reset(self):
        pass

    def getData(self, name: str):
        pass

    def __getObservations(self):
        pass

    def __doneFunctions(self):
        pass

    def __trunkationsFunctions(self):
        pass

    def __environmentFunctions(self):
        pass