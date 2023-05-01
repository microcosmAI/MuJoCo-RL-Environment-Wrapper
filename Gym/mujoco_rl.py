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
        self.doneFunctions = configDict.get("doneFunctions", [])
        self.environmentDynamics = configDict.get("environmentDynamics", [])

        self.dataStore = {}

        MuJoCoParent.__init__(self, self.xmlPath, self.exportPath, render=self.renderMode, freeJoint=self.freeJoint, agents=self.agents, skipFrames=self.skipFrames)
        ParallelEnv.__init__(self)

        self.__checkDynamics(self.environmentDynamics)
        self.__checkDoneFunctions(self.doneFunctions)
        self.__checkRewardFunctions(self.rewardFunctions)

        self.observationSpace = self.__createObservationSpace()
        self.actionSpace = self.__createActionSpace()

        if self.infoJson != "":
            jsonFile = open(self.infoJson)
            self.infoJson = json.load(jsonFile)
            self.infoNameList = [object["name"] for object in self.infoJson["objects"]]
        else:
            self.infoJson = None
            self.infoNameList = []

    def __checkDynamics(self, environmentDynamics):
        '''
        Check the output of the dynamic function in every Dynamic Class. 
        I.e. whether the observation has the shape of and suits the domain of the observation space and whether the reward is a float. 

        Parameter:
            environmentDynamics (list): list of all environment dynamic classes
        '''
        for environmentDynamic in environmentDynamics:
            environmentDynamicInstance = environmentDynamic(self)
            reward, observations = environmentDynamicInstance.dynamic()
            # check observations
            if not len(environmentDynamicInstance.observation_space["low"]) == len(observations):
                raise Exception(f"Observation, the second return variable of dynamic function, must match length of lower bound of observation space of {environmentDynamicInstance}")
            if not np.all(environmentDynamicInstance.observation_space["low"] <= observations):
                raise Exception(f"Observation, the second return variable of dynamic function, exceeds the lower bound on at least one axis of the observation space of {environmentDynamicInstance}")
            if not len(environmentDynamicInstance.observation_space["high"]) == len(observations):
                raise Exception(f"Observation, the second return variable of dynamic function, must match length of upper bound of observation space of {environmentDynamicInstance} must at least be three dimensional")
            if not np.all(environmentDynamicInstance.observation_space["high"] >= observations):
                raise Exception(f"Observation, the second return variable of dynamic function, exceeds the upper bound on at least one axis of the observation space of observation space of {environmentDynamicInstance}")
            # check reward
            if not isinstance(reward, float):
                raise Exception(f"Reward, the first return variable of dynamic function of {environmentDynamicInstance}, must be a float")

    def __checkDoneFunctions(self, doneFunctions):
        '''
        Check the output of every done function.
        I.e. whether done is a boolean and whether reward is a float.

        Parameter:
            doneFunctions (list): list of all done functions
        '''
        for doneFunction in doneFunctions:
            done, reward = doneFunction()
            # check done
            if not isinstance(done, int):
                raise Exception(f"Done, the first return variable of {doneFunction}, must be a boolean")
            # check reward
            if not isinstance(reward, float):
                raise Exception(f"Reward, the second return variable of {doneFunction}, must be a float")
    
    def __checkRewardFunctions(self, rewardFunctions):
        '''
        Check the output of every reward function.
        I.e. whether reward is a float.

        Parameter:
            rewardFunctions (list): list of all reward functions
        '''
        for rewardFunction in rewardFunctions:
            reward = rewardFunction()
            # check reward
            if not isinstance(reward, float):
                raise Exception(f"Reward, the second return variable of {rewardFunction}, must be a float")

    def __createActionSpace(self) -> dict:
        """
        Creates the action space for the current environment.
        returns:
            actionSpace (dict): a dictionary of action spaces for each agent
        """
        actionSpace = {}
        return actionSpace

    def __createObservationSpace(self) -> dict:
        """
        Creates the observation space for the current environment
        returns:
            observationSpace (dict): a dictionary of observation spaces for each agent
        """
        observationSpace = {}
        return observationSpace

    def step(self, action: dict):
        """
        Applies the actions for each agent and returns the observations, rewards, terminations, truncations, and infos for each agent.
        arguments:
            action (dict): a dictionary of actions for each agent
        returns:
            observations (dict): a dictionary of observations for each agent
            rewards (dict): a dictionary of rewards for each agent
            terminations (dict): a dictionary of booleans indicating whether each agent is terminated
            truncations (dict): a dictionary of booleans indicating whether each agent is truncated
            infos (dict): a dictionary of dictionaries containing additional information for each agent
        """
        self.mujocoStep()
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        return observations, rewards, terminations, truncations, infos

    def reset(self, returnInfos=False):
        """
        Resets the environment and returns the observations for each agent.
        arguments:
            returnInfos (bool): whether to return the infos for each agent
        returns:
            observations (dict): a dictionary of observations for each agent
            infos (dict): a dictionary of dictionaries containing additional information for each agent
        """
        observations = {}
        infos = {}
        if returnInfos:
            return observations, infos
        return observations
    
    def filterByTag(self, tag) -> list:
        """
        Filter environment for object with specific tag
        Parameters:
            tag (str): tag to be filtered for
        Returns:
            filtered (list): list of objects with the specified tag
        """
        filtered = []
        for object in self.infoJson["objects"]:
            if "tags" in object.keys():
                if tag in object["tags"]:
                    data = self.getData(object["name"])
                    filtered.append(data)
        return filtered

    def getData(self, name: str) -> np.array:
        """
        Returns the data for an object/geom with the given name.
        arguments:
            name (str): the name of the object/geom
        returns:
            data (np.array): the data for the object/geom
        """
        data = MuJoCoParent.getData(self, name)
        if name in self.infoNameList:
            index = self.infoNameList.index(name)
            for key in self.infoJson["objects"][index].keys():
                data[key] = self.infoJson["objects"][index][key]
        return data

    def __getObservations(self) -> dict:
        """
        Returns the observations for each agent.
        returns:
            observations (dict): a dictionary of observations for each agent
        """
        observations = {}
        return observations

    def __doneFunctions(self) -> dict:
        """
        Executes the list of done functions and returns the terminations for each agent.
        returns:
            terminations (dict): a dictionary of booleans indicating whether each agent is terminated
        """
        terminations = {}
        return terminations

    def __trunkationsFunctions(self):
        """
        Executes the list of truncation functions and returns the truncations for each agent.
        returns:
            truncations (dict): a dictionary of booleans indicating whether each agent is truncated
        """
        truncations = {}
        return truncations

    def __environmentFunctions(self):
        """
        Executes the list of environment functions.
        returns:
            reward (dict): a dictionary of rewards for each agent
            observations (dict): a dictionary of observations for each agent
            infos (dict): a dictionary of dictionaries containing additional information for each agent
        """
        reward = {}
        observations = {}
        infos = {}
        return reward, observations, infos