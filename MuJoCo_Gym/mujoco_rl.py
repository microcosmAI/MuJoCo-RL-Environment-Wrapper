import numpy as np
from gymnasium.spaces import Discrete, Box
import mujoco as mj
import xml.etree.ElementTree as ET
import functools
import json
from scipy.spatial.transform import Rotation 
from mujoco_parent import MuJoCoParent
from ray.rllib.env import MultiAgentEnv
import copy
from helper import updateDeep
import time

class MuJoCo_RL(MultiAgentEnv, MuJoCoParent):

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
        self.agentCameras = configDict.get("agentCameras", False)

        self.timestep = 0
        self.start_time = time.time()

        self.actionRouting = {"physical":[],"dynamic":{}}

        self.dataStore = {agent:{} for agent in self.agents}

        MuJoCoParent.__init__(self, self.xmlPath, self.exportPath, render=self.renderMode, freeJoint=self.freeJoint, agentCameras=self.agentCameras, agents=self.agents, skipFrames=self.skipFrames)
        MultiAgentEnv.__init__(self)

        self.__checkDynamics(self.environmentDynamics)
        self.__checkDoneFunkcions(self.doneFunctions)
        self.__checkRewardFunctions(self.rewardFunctions)

        self.environmentDynamics = [dynamic(self) for dynamic in self.environmentDynamics]

        self._observation_space = self.__createObservationSpace()
        self.observation_space = self._observation_space[list(self._observation_space.keys())[0]]
        self._action_space = self.__createActionSpace()
        self.action_space = self._action_space[list(self._action_space.keys())[0]]

        if self.infoJson != "":
            jsonFile = open(self.infoJson)
            self.infoJson = json.load(jsonFile)
            self.infoNameList = [object["name"] for object in self.infoJson["objects"]]
        else:
            self.infoJson = None
            self.infoNameList = []

    # TO-DO Lisa: Implement those functions
    def __checkDynamics(self, environmentDynamics):
        pass

    def __checkDoneFunkcions(self, doneFunctions):
        pass
    
    def __checkRewardFunctions(self, rewardFunctions):
        pass

    def __createActionSpace(self) -> dict:
        """
        Creates the action space for the current environment.
        returns:
            actionSpace (dict): a dictionary of action spaces for each agent
        """
        actionSpace = {}
        newActionSpace = {}
        for agent in self.agents:
            # Gets the action space from the MuJoCo environment
            actionSpace[agent] = MuJoCo_RL.getActionSpaceMuJoCo(self, agent)
            self.actionRouting["physical"] = [0, len(actionSpace[agent]["low"])]
            for dynamic in self.environmentDynamics:
                dynActionSpace = dynamic.action_space
                self.actionRouting["dynamic"][dynamic.__class__.__name__] = [len(actionSpace[agent]["low"]), len(actionSpace[agent]["low"])+len(dynActionSpace["low"])]
                actionSpace[agent]["low"] += dynActionSpace["low"]
                actionSpace[agent]["high"] += dynActionSpace["high"]

            newActionSpace[agent] = Box(low=np.array(actionSpace[agent]["low"]), high=np.array(actionSpace[agent]["high"]))
        return newActionSpace

    def __createObservationSpace(self) -> dict:
        """
        Creates the observation space for the current environment
        returns:
            observationSpace (dict): a dictionary of observation spaces for each agent
        """
        observationSpace = {}
        newObservationSpace = {}
        for agent in self.agents:
            observationSpace[agent] = MuJoCo_RL.getObservationSpaceMuJoCo(self, agent)
            # Get the action space for the environment dynamics
            for dynamic in self.environmentDynamics:
                observationSpace[agent]["low"] += dynamic.observation_space["low"]
                observationSpace[agent]["high"] += dynamic.observation_space["high"]
            newObservationSpace[agent] = Box(low=np.array(observationSpace[agent]["low"]), high=np.array(observationSpace[agent]["high"]))
        return newObservationSpace

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
        mujocoActions = {key:action[key][self.actionRouting["physical"][0]:self.actionRouting["physical"][1]] for key in action.keys()}
        self.applyAction(mujocoActions)
        observations = {agent:self.getSensorData(agent) for agent in self.agents}
        rewards = {agent:0 for agent in self.agents}

        dataStoreCopies = [copy.deepcopy(self.dataStore) for _ in range(len(self.environmentDynamics))]
        originalDataStore = copy.deepcopy(self.dataStore)

        for i, dynamic in enumerate(self.environmentDynamics):
            self.dataStore = dataStoreCopies[i]
            for agent in self.agents:
                dynamicIndizes = self.actionRouting["dynamic"][dynamic.__class__.__name__]
                dynamicActions = action[agent][dynamicIndizes[0]:dynamicIndizes[1]]
                reward, obs = dynamic.dynamic(agent, dynamicActions)
                observations[agent] = np.concatenate((observations[agent], obs))
                rewards[agent] += reward

        
        self.dataStore = originalDataStore
        for data in dataStoreCopies:
            self.dataStore = updateDeep(self.dataStore, data)

        for reward in self.rewardFunctions:
            rewards = {agent:rewards[agent] + reward(self, agent) for agent in self.agents}

        terminations = self.__checkTerminations()

        truncations = {}
        if len(self.doneFunctions) == 0:
            truncations = {agent:terminations[agent] for agent in self.agents}
        else:
            for done in self.doneFunctions:
                truncations = {agent:terminations[agent] or done(self, agent) for agent in self.agents}
        truncations["__all__"] = all(truncations.values())

        infos = {agent:{} for agent in self.agents}
        self.timestep += 1
        if self.timestep % 50 == 0:
            print(self.timestep / self.maxSteps)
        return observations, rewards, terminations, truncations, infos

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment and returns the observations for each agent.
        arguments:
            returnInfos (bool): whether to return the infos for each agent
        returns:
            observations (dict): a dictionary of observations for each agent
            infos (dict): a dictionary of dictionaries containing additional information for each agent
        """
        observations = {agent:self.getSensorData(agent) for agent in self.agents}
        print("Timesteps per second:", self.timestep / (time.time() - self.start_time))
        self.start_time = time.time()

        for dynamic in self.environmentDynamics:
            for agent in self.agents:
                actions = dynamic.action_space["low"]
                reward, obs = dynamic.dynamic(agent, actions)
                observations[agent] = np.concatenate((observations[agent], obs))
        self.dataStore = {agent:{} for agent in self.agents}
        self.timestep = 0
        infos = {agent:{} for agent in self.agents}
        return observations, infos
        # return observations
    
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

    def __checkTerminations(self) -> dict:
        """
        Checks whether each agent is terminated.
        """
        if self.timestep >= self.maxSteps:
            terminations = {agent:True for agent in self.agents}
        else:
            terminations = {agent:False for agent in self.agents}
        terminations["__all__"] = all(terminations.values())
        if terminations["__all__"]:
            print("Episode done")
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