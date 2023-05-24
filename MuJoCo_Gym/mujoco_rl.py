import numpy as np
from gymnasium.spaces import Discrete, Box
import mujoco as mj
import xml.etree.ElementTree as ET
import functools
import json
from scipy.spatial.transform import Rotation
from ray.rllib.env import MultiAgentEnv
import copy
import time
try:
    from mujoco_parent import MuJoCoParent
    from helper import updateDeep
except:
    from MuJoCo_Gym.mujoco_parent import MuJoCoParent
    from MuJoCo_Gym.helper import updateDeep

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

        if self.infoJson != "":
            jsonFile = open(self.infoJson)
            self.infoJson = json.load(jsonFile)
            self.infoNameList = [object["name"] for object in self.infoJson["objects"]]
        else:
            self.infoJson = None
            self.infoNameList = []

        self.__checkDynamics(self.environmentDynamics)
        self.__checkRewardFunctions(self.rewardFunctions)
        self.__checkDoneFunctions(self.doneFunctions)

        self.environmentDynamics = [dynamic(self) for dynamic in self.environmentDynamics]

        self._observation_space = self.__createObservationSpace()
        self.observation_space = self._observation_space[list(self._observation_space.keys())[0]]
        self._action_space = self.__createActionSpace()
        self.action_space = self._action_space[list(self._action_space.keys())[0]]

    def __checkDynamics(self, environmentDynamics):
        '''
        Check the output of the dynamic function in every Dynamic Class. 
        I.e. whether the observation has the shape of and suits the domain of the observation space and whether the reward is a float. 

        Parameter:
            environmentDynamics (list): list of all environment dynamic classes
        '''
        for environmentDynamic in environmentDynamics:
            environmentDynamicInstance = environmentDynamic(self)
            actions = environmentDynamicInstance.action_space["low"]
            reward, observations = environmentDynamicInstance.dynamic(self.agents[0], actions)
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
            if not (isinstance(reward, float) or isinstance(reward, int)):
                raise Exception(f"Reward, the first return variable of dynamic function of {environmentDynamicInstance}, must be a float")

    def __checkDoneFunctions(self, doneFunctions):
        '''
        Check the output of every done function.
        I.e. whether done is a boolean and whether reward is a float.

        Parameter:
            doneFunctions (list): list of all done functions
        '''
        for doneFunction in doneFunctions:
            done = doneFunction(self, self.agents[0])
            # check done
            if not isinstance(done, int):
                raise Exception(f"Done, the first return variable of {doneFunction}, must be a boolean")
    
    def __checkRewardFunctions(self, rewardFunctions):
        '''
        Check the output of every reward function.
        I.e. whether reward is a float.

        Parameter:
            rewardFunctions (list): list of all reward functions
        '''
        for rewardFunction in rewardFunctions:
            reward = rewardFunction(self, self.agents[0])
            # check reward
            if not (isinstance(reward, float) or isinstance(reward, int)):
                raise Exception(f"Reward, the second return variable of {rewardFunction}, must be a float")


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

        self.timestep += 1

        infos = {agent:{} for agent in self.agents}
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
        MuJoCoParent.reset(self)
        observations = {agent:self.getSensorData(agent) for agent in self.agents}

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

    def getData(self, name: str):
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