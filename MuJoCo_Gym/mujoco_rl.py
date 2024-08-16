import json
import copy
import time
import numpy as np
from gymnasium.spaces import Box, Space
# from ray.rllib.env import MultiAgentEnv
from pettingzoo import ParallelEnv
import os

try:
    from mujoco_parent import MuJoCoParent
    from helper import update_deep
except:
    from MuJoCo_Gym.mujoco_parent import MuJoCoParent
    from MuJoCo_Gym.helper import update_deep


class MuJoCoRL(ParallelEnv, MuJoCoParent):
    """A class representing a MuJoCo Reinforcement Learning environment.

    Args:
        config_dict (dict): A dictionary containing the configuration parameters for the environment.

    Attributes:
        agents (list): A list of agent names in the environment.
        xml_paths (str or list): The path(s) to the XML file(s) defining the MuJoCo model(s) for the environment.
        info_jsons (str or list): The path(s) to the JSON file(s) containing additional information about the environment.
        render_mode (bool): A flag indicating whether to render the environment.
        export_path (str): The path to export the environment.
        free_joint (bool): A flag indicating whether to enable free joint movement.
        skip_frames (int): The number of frames to skip between each step.
        max_steps (int): The maximum number of steps in the environment.
        reward_functions (list): A list of reward functions for the environment.
        done_functions (list): A list of done functions for the environment.
        environment_dynamics (list): A list of environment dynamics classes.
        agent_cameras (bool): A flag indicating whether to enable agent cameras.
        sensor_resolution (tuple): The resolution of the sensors in the environment.

    Methods:
        __init__(self, config_dict: dict): Initializes the MuJoCoRL environment.
        __instantiateJson(self): Instantiates the JSON file(s) for the environment.
        __check_dynamics(self, environment_dynamics: list): Checks the output of the dynamic function in each Dynamic Class.
        __check_done_functions(self, done_functions: list): Checks the output of each done function.
        __check_reward_functions(self, reward_functions: list): Checks the output of each reward function.
        __create_action_space(self) -> dict: Creates the action space for the environment.
        __create_observation_space(self) -> dict: Creates the observation space for the environment.
        step(self, action: dict) -> [dict, dict, dict, dict]: Applies the actions for each agent and returns the observations, rewards, terminations, truncations, and infos for each agent.
    """

    def __init__(self, config_dict: dict):
        self.agents = config_dict.get("agents", [])
        self.possible_agents = self.agents
        self.xml_paths = config_dict.get("xmlPath")
        self.info_jsons = config_dict.get("infoJson", None)
        self.render_mode = config_dict.get("renderMode", False)
        self.export_path = config_dict.get("exportPath")
        self.free_joint = config_dict.get("freeJoint", False)
        self.skip_frames = config_dict.get("skipFrames", 1)
        self.max_steps = config_dict.get("maxSteps", 1024)
        self.reward_functions = config_dict.get("rewardFunctions", [])
        self.done_functions = config_dict.get("doneFunctions", [])
        self.environment_dynamics = config_dict.get("environmentDynamics", [])
        self.agent_cameras = config_dict.get("agentCameras", False)
        sensor_resolution = config_dict.get("sensorResolution", (64, 64))

        self.timestep = 0
        self.start_time = time.time()

        self.action_routing = {"physical": [], "dynamic": {}}

        self.data_store = {agent: {} for agent in self.agents}

        MuJoCoParent.__init__(self, xml_paths=self.xml_paths, export_path=self.export_path, render=self.render_mode,
                              #ToDo: why is this None?
                              free_joint=self.free_joint, agent_cameras=self.agent_cameras,
                              sensor_resolution=sensor_resolution)

        self.__instantiateJson()

        self.environment_dynamics = [dynamic(self) for dynamic in self.environment_dynamics]
        self.__check_dynamics(self.environment_dynamics)
        self.__check_reward_functions(self.reward_functions)
        self.__check_done_functions(self.done_functions)

        self._observation_space = self.__create_observation_space()
        # ToDo: why is this a list?
        self.__observation_space = self._observation_space[list(self._observation_space.keys())[0]]

        self._action_space = self.__create_action_space()
        # ToDo: why is this a list?
        self.__action_space = self._action_space[list(self._action_space.keys())[0]]

    def __instantiateJson(self):
        """If a json file or a list of json files is provided, it is loaded into the environment in this function.
        """
        if isinstance(self.info_jsons, list):
            if len(self.info_jsons) != len(self.xml_paths):
                raise Exception("Length mismatch between info_json list and xml_paths list")
            head, tail = os.path.split(self.xml_path)
            json_file = tail.split(".")[0] + ".json"
            json_file = [current for current in self.info_jsons if json_file in current][0]
            json_file = open(json_file)
            self.info_json = json.load(json_file)
            self.info_name_list = [key for key in self.info_json["environment"]["objects"].keys()]
        elif isinstance(self.info_jsons, str):
            json_file = open(self.info_jsons)
            self.info_json = json.load(json_file)
            self.info_name_list = [key for key in self.info_json["environment"][
                "objects"].keys()]  #ToDo: is this not a df? looks like a list or sth
        else:
            self.info_json = None
            self.info_name_list = []

    def __check_dynamics(self, environment_dynamics: list):
        """Check the output of the dynamic function in every Dynamic Class.
           I.e. whether the observation has the shape of and suits the domain of the observation space and whether
           the reward is a float.

        Parameters:
            environment_dynamics (list): List of all environment dynamic classes
        """
        for environment_dynamic_instance in environment_dynamics:
            actions = environment_dynamic_instance.action_space["low"]
            reward, observations, done, info = environment_dynamic_instance.dynamic(self.agents[0], actions)
            # check observations
            if not len(environment_dynamic_instance.observation_space["low"]) == len(observations):
                raise Exception(f"Observation, the second return variable of dynamic function, must match length"
                                f" of lower bound of observation space of {environment_dynamic_instance}")
            if not np.all(environment_dynamic_instance.observation_space["low"] <= observations):
                raise Exception(f"Observation, the second return variable of dynamic function, exceeds the lower bound"
                                f" on at least one axis of the observation space of {environment_dynamic_instance}")
            if not len(environment_dynamic_instance.observation_space["high"]) == len(observations):
                raise Exception(f"Observation, the second return variable of dynamic function, must match length of"
                                f" upper bound of observation space of {environment_dynamic_instance} must at least be"
                                f" three dimensional")
            if not np.all(environment_dynamic_instance.observation_space["high"] >= observations):
                raise Exception(f"Observation, the second return variable of dynamic function, exceeds the upper bound"
                                f" on at least one axis of the observation space of observation space"
                                f" of {environment_dynamic_instance}")
            # check reward
            if not (isinstance(reward, float) or isinstance(reward, int)):
                raise Exception(f"Reward, the first return variable of dynamic function"
                                f" of {environment_dynamic_instance}, must be a float")

    def __check_done_functions(self, done_functions: list):
        """Check the output of every done function.
           I.e. whether done is a boolean and whether reward is a float.

        Parameters:
            done_functions (list): List of all done functions
        """
        for doneFunction in done_functions:
            done = doneFunction(self, self.agents[0])
            # check done
            if not isinstance(done, int):
                raise Exception(f"Done, the first return variable of {doneFunction}, must be a boolean")

    def __check_reward_functions(self, reward_functions: list):
        """Check the output of every reward function.
           I.e. whether reward is a float.

        Parameters:
            reward_functions (list): List of all reward functions
        """
        for reward_function in reward_functions:
            reward = reward_function(self, self.agents[0])
            # check reward
            if not (isinstance(reward, float) or isinstance(reward, int)):
                raise Exception(f"Reward, the second return variable of {reward_function}, must be a float")

    def __create_action_space(self) -> dict:
        """Creates the action space for the current environment

        Returns:
            new_action_space (dict): A dictionary of action spaces for each agent
        """
        action_space = {}
        new_action_space = {}
        for agent in self.agents:
            # Gets the action space from the MuJoCo environment
            action_space[agent] = MuJoCoRL.get_action_space_mujoco(self, agent)
            self.action_routing["physical"] = [0, len(action_space[agent]["low"])]
            for dynamic in self.environment_dynamics:
                dyn_action_space = dynamic.action_space
                self.action_routing["dynamic"][dynamic.__class__.__name__] = [len(action_space[agent]["low"]),
                                                                              len(action_space[agent]["low"]) +
                                                                              len(dyn_action_space["low"])]
                action_space[agent]["low"] += dyn_action_space["low"]
                action_space[agent]["high"] += dyn_action_space["high"]

            new_action_space[agent] = Box(low=np.array(action_space[agent]["low"]),
                                          high=np.array(action_space[agent]["high"]))
        return new_action_space

    def __create_observation_space(self) -> dict:
        """Creates the observation space for the current environment

        Returns:
            new_observation_space (dict): A dictionary of observation spaces for each agent
        """
        observation_space = {}
        new_observation_space = {}
        for agent in self.agents:
            # print(agent)
            observation_space[agent] = MuJoCoRL.get_observation_space_mujoco(self, agent)
            # print(observation_space[agent]["low"], observation_space[agent]["high"])
            # Get the action space for the environment dynamics
            for dynamic in self.environment_dynamics:
                observation_space[agent]["low"] += dynamic.observation_space["low"]
                observation_space[agent]["high"] += dynamic.observation_space["high"]
            new_observation_space[agent] = Box(low=np.array(observation_space[agent]["low"]),
                                               high=np.array(observation_space[agent]["high"]))
        return new_observation_space

    def __apply_dynamics(self, action: dict, observations: dict, rewards: dict, terminations: dict, infos: dict,
                         data_store_copies: list):
        """
            Applies the dynamics of the environment to update the observations, rewards, terminations, and infos for each agent.

            Args:
                action (dict): A dictionary containing the actions for each agent.
                observations (dict): A dictionary containing the observations for each agent.
                rewards (dict): A dictionary containing the rewards for each agent.
                terminations (dict): A dictionary containing the termination status for each agent.
                infos (dict): A dictionary containing additional information for each agent.
                data_store_copies (list): A list of data store copies for each environment dynamic.

            Returns:
                tuple: A tuple containing the updated observations, rewards, terminations, infos, and data store copies.
            """
        for i, dynamic in enumerate(self.environment_dynamics):
            # self.data_store = data_store_copies[i]
            for agent in self.agents:
                dynamic_indizes = self.action_routing["dynamic"][dynamic.__class__.__name__]
                dynamic_actions = action[agent][dynamic_indizes[0]:dynamic_indizes[1]]
                reward, obs, done, info = dynamic.dynamic(agent, dynamic_actions)
                observations[agent] = np.concatenate((observations[agent], obs))
                rewards[agent] += reward
                terminations[agent] = any([terminations[agent], done])
                infos[agent][dynamic.__class__.__name__] = info
        return observations, rewards, terminations, infos, data_store_copies

    def step(self, action: dict) -> [dict, dict, dict, dict, dict]:
        """Applies the actions for each agent and returns the observations, rewards, terminations, truncations,
           and infos for each agent

        Parameters:
            action (dict): A dictionary of actions for each agent

        Returns:
            observations (dict): A dictionary of observations for each agent
            rewards (dict): A dictionary of rewards for each agent
            terminations (dict): A dictionary of booleans indicating whether each agent is terminated
            truncations (dict): A dictionary of booleans indicating whether each agent is truncated
            infos (dict): A dictionary of dictionaries containing additional information for each agent
        """
        mujoco_actions = {key: action[key][self.action_routing["physical"][0]:self.action_routing["physical"][1]]
                          for key in action.keys()}
        self.apply_action(mujoco_actions, skip_frames=self.skip_frames)

        observations = {agent: self.get_sensor_data(agent) for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        data_store_copies = [copy.deepcopy(self.data_store) for _ in range(len(self.environment_dynamics))]
        original_data_store = copy.deepcopy(self.data_store)

        observations, rewards, terminations, infos, data_store_copies = self.__apply_dynamics(action, observations,
                                                                                              rewards, terminations,
                                                                                              infos, data_store_copies)

        # self.data_store = original_data_store
        # for data in data_store_copies:
        #     self.data_store = update_deep(self.data_store, data)

        for reward in self.reward_functions:
            rewards = {agent: rewards[agent] + reward(self, agent) for agent in self.agents}

        truncations = self.__check_truncations()

        if len(self.done_functions) != 0:
            for done in self.done_functions:
                terminations = {agent: any([terminations[agent], done(self, agent)]) for agent in self.agents}
                terminations["__all__"] = any(terminations.values())
                if terminations["__all__"] == True:
                    break

        self.timestep += 1
        return observations, rewards, terminations, truncations, infos

    def reset(self, *, seed: int = None, options=None) -> [dict, dict]:
        """Resets the environment and returns the observations for each agent

        Parameters:
            seed (int): Seed
            options (ToDo): options

        Returns:
            observations (dict): A dictionary of observations for each agent
            infos (dict): A dictionary of dictionaries containing additional information for each agent
        """
        MuJoCoParent.reset(self)

        if isinstance(self.info_jsons, list):
            head, tail = os.path.split(self.xml_path)
            json_file = tail.split(".")[0] + ".json"
            json_file = [current for current in self.info_jsons if json_file in current][0]
            json_file = open(json_file)
            self.info_json = json.load(json_file)
            self.info_name_list = [key for key in self.info_json["environment"]["objects"].keys()]

        self.data_store = {agent: {} for agent in self.agents}

        observations = {agent: self.get_sensor_data(agent) for agent in self.agents}
        action = {agent: self.__action_space.sample() for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        data_store_copies = [copy.deepcopy(self.data_store) for _ in range(len(self.environment_dynamics))]
        original_data_store = copy.deepcopy(self.data_store)

        observations, rewards, terminations, infos, data_store_copies = self.__apply_dynamics(action, observations,
                                                                                              rewards, terminations,
                                                                                              infos, data_store_copies)

        self.data_store = original_data_store
        for data in data_store_copies:
            self.data_store = update_deep(self.data_store, data)

        self.timestep = 0
        return observations, infos

    def action_space(self, agent: str) -> Space:
        """Returns the action space for the specified agent

        Parameters:
            agent (AgentID): The ID of the agent

        Returns:
            action_space (gymnasium.spaces.Space): The action space for the agent
        """
        return self._action_space[agent]

    def observation_space(self, agent: str) -> Space:
        """Returns the observation space for the specified agent

        Parameters:
            agent (AgentID): The ID of the agent

        Returns:
            observation_space (gymnasium.spaces.Space): The observation space for the agent
        """
        return self._observation_space[agent]

    def filter_by_tag(self, tag: str) -> list:
        """Filter environment for object with specific tag

        Parameters:
            tag (str): Tag to be filtered for

        Returns:
            filtered (list): List of objects with the specified tag
        """
        filtered = []
        for object in self.info_json["environment"]["objects"]:
            if "tags" in self.info_json["environment"]["objects"][object].keys():
                if self.info_json["environment"]["objects"][object]["tags"] != None:
                    if tag in self.info_json["environment"]["objects"][object]["tags"]:
                        data = self.get_data(object)
                        filtered.append(data)
        for area in self.info_json["areas"]:
            for object in self.info_json["areas"][area]["objects"]:
                if "tags" in self.info_json["areas"][area]["objects"][object].keys():
                    if self.info_json["areas"][area]["objects"][object]["tags"] != None:
                        if tag in self.info_json["areas"][area]["objects"][object]["tags"]:
                            data = self.get_data(object)
                            filtered.append(data)
        return filtered

    def get_data(self, name: str) -> np.array:
        """Returns the data for an object/geom with the given name

        Parameters:
            name (str): The name of the object/geom

        Returns:
            data (np.array): The data for the object/geom
        """
        data = MuJoCoParent.get_data(self, name)
        if name in self.info_name_list:
            # index = self.info_name_list.index(name)
            for key in self.info_json["environment"]["objects"][name].keys():
                if key not in ["position", "orientation", "mass"]:
                    data[key] = self.info_json["environment"]["objects"][name][key]
        return data

    def __get_observations(self) -> dict:
        """Returns the observations for each agent.
        # ToDo: to be implemented
        Returns:
            observations (dict): A dictionary of observations for each agent
        """
        observations = {}
        return observations

    def __check_truncations(self) -> dict:
        """Checks whether each agent is terminated

        Returns:
            truncations (dict): A dictionary of booleans indicating whether each agent is terminated #ToDo: i added this, is this correct?
        """
        if self.timestep >= self.max_steps:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}
        truncations["__all__"] = all(truncations.values())
        return truncations

    def __environment_functions(self) -> [dict, dict, dict]:
        """Executes the list of environment functions
        ToDo: to be implemented
        Returns:
            reward (dict): A dictionary of rewards for each agent
            observations (dict): A dictionary of observations for each agent
            infos (dict): A dictionary of dictionaries containing additional information for each agent
        """
        reward = {}
        observations = {}
        infos = {}
        return reward, observations, infos
