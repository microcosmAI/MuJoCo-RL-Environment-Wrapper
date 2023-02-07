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
from gym_parent import MuJoCoParent


"""
The environment provides 3 agents (number is not changable at the moment) and randomly assigns one of the three targets to the agents.
At the moment it is not possible to change the number of agents.
"""


class MultiAgentEnv(ParallelEnv, MuJoCoParent):
    metadata = {"render_modes": ["human"], "name": "MultiAgentEnv"}

    def __init__(self, xml_path, infoJson=None, targets = ["target_1", "target_2", "target_3"], print_camera_config = False, add_target_coordinates = True, add_agent_coordinates = True, render=False, max_steps=1024, reward_function=None, done_function=None, env_dynamics=[], export_path=None):
        """
        Initializes the environment.
        arguments:
            xml_path: path to the xml file
            targets: list of targets for the default reward function
            print_camera_config: if True, the camera configuration is printed
            add_target_coordinates: if True, the target coordinates are added to the observation of each agent
            add_agent_coordinates: if True, the agent coordinates are added to the observation of each agent
            render_mode: if set to "human", the environment is rendered
            max_steps: maximum number of steps per epoch
            reward_function: function that returns the reward for each agent
            done_function: function that returns if the episode is done
            env_dynamics: list of functions that are called every step
            export_path: path to the folder where the data is exported
        """
        MuJoCoParent.__init__(self, xml_path, infoJson, export_path, render)
        self.print_camera_config = print_camera_config

        #get the full path
        self.xml_path = xml_path
        self.export_path = export_path

        self.add_target_coordinates = add_target_coordinates
        self.add_agent_coordinates = add_agent_coordinates

        self.overall_reward = 0
        self.number_of_steps = 0

        self.max_steps = max_steps
        self.targets = targets

        self.xml_tree = ET.parse(self.xml_path)
        self.xml_dict = self.elem2dict(self.xml_tree.getroot())

        self.reward_function = reward_function
        self.done_function = done_function
        self.env_dynamics = env_dynamics

        self.data_store = {} 

        self.time_prev = self.data.time

        # Create the names and mapping for the three agents
        self.possible_agents = ["agent" + str(r) for r in range(1,4)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Selects the target for each agent at random
        self.agent_targets = {agent:random.choice(self.targets) for agent in self.possible_agents}
        self.last_distances = {agent:None for agent in self.possible_agents}

        # Create the action and observation spaces
        self.__create_action_space()
        self.__create_observation_space()

        # If the render mode is set to "human", the environment is rendered
        if self.render_mode == 'human':
            self.__initializeWindow()


    def __create_action_space(self):
        """
        The functions creates a dict containing the action spaces for each individual agent
        """
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self._action_spaces = {agent: Box(low=low[0:int(len(low)/len(self.possible_agents))], high=high[0:int(len(high)/len(self.possible_agents))], dtype=np.float32) for agent in self.possible_agents}


    def __create_observation_space(self):
        """
        The functions creates a dict containing the observation spaces for each individual agent
        """
        # Dimensions of the actuator controls
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        dimensions = len(low)
        # Dimensions of the sensor data
        dimensions += len(self.data.sensordata)
        # Dimensions of the joint positions and velocities
        dimensions += len(np.concatenate([self.data.qpos.flat, self.data.qvel.flat]))
        # Dimensions of the body positions and velocities
        dimensions = int(dimensions / 3)
        if self.add_agent_coordinates:
            dimensions += 3
        if self.add_target_coordinates:
            dimensions += 3
        self._observation_spaces = {agent: Box(low=np.inf, high=np.inf, shape=(dimensions,), dtype=np.float32) for agent in self.possible_agents}

    def step(self, actions: dict):
        """
        Executes the next step in the simulation
        arguments:
            actions: a dict of actions for each respective agent. For example: {'agent1':[], 'agent2':[], 'agent3':[]}
        returns:
            observations: a dict of observations for each respective agent. For example: {'agent1':[], 'agent2':[], 'agent3':[]}
            rewards: a dict of rewards for each respective agent. For example: {'agent1':[], 'agent2':[], 'agent3':[]}
            terminations: a dict of terminations for each respective agent. For example: {'agent1':[], 'agent2':[], 'agent3':[]}
            truncations: a dict of truncations for each respective agent. For example: {'agent1':[], 'agent2':[], 'agent3':[]}
            infos: a dict of infos for each respective agent. For example: {'agent1':[], 'agent2':[], 'agent3':[]}
        """
        # Set the action
        for agent in actions.keys():
            index_of_agent = self.possible_agents.index(agent)
            ctrl_devider = len(self.data.ctrl) / len(self.possible_agents)
            self.data.ctrl[int(ctrl_devider * index_of_agent): int(ctrl_devider * (index_of_agent + 1))] = actions[agent]
        
        # Execute next step in the simulation
        mj.mj_step(self.model, self.data)

        # Getting the new observation
        observations = {agent:self.__get_observations(agent) for agent in self.possible_agents}

        for dynamic in self.env_dynamics:
            dynamic(self, observations)

        # Getting the rewards for the last step
        if self.reward_function is None:
            rewards = {agent:self.__reward(agent) for agent in self.possible_agents}
        else:
            rewards = {agent:self.reward_function(self, agent) for agent in self.possible_agents}

        # Checking whether one of the agents is done
        if self.done_function is None:
            terminations = {agent:self.__done(agent) for agent in self.possible_agents}
        else:
            terminations = {agent:self.done_function(self, agent) for agent in self.possible_agents}

        if self.max_steps < self.number_of_steps:
            truncations = {agent:True for agent in self.possible_agents}

        # Checking whether the episode is over
        truncations = self.__trunkations()

        # Setting infos (none at this point) for each agent
        infos = {agent: {} for agent in self.agents}

        self.number_of_steps += 1

        if self.export_path is not None:
            self.__export(self.export_path)

        if self.render_mode == 'human':
            self.time_prev = self.data.time
            if self.data.time - self.time_prev < 1.0/30.0:
                self.render()
        return observations, rewards, terminations, truncations, infos


    def reset(self, seed=None, return_info=False, options=None):
        """
        Resets the simulation
        arguments:
            seed: seed for the random number generator
            return_info: if True, the infos are returned
            options: options for the simulation
        returns:
            observations: a dict of observations for each respective agent. For example: {'agent1':[], 'agent2':[], 'agent3':[]}
            infos: a dict of infos for each respective agent. For example: {'agent1':[], 'agent2':[], 'agent3':[]}
        """
        # Reset mujoco simulation
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        self.agents = self.possible_agents[:]
        observations = {agent:self.__get_observations(agent) for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        # Reset the debug infos
        self.overAllReward = 0
        self.numberOfSteps = 0
        self.agent_targets = {agent:random.choice(self.targets) for agent in self.agents}
        self.last_distances = {agent:None for agent in self.agents}
        if return_info:
            return observations, infos
        else:
            return observations


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Box:
        """
        Returns the observation space for the agent, which is handed over as an argument
        arguments:
            agent: name of the agent
        returns:
            observation space for the agent
        """
        return self._observation_spaces[agent]


    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Box:
        """
        Returns the action space for the agent, which is handed over as an argument
        arguments:
            agent: name of the agent
        returns:
            action space for the agent
        """
        return self._action_spaces[agent]


    def observe(self, agent: str) -> np.ndarray:
        """
        Returns the current observations for the agent, which is handed over as an argument
        arguments:
            agent: name of the agent
        returns:
            observations of the environment for the agent
        """
        return self.__get_observations(agent)


    def __get_observations(self, agent):
        """
        get the observations for the agent
        arguments:
            agent: name of the agent
        """
        index_of_agent = self.possible_agents.index(agent)
        # Get the sensor data
        sensor_data = self.data.sensordata
        sensor_devider = len(sensor_data) / len(self.possible_agents)
        sensor_data = sensor_data[int(sensor_devider * index_of_agent): int(sensor_devider * (index_of_agent + 1))]
        observations = np.array(sensor_data)

        # Get the data of controlable elements (joints, motors etc.)
        ctrl_data = self.data.ctrl
        ctrl_devider = len(ctrl_data) / len(self.possible_agents)
        ctrl_data = ctrl_data[int(ctrl_devider * index_of_agent): int(ctrl_devider * (index_of_agent + 1))]
        observations = np.concatenate((observations, ctrl_data))

        # Get the joint positions and velocities
        state_vector = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        state_devider = len(state_vector) / len(self.possible_agents)
        state_data = state_vector[int(state_devider * index_of_agent): int(state_devider * (index_of_agent + 1))]
        observations = np.concatenate((observations, state_data))

        # add agent and target coordinates to observation if they are applied
        if self.add_agent_coordinates:
            observations = np.concatenate((observations, np.array(self.data.body(agent + "_torso").xipos)))
        if self.add_target_coordinates:
            observations = np.concatenate((observations, np.array(self.data.body(self.agent_targets[agent]).xipos)))
        return observations


    def __trunkations(self):
        if self.number_of_steps < 4096:
            truncations = {agent: False for agent in self.possible_agents}
        else:
            truncations = {agent: True for agent in self.possible_agents}
        return truncations


    def __reward(self, agent):
        """
        Calculates reward for the agent, which is handed over as an argument
        arguments:
            agent: name of the agent
        returns:
            reward for the agent
        """
        distance = math.dist(self.data.body(agent + "_torso").xipos, self.data.body(self.agent_targets[agent]).xipos)
        if self.last_distances[agent] is None:
            self.last_distances[agent] = distance
            reward = 0
        else:
            reward = self.last_distances[agent] - distance
            self.last_distances[agent] = distance
        self.overAllReward += reward
        return reward


    def __done(self, agent):
        """
        Calculates if the agent is done
        arguments:
            agent: name of the agent
        returns:
            True if the agent is done, False otherwise
        """
        if math.dist(self.data.body(agent + "_torso").xipos, self.data.body(self.agent_targets[agent]).xipos) < 0.1:
            print("done")
            return True
        if self.overAllReward < -100:
            print("done")
            return True
        return False