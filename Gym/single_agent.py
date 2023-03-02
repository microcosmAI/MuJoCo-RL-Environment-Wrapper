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
from gym_parent import MuJoCoParent

class SingleAgent(gym.Env, MuJoCoParent):
    def __init__(self, xml_path, infoJson="", agent="torso", target="target", render=False, print_camera_config=False, add_target_coordinates=False, add_agent_coordinates=False, end_epoch_on_turn=False, use_ctrl_cost=False, ctrl_factor=0.01, use_head_sensor=False, reward_function=None, done_function=None, env_dynamics=[], max_step=1024):
        """
        Initializes the environment
        Parameters:
            xml_path: path to the xml file
            agent: name of the agent
            target: name of the target
            render: whether to render the environment
            print_camera_config: whether to print the camera config
            add_target_coordinates: whether to add the target coordinates to the observation
            add_agent_coordinates: whether to add the agent coordinates to the observation
            end_epoch_on_turn: whether to end the epoch when the agent flips on its head
            use_ctrl_cost: whether to use a control cost
            use_head_sensor: whether to use the head sensor to determine whether the agent is upside down
            max_step: maximum number of steps in an epoch
        """
        gym.Env.__init__(self)
        MuJoCoParent.__init__(self, xml_path, infoJson = infoJson, render=render)

        self.print_camera_config = print_camera_config

        #get the full path
        self.xml_path = xml_path
        self.max_step = max_step

        self.xml_tree = ET.parse(self.xml_path)

        self._healthy_z_range = (0.35, 1.1)

        self.add_target_coordinates = add_target_coordinates
        self.add_agent_coordinates = add_agent_coordinates

        self.ctrl_cost_weight = 0.5
        self.use_ctrl_cost = use_ctrl_cost
        self.control_factor = ctrl_factor
        self.use_head_sensor = use_head_sensor

        self.agent = agent
        self.target = target

        self.lastDistance = None
        self.overAllReward = 0
        self.numberOfSteps = 0

        self.reward_function = reward_function
        self.done_function = done_function
        self.env_dynamics = env_dynamics

        self.end_epoch_on_turn = end_epoch_on_turn 

        self.data_store = {}

        self.__createActionSpace()
        self.__createObservationSpace()

    def __createActionSpace(self):
        """
        Creates the action space
        """
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def __createObservationSpace(self):
        """
        Creats the observation space
        """
        # Dimensions of the actuator controls
        space_dict = {"low":np.array([]), "high":np.array([])}
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        dimensions = len(low)
        space_dict["low"] = np.array(low)
        space_dict["high"] = np.array(high)
        # Dimensions of the sensor data
        dimensions += len(self.data.sensordata)
        space_dict["low"] = np.concatenate([space_dict["low"], np.array([0, np.inf, np.inf, np.inf, 0])])
        space_dict["high"] = np.concatenate([space_dict["high"], np.array([20, np.inf, np.inf, np.inf, 20])])
        # Dimensions of the joint positions and velocities
        dimensions += len(np.concatenate([self.data.qpos.flat, self.data.qvel.flat]))

        space_dict["low"] = np.concatenate([space_dict["low"], np.full(shape=(len(self.data.qvel.flat),), fill_value=np.inf)])
        space_dict["high"] = np.concatenate([space_dict["high"], np.full(shape=(len(self.data.qvel.flat),), fill_value=np.inf)])

        space_dict["low"] = np.concatenate([space_dict["low"], np.full(shape=(len(self.data.qpos.flat),), fill_value=-70)])
        space_dict["high"] = np.concatenate([space_dict["high"], np.full(shape=(len(self.data.qpos.flat),), fill_value=70)])


        # Add observations from environment dynamics
        for dynamic in self.env_dynamics:
            try:
                reward, observations = dynamic(self, self.data, self.model)
                space_dict["low"] = np.concatenate([space_dict["low"], np.full(shape=(len(observations),), fill_value=np.inf)])
                space_dict["high"] = np.concatenate([space_dict["high"], np.full(shape=(len(observations),), fill_value=np.inf)])
                dimensions += len(observations)
            except Exception as e:
                print("ERROR -- env dynamics: " + dynamic.__name__)
                print(e)
                exit()
        # Dimensions of the body positions and velocities
        if self.add_agent_coordinates:
            dimensions += 3
            space_dict["low"] = np.concatenate([space_dict["low"], np.array([-70, -70, -70])])
            space_dict["high"] = np.concatenate([space_dict["high"], np.array([70, 70, 70])])
        if self.add_target_coordinates:
            dimensions += 3
            space_dict["low"] = np.concatenate([space_dict["low"], np.array([-70, -70, -70])])
            space_dict["high"] = np.concatenate([space_dict["high"], np.array([70, 70, 70])])
        # self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(dimensions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=space_dict["low"], high=space_dict["high"], dtype=np.float32)

    def reward(self, agent: str, target: str) -> float:
        """
        Calculates the reward based only on the agent's distance to the target
        Parameters:
            agent (str): name of the agent
            target (str): name of the target
        Returns:
            reward (float): reward for the agent
        """
        distance = math.dist(self.data.body(agent).xipos, self.data.body(target).xipos)
        if self.lastDistance is None:
            self.lastDistance = distance
            reward = 0
        else:
            reward = self.lastDistance - distance
            self.lastDistance = distance
        if self.use_head_sensor:
            if self.data.sensordata.flat[4] < 0.15:
                reward = reward - 0.01
        return reward

    def check_done(self, agent: str, target: str):
        """
        Check whether the current episode is over
        Parameters:
            agent (str): name of the agent
            target (str): name of the target
        Returns:
            done (bool): whether the episode is over
            healthy (bool): whether the agent is flipped upside down
        """
        distance = math.dist(self.data.body(agent).xipos, self.data.body(target).xipos)
        if distance > 20:
            return True, True
        elif self.end_epoch_on_turn:
            if self.data.body(agent).xipos[2] < self._healthy_z_range[0] or self.data.body(agent).xipos[2] > self._healthy_z_range[1]:
                return True, False
            else:
                return False, True
        return False, True

    def calculate_distance(self, object_1, object_2):
        """
        Calculates the distance between object_1 and object_2.

        Parameters:
            object_1 (str or array): name or coordinates of object_1
            object_2 (str or array): name or coordinates of object_2
            
        Returns:
            distance (float): distance between object_1 and object_2
        """
        def __name_to_coordinates(object):
            if isinstance(object, str): 
                try:
                    object = self.data.body(object).xipos
                except:
                    object = self.data.geom(object).xpos
            return object

        object_1 = __name_to_coordinates(object_1) 
        object_2 = __name_to_coordinates(object_2)

        return math.dist(object_1, object_2)
    
    def reset(self):
        """
        Resets the environment.
        """
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        observations = self.__getObs()

        for dynamic in self.env_dynamics:
            reward, new_observations = dynamic(self, self.data, self.model)
            observations = np.concatenate([observations, new_observations])

        self.data_store = {}
        # Reset the debug infos
        self.overAllReward = 0
        self.frame = 0
        self.lastDistance = None
        return observations

    def step(self, action, export_frames=False, skip_frames=5, print_sensor_data=False):
        """
        Perform step in the environment.
        Parameters:
            action (np.array): action to be performed
            export_frames (bool): whether to export the frames
            skip_frames (int): number of frames to skip until the next observation is taken
            print_sensor_data (bool): whether to print the sensor data
        Returns:
            observations (np.array): observations from the environment
            reward (float): reward for the action
            done (bool): whether the episode is over
            info (dict): Dubug info
        """
        
        reward = 0
        # Apply actions to the env
        self.applyAction(action, skip_frames)
        # Get the new observations from the env
        observations = self.__getObs(print_sensor_data)
        # healthy = False

        for dynamic in self.env_dynamics:
            new_reward, obs = dynamic(self, self.data, self.model)
            reward = reward + new_reward
            observations = np.concatenate((observations, obs))

        # Get the reward
        if self.reward_function == None:
            new_reward = self.reward("torso", "target") * 10
        else:
            new_reward = self.reward_function(self, self.data, self.model)
        reward = reward + new_reward

        # Check if the env is done
        if self.done_function == None:
            done, healthy = self.check_done("torso", "target")
            if not healthy:
                reward = -1 * abs(reward) * 5
        else:
            done, new_reward = self.done_function(self, self.data, self.model)
            reward = reward + new_reward

        if self.use_ctrl_cost:
            cost = self.control_cost(action)
            reward = reward - cost

        if type(self.data_store) is not dict:
            print("WARNING: data_store is not a dict. You probably overwrote the variable in one of your custom functions. Don't do that!")

        self.numberOfSteps += 1
        self.frame += 1
        self.overAllReward += reward

        if self.end_epoch_on_turn:
            if self.data.body(self.agent).xipos[2] < self._healthy_z_range[0] or self.data.body(self.agent).xipos[2] > self._healthy_z_range[1]:
                done = True

        if self.frame > self.max_step:
            done = True
        if export_frames:
            self.export_json(self.model, self.data, "test_export/last" + str(self.frame) + ".json")
        
        return observations, reward, done, {"overall_reward":self.overAllReward}

    def control_cost(self, action):
        """
        Calculates the control cost depending on the action
        Parameters:
            action (np.array): action to be performed
        Returns:
            control_cost (float): control cost for the action
        """
        control_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        return control_cost * self.control_factor


    def __getObs(self, print_sensor_data=False) -> np.array:
        """
        Get observations from the environment containing the sensor data, control data, state vector, agent coordinates and target coordinates
        Parameters:
            print_sensor_data (bool): whether to print the sensor data
        Returns:
            observations (np.array): observations from the environment
        """
        # get all sensor data from mujoco
        observations = np.array(self.data.sensordata)
        # get all control data from mujoco
        observations = np.concatenate((observations, np.array(self.data.ctrl)))
        # get coordinate and linear and angular velocity data from mujoco
        state_vector = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

        if print_sensor_data:
            print("State Vector: ", state_vector)

        # concatenate all data
        observations = np.concatenate((observations, state_vector))

        # add agent and target coordinates to observation if they are applied
        if self.add_agent_coordinates:
            observations = np.concatenate((observations, np.array(self.data.body(self.agent).xipos)))
        if self.add_target_coordinates:
            observations = np.concatenate((observations, np.array(self.data.body(self.target).xipos)))
        return observations

    def state_vector(self):
        """Return the position and velocity joint states of the model"""
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
    