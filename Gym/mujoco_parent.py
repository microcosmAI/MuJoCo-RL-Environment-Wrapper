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
from helper import mat2eulerScipy
from abc import ABC, abstractmethod

class MuJoCoParent():
    def __init__(self, xmlPath, exportPath=None, render=False, freeJoint=False, agents=[], skipFrames=1):
        self.xmlPath = xmlPath
        self.exportPath = exportPath

        # Load and create the MuJoCo Model and Data
        self.model = mj.MjModel.from_xml_path(xmlPath)   # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.__cam = mj.MjvCamera()                      # Abstract camera
        self.__opt = mj.MjvOption()                      # visualization options
        self.frame = 0                                   # frame counter

    def applyAction(self, action, skipFrames=1, export=False):
        """
        Applies the actions to the environment.
        arguments:
            action (np.array): The action to be applied to the environment.
            skipFrames (int): The number of frames to skip after applying the action.
            export (bool): Whether to export the environment to a json file.
        """
        self.data.ctrl = action
        for _ in range(skipFrames):
            mj.mj_step(self.model, self.data)
            self.frame += 1
        
    def getSensorData(self) -> np.array:
        """
        Returns the sensor data of the environment.
        returns:
            np.array: The sensor data of the environment.
        """
        return self.data.sensordata
    
    def getSensorData(self, agent) -> np.array:
        """
        Returns the sensor data of a specific agent.
        arguments:
            agent (str): The name of the agent.
        returns:
            np.array: The sensor data of the agent.
        """
        pass

    def getCameraData(self, agent) -> np.array:
        """
        Returns the camera data of a specific agent.
        arguments:
            agent (str): The name of the agent.
        returns:
            np.array: The camera data of the agent.
        """
        pass

    def getData(self, name) -> dict:
        """
        Returns the data of a specific object/geom.
        arguments:
            name (str): The name of the object/geom.
        returns:
            dict: The data of the object/geom.
        """
        pass

    def distance(self, agent1, agent2) -> float:
        """
        Calculates the distance between two objects/geoms
        arguments:
            agent1 (str): The name of the first object/geom.
            agent2 (str): The name of the second object/geom.
        returns:
            float: The distance between the two objects/geoms.
        """
        pass

    def __exportJson(self):
        pass

    def __render(self):
        pass