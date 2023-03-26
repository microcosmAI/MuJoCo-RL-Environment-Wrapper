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
        self.data.ctrl = action
        for _ in range(skipFrames):
            mj.mj_step(self.model, self.data)
            self.frame += 1
        
    def getSensorData(self):
        return self.data.sensordata
    
    def getSensorData(self, agent):
        pass

    def getCameraData(self, agent):
        pass

    def getData(self, name):
        pass

    def distance(self, agent1, agent2):
        pass

    def __exportJson(self):
        pass

    def __render(self):
        pass