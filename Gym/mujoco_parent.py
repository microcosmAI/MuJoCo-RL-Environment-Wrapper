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
        self.render = render                             # whether to render the environment

        if render:
            self.cam = mj.MjvCamera()                    # Abstract camera
            self.opt = mj.MjvOption()                    # visualization options
            self.__initRender()

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

    def mujocoStep(self):
        """
        Performs a mujoco step.
        """
        mj.mj_step(self.model, self.data)
        if self.render:
            self.__render()
        
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
        agentIndex = self.agents.index(agent)
        sensorData = self.data.sensordata
        sensorDevider = len(sensorData) / len(self.agents)
        sensorData = sensorData[int(sensorDevider * agentIndex): int(sensorDevider * (agentIndex + 1))]
        return sensorData

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
        try:
            dataBody = self.data.body(name)
            modelBody = self.model.body(name)
            infos = {
                "position": dataBody.xipos,
                "mass": modelBody.mass,
                "orientation": mat2eulerScipy(dataBody.xmat),
                "id": dataBody.id,
                "name": dataBody.name,
                "type": "body",
            }
        except Exception as e:
            dataBody = self.data.geom(name)
            infos = {
                "position": dataBody.xpos,
                "orientation": mat2eulerScipy(dataBody.xmat),
                "id": dataBody.id,
                "name": dataBody.name,
                "type": "geom"
            }
        return infos

    def distance(self, object_1, object_2) -> float:
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

    def __exportJson(self):
        pass

    def startRender(self):
        """
        Starts the render window.
        """
        if not self.render:
            self.render = True
            self.__render()

    def endRender(self):
        """
        Ends the render window.
        """
        if self.render:
            self.render = False
            glfw.terminate()

    def __initRender(self):
        """
        Starts the render window
        """
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        glfw.set_scroll_callback(self.window, self.__scroll)
        mj.set_mjcb_control(self.__controller)

    def __render(self):
        """
        Renders the environment. Only works if the environment is created with the render flag set to True.
        Parameters:
            mode (str): rendering mode
        """
        
        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    def __scroll(self, window, xoffset, yoffset):
        """
        Makes the camera zoom in and out when rendered
        Parameters:
            window (glfw.window): the window
            xoffset (float): x offset
            yoffset (float): y offset
        """
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)

    def __controller(self, model, data):
        pass