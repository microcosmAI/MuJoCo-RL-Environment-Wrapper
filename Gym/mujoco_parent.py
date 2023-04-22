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
from gymnasium.spaces import Discrete, Box
from scipy.spatial.transform import Rotation
from helper import mat2eulerScipy
from abc import ABC, abstractmethod
import xmltodict
import collections

class MuJoCoParent():
    def __init__(self, xmlPath, exportPath=None, render=False, freeJoint=False, agents=[], skipFrames=1):
        self.xmlPath = xmlPath
        self.exportPath = exportPath

        #open text file in read mode
        textFile = open(xmlPath, "r")

        #read whole file to a string
        data = textFile.read()
        self.xmlDict = xmltodict.parse(data)

        # Load and create the MuJoCo Model and Data
        self.model = mj.MjModel.from_xml_path(xmlPath)   # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.__cam = mj.MjvCamera()                      # Abstract camera
        self.__opt = mj.MjvOption()                      # visualization options
        self.frame = 0                                   # frame counter
        self.render = render                             # whether to render the environment

        self.freeJoint = freeJoint

        self.agentsActionIndex = {}
        self.agentsObservationIndex = {}

        if render:
            self.previous_time = self.data.time
            self.cam = mj.MjvCamera()                    # Abstract camera
            self.opt = mj.MjvOption()                    # visualization options
            self.__initRender()

    def getObservationSpaceMuJoCo(self, agent):
        """
        Returns the observation space of the environment from all the mujoco sensors.
        returns:
            np.array: The observation space of the environment.
        """
        observationSpace = {"low":[], "high":[]}
        agentDict = self.__findInNestedDict(self.xmlDict, name=agent, filterKey="@name")
        agentSites = self.__findInNestedDict(agentDict, parent="site")
        sensorDict = self.__findInNestedDict(self.xmlDict, parent="sensor")

        indizes = {}
        new_indizes = {}
        index = 0

        # Stores all the sensors and their indizes in the mujoco data object in a dictionary.
        for sensorType in sensorDict:
            for key in sensorType.keys():
                for sensor in sensorType[key]:
                    current = self.data.sensor(sensor["@name"])
                    indizes[current.id] = {"name":sensor["@name"], "data":current.data}
                    if "@site" in sensor.keys():
                        indizes[current.id]["site"] = sensor["@site"]
                        indizes[current.id]["type"] = "rangefinder"
                        indizes[current.id]["cutoff"] = sensor["@cutoff"]
                    if "@objtype" in sensor.keys():
                        indizes[current.id]["site"] = sensor["@objname"]
                        indizes[current.id]["type"] = "frameyaxis"

        # Filters for the sensors that are on the agent and sorts them by number.
        for item in sorted(indizes.items()):
            new_indizes[item[1]["name"]] = {"indizes":[], "site":item[1]["site"], "type":item[1]["type"]}
            if item[1]["type"] == "rangefinder":
                new_indizes[item[1]["name"]]["cutoff"] = item[1]["cutoff"]
            for i in range(len(item[1]["data"])):
                new_indizes[item[1]["name"]]["indizes"].append(index)
                index += 1

        # Stores the indizes of the sensors that are on the agent.
        agentSensors = [current for current in new_indizes.values() if current["site"] in [site["@name"] for site in agentSites]]
        agentIndizes = [current["indizes"] for current in agentSensors]
        agentIndizes = [item for sublist in agentIndizes for item in sublist]
        self.agentsObservationIndex[agent] = agentIndizes

        # Creates the observation space from the sensors.
        for sensorType in agentSensors:
            if sensorType["type"] == "rangefinder":
                observationSpace["low"].append(-1)
                observationSpace["high"].append(float(sensorType["cutoff"]))
            elif sensorType["type"] == "frameyaxis":
                for _ in range(3):
                    observationSpace["low"].append(-360)
                    observationSpace["high"].append(360)
        
        return observationSpace
    
    def getActionSpaceMuJoCo(self, agent):
        """
        Returns the action space of the environment from all the mujoco actuators.
        returns:
            np.array: The action space of the environment.
        """
        actionSpace = {"low":[], "high":[]}
        actionIndexs = []
        agentDict = self.__findInNestedDict(self.xmlDict, name=agent, filterKey="@name")
        agentJoints = self.__findInNestedDict(agentDict, parent="joint")
        if self.freeJoint:
            try:
                freeJoint = agentDict[0]["joint"]
            except:
                raise Exception("The agent {} has to have a free joint".format(agent))
            if freeJoint["@type"] == "free":
                idx = self.model.joint(freeJoint["@name"]).dofadr[0]
                for _ in range(3):
                    actionSpace["low"].append(-1)
                    actionSpace["high"].append(1)
                indizes = [idx, idx+1, idx+4]
                self.agentsActionIndex[agent] = indizes
                return actionSpace
            else:
                raise Exception("The joint of agent {} has to be of type free".format(agent))

        else:
            actuatorDict = self.__findInNestedDict(self.xmlDict, parent="actuator")
            agentJoints = [joint["@name"] for joint in agentJoints]
            for joint in agentJoints:
                agentMotors = self.__findInNestedDict(self.xmlDict, parent="motor", filterKey="@joint", name=joint)
                for motor in agentMotors:
                    actionIndexs.append(actuatorDict[0]["motor"].index(motor))
                    ctrlrange = motor["@ctrlrange"].split(" ")
                    actionSpace["low"].append(float(ctrlrange[0]))
                    actionSpace["high"].append(float(ctrlrange[1]))
            self.agentsActionIndex[agent] = actionIndexs
            return actionSpace

    def applyAction(self, actions, skipFrames=1, export=False):
        """
        Applies the actions to the environment.
        arguments:
            action (dict): The action of every agent to be applied to the environment.
            skipFrames (int): The number of frames to skip after applying the action.
            export (bool): Whether to export the environment to a json file.
        """
        for agent in actions.keys():
            if self.freeJoint:
                self.data.qvel[self.agentsActionIndex[agent]] = actions[agent] * 0.5
            else:
                try:
                    actionIndexs = self.agentsActionIndex[agent]
                    mujoco_actions = actions[agent][:len(self.agentsActionIndex[agent])]
                    self.data.ctrl[actionIndexs] = mujoco_actions
                except IndexError:
                    print("The number of actions for agent {} is not correct.".format(agent))
        for _ in range(skipFrames):
            mj.mj_step(self.model, self.data)
            self.frame += 1
        if self.render and self.data.time - self.previous_time > 1.0/30.0:
            self.previous_time = self.data.time
            self.__render()

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
        sensorData = [self.data.sensordata[i] for i in self.agentsObservationIndex[agent]]
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
            dictionary (str): The data of the object/geom.
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

    # def __getSensors(self, xmlDict):
        

    def __findInNestedDict(self, dictionary, name=None, filterKey="@name", parent=None):
        """
        Finds a key in a nested dictionary.
        Parameters:
            dictionary (dict): The dictionary to search in.
            key (str): The key to search for.
        Returns:
            dictionary (dict): The dictionary containing the key.
        """
        results = []
        if isinstance(dictionary, dict):
            if parent is not None and parent in dictionary.keys():
                if isinstance(dictionary[parent], list):
                    for item in dictionary[parent]:
                        if (filterKey in item.keys() and item[filterKey] == name) or not name:
                            results.append(item)
                elif not name or dictionary[parent][filterKey] == name:
                    results.append(dictionary[parent])
            for key, value in dictionary.items():
                if (key == filterKey or not filterKey) and (value == name or not name) and not parent:
                    results.append(dictionary)
                elif isinstance(value, (dict, list)):
                    results.extend(self.__findInNestedDict(value, name, filterKey=filterKey, parent=parent))
        elif isinstance(dictionary, list):
            for item in dictionary:
                if isinstance(item, (dict, list)):
                    results.extend(self.__findInNestedDict(item, name, filterKey=filterKey, parent=parent))
        return results


    def __controller(self, model, data):
        pass