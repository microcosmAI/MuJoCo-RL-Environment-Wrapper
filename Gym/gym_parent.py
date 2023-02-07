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
    def __init__(self, xmlPath, infoJson="", exportPath=None, render=False, **kwargs):
        self.xmlPath = xmlPath
        self.exportPath = exportPath

        jsonFile = open(infoJson)
        self.infoJson = json.load(jsonFile)
        self.xmlTree = ET.parse(self.xmlPath)

        # Load and create the MuJoCo Model and Data
        self.model = mj.MjModel.from_xml_path(xmlPath)   # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.cam = mj.MjvCamera()                        # Abstract camera
        self.opt = mj.MjvOption()                        # visualization options
        self.frame = 0                                   # frame counter

        if render:
            self.__initializeWindow()
        
    def applyAction(self, action, skipFrames=1, export=False):
        self.data.ctrl = action
        for _ in range(skipFrames):
            mj.mj_step(self.model, self.data)
            self.frame += 1
        if export:
            self.exportJson(self.model, self.data, self.exportPath + "/frame" + str(self.frame) + ".json")

    def render(self, mode='human'):
        """
        Renders the environment. Only works if the environment is created with the render flag set to True.
        Parameters:
            mode (str): rendering mode
        """

        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        #print camera configuration (help to initialize the view)
        if (self.print_camera_config==1):
            print('cam.azimuth =',self.cam.azimuth,';','cam.elevation =',self.cam.elevation,';','cam.distance = ',self.cam.distance)
            print('cam.lookat =np.array([',self.cam.lookat[0],',',self.cam.lookat[1],',',self.cam.lookat[2],'])')

        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    def end(self):
        """
        Close the render window
        """
        glfw.terminate()

    def filterByTag(self, tag):
        """
        Filter environment for object with specific tag
        Parameters:
            tag (str): tag to be filtered for
        Returns:
            filtered (list): list of objects with the specified tag
        """
        filtered = []
        for object in self.infoJson["Objects"]:
            if tag in object["Tags"]:
                data = self.get_data(object["Name"])
                data["tags"] = object["Tags"]
                data["description"] = object["Description"]
                filtered.append(data)
        return filtered

    def findObject(self, data, target, quats=[]):
        """
        Find the object in the xml tree.
        Parameters:
            data (xml.ElementTree): xml tree
            target (str): name of the object to be found
            quats (list): list of quaternions (only used for recursion)
        Returns:
            quats (list): list of object parent names
        """
        if "name" in data.attrib.keys():
            if data.attrib["name"] == target:
                del quats[0]
                del quats[-1]
                return quats
        for child in data:
            newQuats = quats.copy()
            if "name" in child.attrib.keys():
                if child.tag == "body":
                    for i in range(self.model.nbody):
                        if self.model.body(i).name == child.attrib["name"]:
                            newQuats.append({child.tag:child.attrib["name"], "id":i})
                    if len(newQuats) == len(quats):
                        newQuats.append({child.tag:child.attrib["name"]})
            else:
                newQuats.append({child.tag:"none"})
            result = self.find_object(child, target, newQuats)
            if result != []:
                return result
        return []
    
    def get_data(self, object_1: str):
        """
        Returns the data of the object
        Parameters:
            object_1 (str): name of the object
        Returns:
            data (np.array): data of the object
        """
        try:
            data_body = self.data.body(object_1)
            model_body = self.model.body(object_1)
            infos = {
                "position": data_body.xipos,
                "mass": model_body.mass,
                "orientation": mat2eulerScipy(data_body.xmat),
                "id": data_body.id,
                "name": data_body.name,
                "type": "body",
            }
        except Exception as e:
            print(e)
            data_geom = self.data.geom(object_1)
            model_geom = self.model.geom(object_1)
            infos = {
                "position": data_geom.xpos,
                "orientation": mat2eulerScipy(data_geom.xmat),
                "id": data_geom.id,
                "name": data_geom.name,
                "type": "geom"
            }
        return infos

    def exportJson(self, model, data, filename):
        """
        Export the model and data to a json file
        Parameters:
            model (Mujoco Model): model to be exported
            data (Mujoco Data): data to be exported
            filename (str): name of the file to be exported to
        """
        export_dict = {}
        for i in range(model.ngeom):
            name = model.geom(i).name
            if name != "":
                parents = self.findObject(self.xml_tree.getroot(), name, quats=[])
                if parents != []:
                    euler = [0, 0, 0]
                    bodyEuler = mat2eulerScipy(data.body(parents[0]["id"]).xmat)
                    objectEuler = mat2eulerScipy(data.geom(i).xmat)
                    euler = [objectEuler[0] - bodyEuler[0], objectEuler[1] - bodyEuler[1], objectEuler[2] - bodyEuler[2]]
                else:
                    euler = mat2eulerScipy(data.geom(i).xmat)
            else:
                euler = mat2eulerScipy(data.geom(i).xmat)
            xpos = data.geom(i).xpos
            export_dict[i] = {
                "pos": list(xpos),
                "quat": list(euler),
                "size": list(model.geom(i).size),
                "type": int(model.geom(i).type[0]),
                "name": str(model.geom(i).name),
            }
        json_object = json.dumps(export_dict, indent=4)
        with open(filename, "w") as outfile:
            outfile.write(json_object)

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

    def __initializeWindow(self):
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

    def __controller(self, model, data):
        pass

    def distance(self, agent: str, target: str):
        """
        Calculates the distance between the agent and the target
        Parameters:
            agent (str): name of the agent
            target (str): name of the target
        Returns:
            distance (float): distance between the agent and the target
        """
        return math.dist(self.data.body(agent).xipos, self.data.body(target).xipos)