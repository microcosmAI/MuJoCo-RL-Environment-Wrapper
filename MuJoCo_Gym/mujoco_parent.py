import math
import xmltodict
import ctypes
import random
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import json
import time

try:
    from helper import mat2euler_scipy
    from sensor import process_sensors, create_sensor_observation_space
except:
    from MuJoCo_Gym.helper import mat2euler_scipy
    from MuJoCo_Gym.sensor import process_sensors, create_sensor_observation_space


class MuJoCoParent:
    """
    The MuJoCoParent class represents a parent class for MuJoCo environments.

    Attributes:
        xml_paths (str or list): Path(s) to xml-file(s) containing finished world environment
        export_path (str): Path to output directory
        render (bool): Boolean for rendering option (set to "True" for rendering)
        free_joint (bool): Boolean for free-joints option (set to "True" for unrestricted movement)
        agent_cameras (bool): Boolean for cameras option (set to "True" for enabling cameras)
        sensor_resolution (tuple): Resolution of the sensors (default is (64, 64))

    Methods:
        __init__(self, xml_paths, export_path, render, free_joint, agent_cameras, sensor_resolution):
            Initializes the MuJoCoParent class.

        __init_environment(self):
            Initializes the environment by setting up the MuJoCo model, data, camera, and visualization options.

        __process_sensor(self, sensor, indices, key):
            Process a sensor and update the indices dictionary.

        __create_sensor_index_dict(self, sensor_dict):
            Stores all the sensors and their indices in the mujoco data object in a dictionary.

        get_observation_space_mujoco(self, agent):
            Returns the observation space of the environment from all the mujoco sensors.

        get_action_space_mujoco(self, agent):
            Returns the action space of the environment from all the mujoco actuators.

        apply_action(self, actions, skip_frames):
            Applies the actions to the environment.

        reset(self):
            Resets the environment to its initial state.

        mujoco_step(self):
            Performs a mujoco step.

        get_sensor_data(self, agent):
            Returns the sensor data of a specific agent.

        get_data(self, name):
            Returns the data of a specific agent.
    """

    def __init__(self, xml_paths, export_path: str = None, render: bool = False, free_joint: bool = False,
                 agent_cameras: bool = False, sensor_resolution=(64, 64)):
        """Initializes MujocoParent class

        Parameters:
             xml_paths (str or list): Path(s) to xml-file(s) containing finished world environment
             export_path (str): Path to output directory
             render (bool): Boolean for rendering option (set to "True" for rendering)
             free_joint (bool): Boolean for free-joints option (set to "True" for unrestricted movement)
             agent_cameras (bool): Boolean for cameras option (set to "True" for enabling cameras)
        """
        self.xml_paths = xml_paths
        self.export_path = export_path
        self.render = render
        self.free_joint = free_joint
        self.agent_cameras = agent_cameras
        self.sensorWindow = None
        self.scene = None

        # Assigns "xml_path" var; depends on user input if single or multiple xml-files are given.
        # If multiple ones exist, a random xml-file is picked, read and parsed to a string.
        self.xml_path = None
        if isinstance(xml_paths, str):
            self.xml_path = xml_paths
        elif isinstance(xml_paths, list):
            self.xml_path = random.choice(xml_paths)
        text_file = open(self.xml_path, "r")
        data = text_file.read()
        self.xml_dict = xmltodict.parse(data)

        self.sensor_resolution = sensor_resolution
        self.rgb_sensors = {}

        if render or agent_cameras:
            glfw.init()
            # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self.opt = mj.MjvOption()
            self.window = glfw.create_window(1200, 900, "Demo", None, None)
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

        self.__init_environment()
        self.frame = 0                                   # Frame counter

        self.agents_action_index = {}
        self.agents_observation_index = {}
        self.agent_observations_id = []

        if render:
            self.previous_time = self.data.time
            self.cam = mj.MjvCamera()                    # Abstract camera
            self.__init_render()

    def __init_environment(self):
        """Initializes environment

        This function initializes the environment by setting up the MuJoCo model, data, camera, and visualization options.
        If the scene is not provided, it creates a new scene and context for rendering and agent cameras.
        If agent cameras are enabled, it also initializes the RGB sensor.
        """
        self.model = mj.MjModel.from_xml_path(self.xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.__cam = mj.MjvCamera()                      # Abstract camera
        self.__opt = mj.MjvOption()                      # visualization options

        if self.scene == None:
            if self.render or self.agent_cameras:
                self.scene = mj.MjvScene(self.model, maxgeom=10000)
                self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        if self.agent_cameras:
            self.__init_rgb_sensor()

    def __process_sensor(self, sensor, indices, key):
        """
        Process a sensor and update the indices dictionary.

        Args:
            sensor (dict): The sensor dictionary containing sensor information.
            indices (dict): The dictionary to store the sensor indices.
            key (str): The key indicating the type of sensor.

        Returns:
            None
        """
        current = self.data.sensor(sensor["@name"])
        indices[current.id] = {"name": sensor["@name"], "data": current.data}

        if key in ("rangefinder" , "touch" , "accelerometer"):
            indices[current.id]["site"] = sensor["@site"]
            indices[current.id]["type"] = key
            indices[current.id]["cutoff"] = sensor["@cutoff"]
        if key == "framexaxis":
            indices[current.id]["site"] = sensor["@objname"]
            indices[current.id]["type"] = "frameyaxis"

    def __create_sensor_index_dict(self, sensor_dict):
        """
        Stores all the sensors and their indices in the mujoco data object in a dictionary.

        Parameters:
        - sensor_dict (list): A list containing sensor information.
        - indices (dict): Dictionary to store sensor indices.

        Returns:
        None
        """
        indices = {}
        for sensor_type in sensor_dict:
            if sensor_type is not None:
                for key in sensor_type.keys():
                    if isinstance(sensor_type[key], list):
                        for sensor in sensor_type[key]:
                            self.__process_sensor(sensor, indices, key)
                    elif isinstance(sensor_type[key], dict):
                        sensor = sensor_type[key]
                        self.__process_sensor(sensor, indices, key)
        return indices
    
    def retrieve_mujoco_all_indices(self, joint_dicts):
        """
        Given a MuJoCo model and a list of dictionaries representing joint information,
        retrieve all qpos and qvel indices for each joint name using the model's jnt_qposadr and jnt_dofadr.

        :param model: MuJoCo model object containing joint information.
        :param joint_dicts: List of dictionaries containing joint information.
        :return: Two lists, one with all qpos indices and one with all qvel indices.
        """
        # Lists to store all qpos and qvel indices
        qpos_indices = []
        qvel_indices = []

        # Iterate over each dictionary in the list
        for joint_dict in joint_dicts:
            # Extract the joint name
            joint_name = joint_dict.get('@name')
            if joint_name:
                # Find the joint ID using the name
                joint_id = self.model.joint(joint_name).id

                # Retrieve the start index for qpos and qvel
                qpos_start_index = self.model.jnt_qposadr[joint_id]
                qvel_start_index = self.model.jnt_dofadr[joint_id]

                # Number of position DOFs for this joint (typically 1 for hinge, but can be more for free joints)
                # If current joint is a free joint
                if self.model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
                    # Number of position DOFs for this joint = 7
                    qpos_dof = 7
                else:
                    # Number of position DOFs for this joint = 1
                    qpos_dof = 1
                
                if self.model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
                    # Number of velocity DOFs for this joint = 6
                    qvel_dof = 6
                else:
                    # Number of velocity DOFs for this joint = 1
                    qvel_dof = 1

                # Append all qpos indices for this joint
                qpos_indices.extend(range(qpos_start_index, qpos_start_index + qpos_dof))
                # Append all qvel indices for this joint
                qvel_indices.extend(range(qvel_start_index, qvel_start_index + qvel_dof))

        return qpos_indices, qvel_indices

    def get_observation_space_mujoco(self, agent: str) -> np.array:
        """Returns the observation space of the environment from all the mujoco sensors

        Parameters:
            agent: Name of agent mujoco (top level)

        Returns:
            observation_space (np.array): The observation space of the environment
        """
        agent_dict = self.__find_in_nested_dict(self.xml_dict, name=agent, filter_key="@name")
        agent_sites = self.__find_in_nested_dict(agent_dict, parent="site")
        sensor_dict = self.__find_in_nested_dict(self.xml_dict, parent="sensor")

        world_dict = self.__find_in_nested_dict(self.xml_dict, parent="worldbody")
        joint_dict = self.__find_in_nested_dict(world_dict, parent="joint")
        agent_joints = self.__find_in_nested_dict(agent_dict, parent="joint")

        if self.agent_cameras:
            self.__find_agent_camera(agent_dict, agent)

        # Stores all the sensors and their indices in the mujoco data object in a dictionary.
        indices = self.__create_sensor_index_dict(sensor_dict)

        # Retrieve all qpos and qvel indices for each joint name using the model's jnt_qposadr and jnt_dofadr.
        qpos_indices, qvel_indices = self.retrieve_mujoco_all_indices(joint_dicts=joint_dict)

        # Extracts the indices of the sensors that are attached to the agent.
        agent_indices, agent_sensors = process_sensors(indices, agent_sites)
        self.agents_observation_index[agent] = {"sensors": agent_indices, "qpos": qpos_indices, "qvel": qvel_indices}

        # Creates the observation space from the sensors and its corresponding indizes.
        observation_space = create_sensor_observation_space(agent_sensors)

        observation_space["low"] += [-np.inf for i in qpos_indices]
        observation_space["high"] += [np.inf for i in qpos_indices]

        observation_space["low"] += [-np.inf for i in qvel_indices]
        observation_space["high"] += [np.inf for i in qvel_indices]

        return observation_space

    def get_action_space_mujoco(self, agent: str) -> np.array:
        """Returns the action space of the environment from all the mujoco actuators

        Parameters:
            agent (str): Name of agent mujoco (top level)

        Returns:
            action_space (np.array): The action space of the environment
        """
        action_space = {"low": [], "high": []}
        action_indexs = []
        agent_dict = self.__find_in_nested_dict(self.xml_dict, name=agent, filter_key="@name")
        agent_joints = self.__find_in_nested_dict(agent_dict, parent="joint")
        if self.free_joint:
            try:
                free_joint = agent_dict[0]["joint"]
            except ValueError:
                raise Exception(f"The agent {agent} has to have a free joint")
            if free_joint["@type"] == "free":
                idx = self.model.joint(free_joint["@name"]).dofadr[0]
                for _ in range(3):
                    action_space["low"].append(-1)
                    action_space["high"].append(1)
                indices = [idx, idx+1, idx+5]
                self.agents_action_index[agent] = indices
                return action_space
            else:
                raise Exception(f"The joint of agent {agent} has to be of type free")

        else:
            actuator_dict = self.__find_in_nested_dict(self.xml_dict, parent="actuator")
            agent_joints = [joint["@name"] for joint in agent_joints]
            for joint in agent_joints:
                agent_motors = self.__find_in_nested_dict(self.xml_dict, parent="motor", filter_key="@joint", name=joint)
                for motor in agent_motors:
                    action_indexs.append(actuator_dict[0]["motor"].index(motor))
                    ctrlrange = motor["@ctrlrange"].split(" ")
                    action_space["low"].append(float(ctrlrange[0]))
                    action_space["high"].append(float(ctrlrange[1]))
            self.agents_action_index[agent] = action_indexs
            return action_space

    def apply_action(self, actions: dict, skip_frames: int = 1):
        """Applies the actions to the environment.

        Parameters:
            actions (dict): The action of every agent to be applied to the environment.
            skip_frames (int): The number of frames to skip after applying the action.
        """
        for agent in actions.keys():
            if self.free_joint:
                self.data.qvel[self.agents_action_index[agent]] = actions[agent]
            else:
                try:
                    action_indexs = self.agents_action_index[agent] # ToDo: mistake of "s"?
                    mujoco_actions = actions[agent][:len(self.agents_action_index[agent])]
                    self.data.ctrl[action_indexs] = mujoco_actions
                except IndexError:
                    print(f"The number of actions for agent {agent} is not correct.")

        for _ in range(skip_frames):
            mj.mj_step(self.model, self.data)
            self.frame += 1
        if self.render and self.data.time - self.previous_time > 1.0/30.0:
            self.previous_time = self.data.time
            self.__render()

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            sensor_data (numpy.ndarray): The sensor data after the reset.
        """
        if isinstance(self.xml_paths, str):
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
        elif isinstance(self.xml_paths, list):
            self.xml_path = random.choice(self.xml_paths)
            self.__init_environment()
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.cam = mj.MjvCamera()
        self.previous_time = self.data.time
        return self.get_sensor_data()

    def mujoco_step(self):
        """Performs a mujoco step """
        mj.mj_step(self.model, self.data)
        if self.render:
            self.__render()

    def get_sensor_data(self, agent: str = None) -> np.array:
        """Returns the sensor data of a specific agent

        Parameters:
            agent (str): The name of the agent.
        returns:
            np.array: The sensor data of the agent.
        """
        if agent is not None:
            sensor_data = [self.data.sensordata[i] for i in self.agents_observation_index[agent]["sensors"]]
            return sensor_data
        else:
            return self.data.sensordata
        
    def get_observations(self, agent: str) -> np.array:
        """Returns the observation of a specific agent

        Parameters:
            agent (str): The name of the agent.

        Returns:
            np.array: The observation of the agent.
        """
        sensor_data = self.get_sensor_data(agent)
        qpos = [self.data.qpos[i] for i in self.agents_observation_index[agent]["qpos"]]
        qvel = [self.data.qvel[i] for i in self.agents_observation_index[agent]["qvel"]]
        return np.array(sensor_data + qpos + qvel)

    def get_data(self, name: str) -> dict:
        """Returns the data of a specific object/geom

        Parameters:
            name (str): The name of the object/geom

        Returns:
            infos (dict): The data of the object/geom
        """
        try:
            data_body = self.data.body(name)
            model_body = self.model.body(name)
            infos = {
                "position": data_body.xipos,
                "mass": model_body.mass,
                "orientation": mat2euler_scipy(data_body.xmat),
                "id": data_body.id,
                "name": data_body.name,
                "type": "body",
            }
        except Exception as e:
            data_geom = self.data.geom(name)
            model_geom = self.model.geom(name)
            infos = {
                "position": data_geom.xpos,
                "orientation": mat2euler_scipy(data_geom.xmat),
                "id": data_geom.id,
                "name": data_geom.name,
                "type": "geom",
                "color": model_geom.rgba,
                "shape": model_geom.type
            }
        return infos

    def distance(self, object_1, object_2) -> float:
        """Calculates the distance between object_1 and object_2

        Parameters:
            object_1 (str or array): Name or coordinates of object_1
            object_2 (str or array): Name or coordinates of object_2

        Returns:
            distance (float): Distance between object_1 and object_2
        """
        def __name_to_coordinates(object):  # ToDo: what is object here and where does it come from?
            if isinstance(object, str):
                try:
                    object = self.data.body(object).xipos
                except:
                    object = self.data.geom(object).xpos
            return object

        object_1 = __name_to_coordinates(object_1)
        object_2 = __name_to_coordinates(object_2)

        return math.dist(object_1, object_2)

    def collision(self, geom_1, geom_2) -> bool:
        """Checks if geom_1 and geom_2 are colliding

        Parameters:
            geom_1 (str or int): Name or id of geom_1
            geom_2 (str or int): Name or id of geom_2

        Returns:
            collision (bool): True if geom_1 and geom_2 are colliding, False otherwise
        """
        try:
            if isinstance(geom_1, str):
                geom_1 = self.data.geom(geom_1).id
        except Exception as e:
            raise Exception(f"Collision object {geom_1} not found in data")
        try:
            if isinstance(geom_2, str):
                geom_2 = self.data.geom(geom_2).id
        except Exception as e:
            raise Exception(f"Collision object {geom_2} not found in data")
        try:
            collision = [self.data.contact[i].geom1 == geom_1 and self.data.contact[i].geom2 == geom_2
                         for i in range(self.data.ncon)]
            collision2 = [self.data.contact[i].geom1 == geom_2 and self.data.contact[i].geom2 == geom_1
                         for i in range(self.data.ncon)]
        except Exception as e:
            raise Exception(f"One of the two objects {geom_1}, {geom_2} not found in data")
        return any(collision + collision2)

    def __export_json(self):
        # ToDo: implement
        pass

    def start_render(self):
        """Starts the render window """
        if not self.render:
            self.render = True
            self.__render()

    def end_render(self):
        """Ends the render window """
        if self.render:
            self.render = False
            glfw.terminate()

    def __init_rgb_sensor(self):
        """Initializes the rgb sensor """
        if self.sensorWindow == None:
            glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self.sensorWindow = glfw.create_window(self.sensor_resolution[0],
                                                self.sensor_resolution[1],
                                                "RGB Sensor", None, None)

    def __find_agent_camera(self, agent_dict: dict, agent: str):
        """Finds the camera of a specific agent

        Parameters:
            agent_dict (dict): ToDo
            agent (str): The name of the agent
        """
        cameras = self.__find_in_nested_dict(agent_dict, parent="camera")

        self.rgb_sensors[agent] = []
        for camera in cameras:
            self.rgb_sensors[agent].append(camera["@name"])

    def __get_specific_camera(self, camera: str):
        """ Returns the image data for a specific camera

        Parameters:
            camera (str): The name of the camera
        Returns:
            np.array: The data from the camera
        """
        sensor = mj.MjvCamera()
        sensor.type = 2
        sensor.fixedcamid = self.model.camera(camera).id
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.sensorWindow)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, sensor,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)
        data = np.array(self.__get_rgbd_buffer(viewport, self.context))

        return data.reshape((viewport_width, viewport_height, 3))

    def get_camera_data(self, cam_object: str) -> np.array:
        """Returns the data of all cameras attached to an agent

        Parameters:
            cam_object (str): The name of the agent or the camera.
        Returns:
            np.array: The data from all cameras.
        """
        if cam_object in self.rgb_sensors.keys():
            all_camera_data = []
            for camera in self.rgb_sensors[cam_object]:
                data = self.__get_specific_camera(camera)
                all_camera_data.append(data)
            return np.array(all_camera_data)
        else:
            return self.__get_specific_camera(cam_object)

    @staticmethod
    def __get_rgbd_buffer(viewport, context):
        """Returns colored buffer image #ToDo: correct description?

        Parameters:
            viewport (ToDo): ToDo
            context (ToDo): ToDo

        Returns:
            color_image (ToDo): colored buffer image #ToDo: correct description?
        """
        # Use preallocated buffer to fetch color buffer and depth buffer in OpenGL
        color_buffer = (ctypes.c_ubyte * (viewport.height * viewport.width * 3))()
        depth_buffer = (ctypes.c_float * (viewport.height * viewport.width * 4))()
        mj.mjr_readPixels(color_buffer, depth_buffer, viewport, context)

        rgb = color_buffer
        color_image = np.copy(rgb)
        return color_image

    def __init_render(self):
        """
        Initializes the visualization data structures and sets up the rendering environment.
        """

        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)

        glfw.set_scroll_callback(self.window, self.__scroll)
        mj.set_mjcb_control(self.__controller)

    def __render(self):
        """Renders the environment. Only works if the environment is created with the render flag set to True """
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

    def __scroll(self, window, x_offset, y_offset):
        """
        Scroll the camera in the MuJoCo environment.

        Parameters:
            window: The window object.
            x_offset: The horizontal offset of the scroll.
            y_offset: The vertical offset of the scroll.
        """
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * y_offset, self.scene, self.cam)

    def __find_in_nested_dict(self, dictionary: dict, name: str = None, filter_key: str = "@name", parent=None) -> list:
        """Finds a key in a nested dictionary

        Parameters:
            dictionary (dict): The dictionary to search in
            name (str): Name to search for
            filter_key (str): The key to search for
            parent (ToDo): ToDo

        Returns:
            results (list(dict)): The dictionary containing the key
        """
        results = []
        if isinstance(dictionary, dict):
            if parent is not None and parent in dictionary.keys():
                if isinstance(dictionary[parent], list):
                    for item in dictionary[parent]:
                        if (filter_key in item.keys() and item[filter_key] == name) or not name:
                            results.append(item)
                elif not name or dictionary[parent][filter_key] == name:
                    results.append(dictionary[parent])
            for key, value in dictionary.items():
                if (key == filter_key or not filter_key) and (value == name or not name) and not parent:
                    results.append(dictionary)
                elif isinstance(value, (dict, list)):
                    results.extend(self.__find_in_nested_dict(value, name, filter_key=filter_key, parent=parent))
        elif isinstance(dictionary, list):
            for item in dictionary:
                if isinstance(item, (dict, list)):
                    results.extend(self.__find_in_nested_dict(item, name, filter_key=filter_key, parent=parent))
        return results

    def __controller(self, model, data):
        # ToDo: is this intended?
        pass
