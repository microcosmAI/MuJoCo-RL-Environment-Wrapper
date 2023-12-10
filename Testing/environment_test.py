import random
import numpy as np
from MuJoCo_Gym.mujoco_rl import MuJoCoRL
import os
import unittest

class EnvironmentDynamic:
    """ ToDo: desctiption """

    def __init__(self, mujoco_gym):
        """Initializes Environment dynamic and defines observation space

        Parameters:
            mujoco_gym (SingleAgent): Instance of single agent environment
        """
        self.mujoco_gym = mujoco_gym
        self.observation_space = {"low": [-70, -70, -70], "high": [70, 70, 70]}
        self.action_space = {"low": [], "high": []}
    
    def dynamic(self, agent, action):
        """Update target if the agent is close enough

        Returns: 
            reward (int): Reward for the agent, always 0 for environment_dynamics
            target_coordinates (np.array): Coordinates of the target
        """
        agent_coordinates = self.mujoco_gym.get_data(agent)

        if agent == "sender":
            self.mujoco_gym.data_store["coordinates"] = agent_coordinates["position"]

        return 0, agent_coordinates["position"], False, {}
    

class TestDataStoreIntegration(unittest.TestCase):
    def setUp(self):
        # xml_files = ["Testing/levels/" + file for file in os.listdir("Testing/levels/")]
        xml_files = "Testing/levels/Model1.xml"
        agents = ["sender", "receiver"]

        config_dict = {"xmlPath": xml_files,
                       "agents": agents,
                       "skipFrames": 5,
                       "environmentDynamics": [EnvironmentDynamic],
                       "freeJoint": True,
                       "renderMode": False,
                       "maxSteps": 1024,
                       "agentCameras": False}
        self.mujoco_gym = MuJoCoRL(config_dict=config_dict)
        self.mujoco_gym.reset()

    def test_data_store_integration(self):
        agent_action = {"sender": np.array([0, 0, 0]), "receiver": np.array([0, 0, 0])}
        self.mujoco_gym.step(agent_action)

        # Access the data_store and check if "coordinates" key is present
        self.mujoco_gym.data_store.set_agent("sender")
        self.assertTrue("coordinates" in self.mujoco_gym.data_store.keys())

        # Access the data_store and check if the value for "coordinates" is correct
        agent_coordinates = self.mujoco_gym.get_data("sender")["position"]
        self.assertEqual(self.mujoco_gym.data_store["coordinates"].tolist(), agent_coordinates.tolist())

    def test_agent_assignment(self):
        agent_action = {"sender": np.array([0, 0, 0]), "receiver": np.array([0, 0, 0])}
        self.mujoco_gym.step(agent_action)

        self.mujoco_gym.data_store.set_agent("sender")
        self.assertEqual(self.mujoco_gym.data_store.current_agent, "sender")
        self.mujoco_gym.data_store.set_agent("receiver")
        self.assertNotIn("coordinates", self.mujoco_gym.data_store.keys())


if __name__ == '__main__':
    unittest.main()