import unittest
from MuJoCo_Gym.mujoco_rl import *

class TestSensorInObservationSpace(unittest.TestCase):

    def test_touch_sensor(self):
        '''
        Test case to verify that the touch sensor is included correctly in the observation space.
        '''
        xml_file = "sensor_levels/Model1.xml"
        agents = ["receiver"]
        config_dict = {"xmlPath": xml_file,
                       "agents": agents,
                       "rewardFunctions": [],
                       "doneFunctions": [],
                       "skipFrames": 5,
                       "environmentDynamics": [],
                       "freeJoint": True,
                       "renderMode": False,
                       "maxSteps": 1024,
                       "agentCameras": True}
        env = MuJoCoRL(config_dict=config_dict)
        self.assertEqual(env.observation_space.low, [0])
        self.assertEqual(env.observation_space.high, [20.0])

    def test_accelerometer_sensor(self):
        '''
        Test case to verify that the accelerometer sensor and all other sensors that create the same observation_space is included correctly.
        '''
        xml_file = "sensor_levels/Model2.xml"
        agents = ["receiver"]
        config_dict = {"xmlPath": xml_file,
                       "agents": agents,
                       "rewardFunctions": [],
                       "doneFunctions": [],
                       "skipFrames": 5,
                       "environmentDynamics": [],
                       "freeJoint": True,
                       "renderMode": False,
                       "maxSteps": 1024,
                       "agentCameras": True}
        env = MuJoCoRL(config_dict=config_dict)
        self.assertEqual(env.observation_space.low.tolist(), [-5.0, -5.0, -5.0])
        self.assertEqual(env.observation_space.high.tolist(), [5.0, 5.0, 5.0])

if __name__ == '__main__':
    unittest.main()
