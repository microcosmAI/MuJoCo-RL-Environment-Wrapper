from MuJoCo_Gym.mujoco_rl import *
import numpy as np
from pettingzoo.test import parallel_api_test

xml_files = "levels/Model1.xml"
agents = ["sender", "receiver"]

config_dict = {"xmlPath": xml_files,
               "agents": agents,
               "rewardFunctions": [],
               "doneFunctions": [],
               "skipFrames": 0,
               "environmentDynamics": [],
               "freeJoint": True,
               "renderMode": False,
               "maxSteps": 1024,
               "agentCameras": True}

env = MuJoCoRL(config_dict=config_dict)

parallel_api_test(env, num_cycles=50)

"""
for i in range(100):
    env.reset()
    start = time.time()
    for j in range(1024):
        env.step({"sender": env.action_space.sample(), "receiver": env.action_space.sample()})
    end = time.time()
    print("Episode {} done".format(i), "FPS: {}".format(1024 / (end-start)))
"""