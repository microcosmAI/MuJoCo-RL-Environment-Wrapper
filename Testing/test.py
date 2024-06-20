from MuJoCo_Gym.mujoco_rl import *

xml_file = "levels/Model1.xml"
agents = ["sender", "receiver"]


config_dict = {"xmlPath": xml_file,
               "agents": agents,
               "rewardFunctions": [],
               "doneFunctions": [],
               "skipFrames": 5,
               "environmentDynamics": [],
               "freeJoint": True,
               "renderMode": True,
               "maxSteps": 1024,
               "agentCameras": True}

env = MuJoCoRL(config_dict=config_dict)

for i in range(100):
    env.reset()
    start = time.time()
    for j in range(1024):
        env.step({"sender": env.action_space.sample(), "receiver": env.action_space.sample()})
    end = time.time()
    print("Episode {} done".format(i), "FPS: {}".format(1024 / (end - start)))
