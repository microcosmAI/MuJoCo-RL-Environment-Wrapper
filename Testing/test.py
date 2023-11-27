from MuJoCo_Gym.mujoco_rl import *
import numpy as np

# xml_files = ["Testing/levels/" + file for file in os.listdir("Testing/levels/")]
xml_files = "Testing/levels/Model1.xml"
agents = ["sender", "receiver"]

class EnvironmentDynamic:
    def __init__(self, environment):
        self.environment = environment
        self.observation_space = {"low": [], "high": []}
        self.action_space = {"low": [], "high": []}

    def dynamic(self, agent, actions):
        return 0, np.array([]), False, {}

config_dict = {"xmlPath":xml_files, 
                   "agents":agents, 
                   "rewardFunctions":[], 
                   "doneFunctions":[], 
                   "skipFrames":5,
                   "environmentDynamics":[],
                   "freeJoint":True,
                   "renderMode":False,
                   "maxSteps":1024,
                   "agentCameras":True}

env = MuJoCoRL(config_dict=config_dict)

for i in range(100):
    env.reset()
    start = time.time()
    for j in range(1024):
        env.step({"sender": env.action_space.sample(), "receiver": env.action_space.sample()})
    end = time.time()
    print("Episode {} done".format(i), "FPS: {}".format(1024 / (end-start)))