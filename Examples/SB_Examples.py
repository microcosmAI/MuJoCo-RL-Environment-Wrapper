import random
from MuJoCo_Gym.mujoco_rl import MuJoCo_RL
from MuJoCo_Gym.single_agent_wrapper import Single_Agent_Wrapper
import time
import copy
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune
import numpy as np

def reward_function(mujoco_gym, agent):
    # Creates all the necessary fields to store the needed data within the dataStore at timestep 0
    distance = mujoco_gym.distance(agent, mujoco_gym.dataStore[agent]["current_target"])
    new_reward = mujoco_gym.dataStore[agent]["distance"] - distance
    mujoco_gym.dataStore[agent]["distance"] = copy.deepcopy(distance)
    reward = new_reward * 10
    return reward

def done_function(mujoco_gym, agent):
    if mujoco_gym.dataStore[agent]["distance"] <= 1 or mujoco_gym.dataStore[agent]["distance"] > 15:
        print("DONE")
        return True
    else:
        return False
    
class Coordinates():
    def __init__(self, mujoco_gym):
        self.mujoco_gym = mujoco_gym
        self.observation_space = {"low":[-40, -40, -40, -40, -40, -40], "high":[40, 40, 40, 40, 40, 40]}
        self.action_space = {"low":[], "high":[]}
        # The datastore is used to store and preserve data over one or multiple timesteps
        self.dataStore = {}

    def dynamic(self, agent, actions):

        if "targets" not in self.mujoco_gym.dataStore.keys():
            self.mujoco_gym.dataStore["targets"] = self.mujoco_gym.filterByTag("target")
            self.mujoco_gym.dataStore[agent]["current_target"] = self.mujoco_gym.dataStore["targets"][random.randint(0, len(self.mujoco_gym.dataStore["targets"]) - 1)]["name"]
            distance = self.mujoco_gym.distance(agent, self.mujoco_gym.dataStore[agent]["current_target"])
            self.mujoco_gym.dataStore[agent]["distance"] = distance
        position = np.array(self.mujoco_gym.getData(self.mujoco_gym.dataStore[agent]["current_target"])["position"])
        agent_position = np.array(self.mujoco_gym.getData(agent)["position"])
        return 0, np.concatenate((position, agent_position))
    
environment_path = "Examples/Environment/MultiEnvs.xml"
info_path = "Examples/Environment/info_example.json"
agents = ["agent1_torso"]
config_dict = {"xmlPath":environment_path, "infoJson":info_path, "agents":agents, "rewardFunctions":[reward_function], "doneFunctions":[done_function], "environmentDynamics":[Coordinates], "freeJoint":True, "renderMode":False, "maxSteps":8192}
environment = MuJoCo_RL(config_dict)


ray.init(num_gpus=1)
config = PPOConfig()

# Update the config object.
config.training(lr=tune.grid_search([0.001, 0.0001]), clip_param=0.2)

# Set the config object's env
config = config.environment(env=MuJoCo_RL)
config["env_config"] = config_dict
config["model"]["fcnet_hiddens"] = [512,512]
config["model"]["use_lstm"] = False
config["model"]["lstm_cell_size"] = 128

print(config.to_dict())

# Use to_dict() to get the old-style python config dict
# when running with tune.
tune.Tuner(  
    "PPO",
    run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
    param_space=config.to_dict(),
).fit()