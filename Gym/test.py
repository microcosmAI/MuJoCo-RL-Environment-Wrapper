from mujoco_rl import MuJoCo_RL
import cv2
import time
import random
import numpy as np
import tensorflow as tf
from ray import tune
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import ppo, a3c
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
import ray
from ray import tune
from ray import air
import copy
from ray.tune import CLIReporter
from inputimeout import inputimeout

class Pick_Up_Dynamic():
    def __init__(self, mujoco_gym):
        """
        Initializes the Pick-up dynamic and defines observation space. 

        Parameters:
            mujoco_gym (SingleAgent): instance of single agent environment
        """
        self.mujoco_gym = mujoco_gym
        self.observation_space = {"low":[-70, -70, -70, 0, -70, -70, -70], "high":[70, 70, 70, 1, 70, 70, 70]}
        self.action_space = {"low":[], "high":[]}

    def dynamic(self, agent, actions):
        """
        Update target and add the inventory to the agent as an observation. 

        Returns: 
            reward (int): reward for the agent
            current_target_coordinates_with_inventory (ndarray): concatenation of current_target and inventory
        """
        reward = 0
        if "inventory" not in self.mujoco_gym.dataStore[agent].keys():
            self.mujoco_gym.dataStore[agent]["inventory"] = [0]
        if "targets" not in self.mujoco_gym.dataStore[agent].keys():
            self.mujoco_gym.dataStore["targets"] = self.mujoco_gym.filterByTag("target")
            self.mujoco_gym.dataStore[agent]["current_target"] = self.mujoco_gym.dataStore["targets"][random.randint(0, len(self.mujoco_gym.dataStore["targets"]) - 1)]["name"]
        distance = self.mujoco_gym.distance(agent, self.mujoco_gym.dataStore[agent]["current_target"])
        if distance < 2:
            if self.mujoco_gym.dataStore[agent]["inventory"][0] == 0:
                self.mujoco_gym.dataStore[agent]["inventory"][0] = 1
                reward = 1
            elif self.mujoco_gym.dataStore[agent]["inventory"][0] == 1:
                self.mujoco_gym.dataStore[agent]["inventory"][0] = 0
                reward = 1
            self.mujoco_gym.dataStore[agent]["current_target"] = self.mujoco_gym.dataStore["targets"][random.randint(0, len(self.mujoco_gym.dataStore["targets"]) - 1)]["name"]
            self.mujoco_gym.dataStore[agent]["distance"] = self.mujoco_gym.distance(agent, self.mujoco_gym.dataStore[agent]["current_target"])
        current_target_coordinates_with_inventory = np.concatenate((self.mujoco_gym.data.body(self.mujoco_gym.dataStore[agent]["current_target"]).xipos, self.mujoco_gym.dataStore[agent]["inventory"], self.mujoco_gym.getData(agent)["position"]))
        reward = 0
        return reward, current_target_coordinates_with_inventory
    

    
def reward_function(mujoco_gym, agent):
    if "targets" not in mujoco_gym.dataStore[agent].keys():
        mujoco_gym.dataStore["targets"] = mujoco_gym.filterByTag("target")
        mujoco_gym.dataStore[agent]["current_target"] = mujoco_gym.dataStore["targets"][random.randint(0, len(self.mujoco_gym.dataStore["targets"]) - 1)]["name"]
        distance = mujoco_gym.distance(agent, mujoco_gym.dataStore[agent]["current_target"])
        mujoco_gym.dataStore[agent]["distance"] = distance
        new_reward = 0
    else:
        distance = mujoco_gym.distance(agent, mujoco_gym.dataStore[agent]["current_target"])
        new_reward = mujoco_gym.dataStore[agent]["distance"] - distance
        mujoco_gym.dataStore[agent]["distance"] = distance
    reward = new_reward * 10
    return reward

def done_function(mujoco_gym, agent):
    if mujoco_gym.dataStore[agent]["distance"] >= 1:
        return True
    else:
        return False
    


    
class Language():

    def __init__(self, mujoco_gym):
        self.mujoco_gym = mujoco_gym
        self.observation_space = {"low":[0], "high":[3]}
        self.action_space = {"low":[0], "high":[3]}
        self.dataStore = {}
        # Word mapping
        # self.words = {0:"Word 1", 1:"Word 2", 2:"Word 3", 3:"Word 4", 4:"Word 5"}

    def dynamic(self, agent, actions):

        # At timestep 0, the utterance field has to be initialized
        if "utterance" not in self.mujoco_gym.dataStore[agent].keys():
            self.mujoco_gym.dataStore[agent]["utterance"] = 0

        # Extract the utterance from the agents action
        utterance = int(actions[0])
        # print(agent + " said: " + str(utterance))

        # Store the utterance in the dataStore for the environment
        self.mujoco_gym.dataStore[agent]["utterance"] = utterance
        otherAgent = [other for other in self.mujoco_gym.agents if other!=agent][0]

        # Check whether the other agent has "spoken" yet (not at timestep 0)
        if "utterance" in self.mujoco_gym.dataStore[otherAgent]:
            utteranceOtherAgent = self.mujoco_gym.dataStore[otherAgent]["utterance"]
            return 0, np.array([utteranceOtherAgent])
        else:
            return 0, np.array([0])
        


        
def test():
    configDict = {"xmlPath":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/MultiEnvs.xml", "rewardFunctions":[reward_function], "doneFunctions":[done_function], "environmentDynamics":[Pick_Up_Dynamic], "agentCameras":False, "freeJoint":True, "agents":["agent3_torso", "agent2_torso", "agent1_torso"], "infoJson":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/info_example.json", "renderMode":False}
    ray.init(num_gpus=1)
    print(ray.available_resources())
    # config = SACConfig()
    config = PPOConfig()
    # config = config.training(gamma=0.9, lr=tune.grid_search([0.0001, 0.001, 0.01]))
    config = config.training(lr=tune.grid_search([0.001, 0.01]))
    config = config.resources(num_gpus=1) 
    config = config.rollouts(num_rollout_workers=8)
    config = config.framework("tf2")
    # config.replay_batch_size = 128
    config.timesteps_per_iteration = 4096
    config = config.environment(env = MuJoCo_RL, env_config=configDict)

    config["model"]["fcnet_hiddens"] = [1024, 512, 256]
    # config["q_model_config"]["fcnet_hiddens"] = [1024, 512, 256]
    # config["policy_model_config"]["fcnet_hiddens"] = [1024, 512, 256]
    # config["target_network_update_freq"] = 100
    # config.log_level = "INFO"
    config.horizon = 4096

    trainer = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(local_dir="./results", name="test_experiment"),
        # run_config=air.RunConfig(stop={"training_iteration": 10, "mean_accuracy": 0.98}),
        param_space=config.to_dict(),
    )
    trainer.fit()


def fpsTest():
    configDict = {"xmlPath":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/MultiEnvs.xml", "environmentDynamics":[Pick_Up_Dynamic, Language],  "rewardFunctions":[reward_function],  "doneFunctions":[done_function], "agentCameras":False, "freeJoint":True, "agents":["agent3_torso", "agent2_torso", "agent1_torso"], "infoJson":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/info_example.json", "renderMode":True}
    test_env = MuJoCo_RL(configDict=configDict)
    test_env.reset()
    step = 0
    rewards = None
    print(tf.config.list_physical_devices('GPU'))
    while True:
        try:
            start_time = time.time() # start time of the loop
            # print(test_env._action_space)
            action = {"agent3_torso": test_env._action_space["agent3_torso"].sample(), "agent2_torso": test_env._action_space["agent2_torso"].sample(), "agent1_torso": test_env._action_space["agent1_torso"].sample()}
            action["agent3_torso"][1:2] = 1
            action["agent2_torso"][1:2] = 1
            observations, current_rewards, terminations, truncations, infos = test_env.step(action)

            print(test_env.dataStore["agent3_torso"]["current_target"], test_env.dataStore["agent2_torso"]["current_target"], test_env.dataStore["agent1_torso"]["current_target"])
            if rewards == None:
                rewards = current_rewards
            rewards["agent3_torso"] += current_rewards["agent3_torso"]
            rewards["agent2_torso"] += current_rewards["agent2_torso"]
            rewards["agent1_torso"] += current_rewards["agent1_torso"]
            print(rewards)
            time.sleep(0.01)
            # print(action)
            # for agent in test_env.agents:
            #     print(observations[agent].shape, test_env._observation_space[agent].shape)
            # print("FPS: ", 1.0 / (time.time() - start_time))
            # print("\n")
        except KeyboardInterrupt:
            break


test()