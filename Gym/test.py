from mujoco_rl import MuJoCo_RL
import cv2
import time
import random
import numpy as np

class Pick_Up_Dynamic():
    def __init__(self, mujoco_gym):
        """
        Initializes the Pick-up dynamic and defines observation space. 

        Parameters:
            mujoco_gym (SingleAgent): instance of single agent environment
        """
        self.mujoco_gym = mujoco_gym
        self.observation_space = {"low":[-70, -70, -70, 0], "high":[70, 70, 70, 1]}

    def dynamic(self, agent):
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
        current_target_coordinates_with_inventory = np.concatenate((self.mujoco_gym.data.body(self.mujoco_gym.dataStore[agent]["current_target"]).xipos, self.mujoco_gym.dataStore[agent]["inventory"]))
        return reward, current_target_coordinates_with_inventory

configDict = {"xmlPath":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/MultiEnvs.xml", "environmentDynamics":[Pick_Up_Dynamic], "agentCameras":False, "freeJoint":True, "agents":["agent3_torso", "agent2_torso", "agent1_torso"], "infoJson":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/info_example.json", "renderMode":False}
test_env = MuJoCo_RL(configDict=configDict)
test_env.reset()
step = 0
while True:
    try:
        start_time = time.time() # start time of the loop
        action = {"agent3_torso": test_env.actionSpace["agent3_torso"].sample(), "agent2_torso": test_env.actionSpace["agent2_torso"].sample(), "agent1_torso": test_env.actionSpace["agent1_torso"].sample()}
        observations, rewards, terminations, truncations, infos = test_env.step(action)
        for agent in test_env.agents:
            print(observations[agent].shape, test_env.observationSpace[agent].shape)
        # data = test_env.getCameraData("agent3_torso")
        # data = test_env.getCameraData("agent2_torso")
        print("FPS: ", 1.0 / (time.time() - start_time))
        print("\n")
    except KeyboardInterrupt:
        break