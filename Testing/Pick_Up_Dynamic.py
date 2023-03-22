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

    def dynamic(self):
        """
        Update target and add the inventory to the agent as an observation. 

        Returns: 
            reward (int): reward for the agent
            current_target_coordinates_with_inventory (ndarray): concatenation of current_target and inventory
        """
        reward = 0
        if "inventory" not in self.mujoco_gym.data_store.keys():
            self.mujoco_gym.data_store["inventory"] = [0]
        if "targets" not in self.mujoco_gym.data_store.keys():
            self.mujoco_gym.data_store["targets"] = self.mujoco_gym.filterByTag("target")
            self.mujoco_gym.data_store["current_target"] = self.mujoco_gym.data_store["targets"][random.randint(0, len(self.mujoco_gym.data_store["targets"]) - 1)]["name"]
        distance = self.mujoco_gym.calculate_distance("torso", self.mujoco_gym.data_store["current_target"])
        if distance < 2:
            print("target reached")
            if self.mujoco_gym.data_store["inventory"][0] == 0:
                self.mujoco_gym.data_store["inventory"][0] = 1
                reward = 1
            elif self.mujoco_gym.data_store["inventory"][0] == 1:
                self.mujoco_gym.data_store["inventory"][0] = 0
                reward = 1
            self.mujoco_gym.data_store["current_target"] = self.mujoco_gym.data_store["targets"][random.randint(0, len(self.mujoco_gym.data_store["targets"]) - 1)]["name"]
            self.mujoco_gym.data_store["distance"] = self.mujoco_gym.calculate_distance("torso", self.mujoco_gym.data_store["current_target"])
        current_target_coordinates_with_inventory = np.concatenate((self.mujoco_gym.data.body(self.mujoco_gym.data_store["current_target"]).xipos, self.mujoco_gym.data_store["inventory"]))
        return reward, current_target_coordinates_with_inventory

