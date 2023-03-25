import random

class Environment_Dynamic():
    def __init__(self, mujoco_gym):
        """
        Initializes Environment dynamic and defines observation space.

        Parameters:
            mujoco_gym (SingleAgent): instance of single agent environment
        """
        self.mujoco_gym = mujoco_gym
        self.observation_space = {"low" : [-70, -70, -70], "high" : [70, 70, 70]}
    
    def dynamic(self):
        """
        Update target if the agent is close enough. 

        Returns: 
            reward (int): reward for the agent, always 0 for environment_dynamics
            target_coordinates (ndarray): coordinates of the target
        """
        if "targets" not in self.mujoco_gym.data_store.keys():
            self.mujoco_gym.data_store["targets"] = self.mujoco_gym.filterByTag("target")
            self.mujoco_gym.data_store["current_target"] = self.mujoco_gym.data_store["targets"][random.randint(0, len(self.mujoco_gym.data_store["targets"]) - 1)]["name"]
        distance = self.mujoco_gym.calculate_distance("torso", self.mujoco_gym.data_store["current_target"])
        if distance < 1:
            self.mujoco_gym.data_store["current_target"] = self.mujoco_gym.data_store["targets"][random.randint(0, len(self.mujoco_gym.data_store["targets"]) - 1)]
            self.mujoco_gym.data_store["distance"] = self.mujoco_gym.calculate_distance("torso", self.mujoco_gym.data_store["current_target"])
        reward = 0
        target_coordinates = self.mujoco_gym.data.body(self.mujoco_gym.data_store["current_target"]).xipos
        return reward, target_coordinates