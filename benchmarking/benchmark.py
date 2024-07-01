from MuJoCo_Gym.mujoco_rl import *
import matplotlib.pyplot as plt


def get_config(xml_file, agents=None):
    """
    Returns the configuration dictionary for the environment
    :param xml_file: Path to the XML file
    :param agents: List of agents
    :return: Configuration dictionary
    """
    if agents is None:
        agents = ["sender"]
    return {"xmlPath": xml_file,
            "agents": agents,
            "rewardFunctions": [],
            "doneFunctions": [],
            "skipFrames": 5,
            "environmentDynamics": [],
            "freeJoint": True,
            "renderMode": False,
            "maxSteps": 1024,
            "agentCameras": True}




#Mulit Agent Environment Testing
env = MuJoCoRL(config_dict=get_config('levels/MultiAgentModel.xml', ['sender', 'receiver']))

# List to store the FPS
frames_per_second_multi = []

# Running the environment for 50 episodes
for i in range(51):
    env.reset()
    start = time.time()
    # Running the environment for 1024 steps
    for j in range(1024):
        env.step({"sender": env.action_space('sender').sample(), "receiver": env.action_space('receiver').sample()})
    end = time.time()
    #
    frames_per_second_multi.append(1024 / (end - start))
    if i % 10 == 0:
        print("Episode {} done".format(i), "FPS: {}".format(frames_per_second_multi[-1]))

# Single Agent Environment Testing
env = MuJoCoRL(config_dict=get_config('levels/SingleAgentModel.xml', ['sender']))

# List to store the FPS
frames_per_second = []

# Running the environment for 50 episodes
for i in range(51):
    env.reset()
    start = time.time()
    # Running the environment for 1024 steps
    for j in range(1024):
        env.step({"sender": env.action_space('sender').sample()})
    end = time.time()
    # Calculating the FPS
    frames_per_second.append(1024 / (end - start))
    if i % 10 == 0:
        print("Episode {} done".format(i), "FPS: {}".format(frames_per_second[-1]))

# Plotting the FPS
plt.plot(frames_per_second, label="Single Agent")
plt.plot(frames_per_second_multi, label="Multi Agent")
plt.xlabel("Episode")
plt.ylabel("FPS")
plt.title("FPS per Episode")
plt.show()
