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
            "skipFrames": 0,
            "environmentDynamics": [],
            "freeJoint": True,
            "renderMode": False,
            "maxSteps": 1024,
            "agentCameras": True}


# Multi Agent Environment Testing
env = MuJoCoRL(config_dict=get_config('../levels/MultiAgentModel.xml', ['sender', 'receiver']))

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
env = MuJoCoRL(config_dict=get_config('../levels/SingleAgentModel.xml', ['sender']))

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

# Multi Agent Environment with two Sensors per agent - Testing
env = MuJoCoRL(config_dict=get_config('../levels/MultiAgentModel2Sensors.xml', ['sender', 'receiver']))

# List to store the FPS
frames_per_second_multi_2 = []

# Running the environment for 50 episodes
for i in range(51):
    env.reset()
    start = time.time()
    # Running the environment for 1024 steps
    for j in range(1024):
        env.step({"sender": env.action_space('sender').sample(), "receiver": env.action_space('receiver').sample()})
    end = time.time()
    #
    frames_per_second_multi_2.append(1024 / (end - start))
    if i % 10 == 0:
        print("Episode {} done".format(i), "FPS: {}".format(frames_per_second_multi_2[-1]))

# Multi Agent Environment with three Sensors per agent - Testing
env = MuJoCoRL(config_dict=get_config('../levels/MultiAgentModel3Sensors.xml', ['sender', 'receiver']))

# List to store the FPS
frames_per_second_multi_3 = []

# Running the environment for 50 episodes
for i in range(51):
    env.reset()
    start = time.time()
    # Running the environment for 1024 steps
    for j in range(1024):
        env.step({"sender": env.action_space('sender').sample(), "receiver": env.action_space('receiver').sample()})
    end = time.time()
    #
    frames_per_second_multi_3.append(1024 / (end - start))
    if i % 10 == 0:
        print("Episode {} done".format(i), "FPS: {}".format(frames_per_second_multi_3[-1]))

# Plotting the FPS
plt.plot(frames_per_second, label="Single Agent")
plt.plot(frames_per_second_multi, label="Multi Agent")
# plt.plot(frames_per_second_multi_2, label="Multi Agent with 2 Sensors")
# plt.plot(frames_per_second_multi_3, label="Multi Agent with 3 Sensors")
plt.xlabel("Episode")
plt.ylabel("FPS")
plt.title("FPS per Episode")
plt.legend()
plt.show()

# Plotting the FPS of Multi Agent Environments with different number of sensors
# plt.plot(frames_per_second, label="Single Agent")
plt.plot(frames_per_second_multi, label="Multi Agent")
plt.plot(frames_per_second_multi_2, label="Multi Agent with 2 Sensors")
plt.plot(frames_per_second_multi_3, label="Multi Agent with 3 Sensors")
plt.xlabel("Episode")
plt.ylabel("FPS")
plt.title("FPS per Episode")
plt.legend()
plt.show()
