import psutil
import time
import matplotlib.pyplot as plt
from MuJoCo_Gym.mujoco_rl import *
import os

# Get the process ID
pid = os.getpid()
process = psutil.Process(pid)


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


def get_system_usage(process=process):
    cpu = process.cpu_percent(interval=None)
    memory = process.memory_percent()
    return cpu, memory


def benchmark_env(env, episodes=50, steps=1024):
    cpu_usage = []
    memory_usage = []
    for i in range(episodes):
        process.cpu_percent(interval=None)
        start_memory = process.memory_percent()
        env.reset()
        for _ in range(steps):
            if len(env.agents) == 1:
                env.step({"sender": env.action_space('sender').sample()})
            else:
                env.step({"sender": env.action_space('sender').sample(), "receiver": env.action_space('receiver').sample()})

        time.sleep(1)
        cpu, memory = get_system_usage()
        cpu_usage.append(cpu)
        memory_usage.append(memory - start_memory)
        if i % 10 == 0:
            print(f"Episode {i} done, CPU: {cpu}%, Memory: {memory}%")
    return cpu_usage, memory_usage


envi = MuJoCoRL(config_dict=get_config('levels/MultiAgentModel3Sensors.xml', ['sender', 'receiver']))
cpu_multi3, memory_multi3 = benchmark_env(envi)

envi = MuJoCoRL(config_dict=get_config('levels/SingleAgentModel.xml', ['sender']))
cpu_single, memory_single = benchmark_env(envi)

envi = MuJoCoRL(config_dict=get_config('levels/MultiAgentModel2Sensors.xml', ['sender', 'receiver']))
cpu_multi2, memory_multi2 = benchmark_env(envi)

envi = MuJoCoRL(config_dict=get_config('levels/MultiAgentModel.xml', ['sender', 'receiver']))
cpu_multi, memory_multi = benchmark_env(envi)

plt.plot(cpu_multi, label='Multi Agent')
plt.plot(cpu_multi2, label='Multi Agent 2 Sensors')
plt.plot(cpu_multi3, label='Multi Agent 3 Sensors')
plt.xlabel('Episode')
plt.ylabel('CPU Usage (%)')
plt.title('Sensor Comparison CPU Usage per Episode')
plt.legend()
plt.show()

plt.plot(cpu_multi, label='Multi Agent')
plt.plot(cpu_single, label='Single Agent')
plt.xlabel('Episode')
plt.ylabel('CPU Usage (%)')
plt.title('Agent Comparison CPU Usage per Episode')
plt.legend()
plt.show()

plt.plot(memory_multi, label='Multi Agent')
plt.plot(memory_multi2, label='Multi Agent 2 Sensors')
plt.plot(memory_multi3, label='Multi Agent 3 Sensors')
plt.xlabel('Episode')
plt.ylabel('Memory Usage (%)')
plt.title('Sensor Comparison Memory Usage per Episode')
plt.legend()
plt.show()

plt.plot(memory_multi, label='Multi Agent')
plt.plot(memory_single, label='Single Agent')
plt.xlabel('Episode')
plt.ylabel('Memory Usage (%)')
plt.title('Agent Comparison Memory Usage per Episode')
plt.legend()
plt.show()

