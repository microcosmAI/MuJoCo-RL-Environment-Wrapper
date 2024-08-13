from MuJoCo_Gym.mujoco_rl import *


def ant_reward_function(env, agent):
    """
    Custom reward function that mimics the reward function of Gym's Ant environment.
    Args:
        env: The environment instance.
        agent: The agent ID.
    Returns:
        float: The reward for the agent.
    """
    xpos_before = env.data_store[agent].get('xpos_before', None)
    xpos_after = env.get_data(agent)['position'][0]

    if xpos_before is None:
        env.data_store[agent]['xpos_before'] = xpos_after
        return 0

    dt = env.model.opt.timestep
    forward_reward = (xpos_after - xpos_before) / dt
    control_cost = 0.5 * np.square(env.data.ctrl).sum()
    contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(env.data.cfrc_ext, -1, 1)))
    reward = forward_reward - control_cost - contact_cost
    env.data_store[agent]['xpos_before'] = xpos_after

    return reward


def get_config(xml_file, agents=None):
    """
    Returns the configuration dictionary for the environment
    :param xml_file: Path to the XML file
    :param agents: List of agents
    :return: Configuration dictionary
    """
    if agents is None:
        agents = ["torso"]
    return {"xmlPath": xml_file,
            "agents": agents,
            "rewardFunctions": [ant_reward_function],
            "doneFunctions": [],
            "skipFrames": 0,
            "environmentDynamics": [],
            "freeJoint": True,
            "renderMode": False,
            "maxSteps": 1024,
            "agentCameras": True}


# Multi Agent Environment Testing
env = MuJoCoRL(config_dict=get_config('../levels/Ant.xml', ['torso']))

# List to store the FPS
frames_per_second_multi = []

# Running the environment for 50 episodes
for i in range(51):
    env.reset()
    start = time.time()
    # Running the environment for 1024 steps
    for j in range(1024):
        env.step({"torso": env.action_space('torso').sample()})
    end = time.time()
    #
    frames_per_second_multi.append(1024 / (end - start))
    if i % 10 == 0:
        print("Episode {} done".format(i), "FPS: {}".format(frames_per_second_multi[-1]))
