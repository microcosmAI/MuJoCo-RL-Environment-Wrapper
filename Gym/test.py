from mujoco_rl import MuJoCo_RL

configDict = {"xmlPath":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/MultiEnvs.xml", "freeJoint":True, "agents":["agent3_torso", "agent2_torso", "agent1_torso"], "infoJson":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/info_example.json", "renderMode":True}
test_env = MuJoCo_RL(configDict=configDict)
test_env.reset()
print(test_env.actionSpace)
print(test_env.observationSpace)
print(test_env.getSensorData("agent3_torso"))

while True:
    action = {"agent3_torso": test_env.actionSpace["agent3_torso"].sample(), "agent2_torso": test_env.actionSpace["agent2_torso"].sample(), "agent1_torso": test_env.actionSpace["agent1_torso"].sample()}
    observations, rewards, terminations, truncations, infos = test_env.step(action)