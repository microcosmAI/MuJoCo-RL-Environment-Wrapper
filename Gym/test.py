from mujoco_rl import MuJoCo_RL

configDict = {"xmlPath":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/MultiEnvs.xml", "agents":["agent3_torso"], "infoJson":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/info_example.json", "renderMode":True}
test_env = MuJoCo_RL(configDict=configDict)
test_env.reset()
print(test_env.actionSpace)
print(test_env.observationSpace)