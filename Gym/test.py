from mujoco_rl import MuJoCo_RL

configDict = {"xmlPath":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/MultiEnvs.xml", "infoJson":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/info_example.json", "renderMode":True}
test_env = MuJoCo_RL(configDict=configDict)
while True:
    test_env.mujocoStep()
    print(test_env.getObservationSpaceMuJoCo("agent3_torso"))
    break