from mujoco_rl import MuJoCo_RL

DIRECTORY = '/home/lisa/Mount/Dateien/StudyProject'

configDict = {"xmlPath":f"{DIRECTORY}/s.mujoco_environment/Environments/multi_agent/MultiEnvs.xml", 
              "infoJson":f"{DIRECTORY}/s.mujoco_environment/Environments/multi_agent/info_example.json", 
              "renderMode":True}
test_env = MuJoCo_RL(configDict=configDict)
while True:
    test_env.mujocoStep()
    print(test_env.filterByTag("target"))