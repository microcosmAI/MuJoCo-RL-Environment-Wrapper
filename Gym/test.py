from mujoco_rl import MuJoCo_RL
import cv2
import time

configDict = {"xmlPath":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/MultiEnvs.xml", "agentCameras":True, "freeJoint":True, "agents":["agent3_torso", "agent2_torso", "agent1_torso"], "infoJson":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/info_example.json", "renderMode":False}
test_env = MuJoCo_RL(configDict=configDict)
test_env.reset()

step = 0
while True:
    try:
        start_time = time.time() # start time of the loop
        action = {"agent3_torso": test_env.actionSpace["agent3_torso"].sample(), "agent2_torso": test_env.actionSpace["agent2_torso"].sample(), "agent1_torso": test_env.actionSpace["agent1_torso"].sample()}
        observations, rewards, terminations, truncations, infos = test_env.step(action)
        data = test_env.getCameraData("agent3_torso")
        data = test_env.getCameraData("agent2_torso")
        print("FPS: ", 1.0 / (time.time() - start_time))
    except KeyboardInterrupt:
        break