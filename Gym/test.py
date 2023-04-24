from mujoco_rl import MuJoCo_RL
import cv2

configDict = {"xmlPath":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/MultiEnvs.xml", "agentCameras":True, "freeJoint":True, "agents":["agent3_torso", "agent2_torso", "agent1_torso"], "infoJson":"/Users/cowolff/Documents/GitHub/s.mujoco_environment/Environments/multi_agent/info_example.json", "renderMode":True}
test_env = MuJoCo_RL(configDict=configDict)
test_env.reset()

step = 0
while True:
    try:
        action = {"agent3_torso": test_env.actionSpace["agent3_torso"].sample(), "agent2_torso": test_env.actionSpace["agent2_torso"].sample(), "agent1_torso": test_env.actionSpace["agent1_torso"].sample()}
        observations, rewards, terminations, truncations, infos = test_env.step(action)
        data = test_env.getCameraData("agent3_torso")
        step += 1
        print("step {}".format(step))
    except KeyboardInterrupt:
        for image in data:
            cv2.imwrite('color_img.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        break