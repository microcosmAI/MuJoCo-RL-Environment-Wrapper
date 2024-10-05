import gymnasium as gym
import time

env = gym.make('Ant-v2')

for i in range(100):
    env.reset()
    start = time.time()
    for j in range(1024):
        env.step(env.action_space.sample())
    end = time.time()
    print("Episode {} done".format(i), "FPS: {}".format(1024 / (end-start)))