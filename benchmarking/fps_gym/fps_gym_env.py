import gymnasium as gym
import time

# Initialize the Gym-Ant environment
gym_env = gym.make('Ant-v4', xml_file='../levels/Ant.xml')

# List to store the FPS
frames_per_second_gym = []

# Running the environment for 50 episodes
for i in range(51):
    gym_env.reset()
    start = time.time()
    # Running the environment for 1024 steps
    for j in range(1024):
        gym_env.step(gym_env.action_space.sample())
    end = time.time()
    #
    frames_per_second_gym.append(1024 / (end - start))
    if i % 10 == 0:
        print("Episode {} done".format(i), "FPS: {}".format(frames_per_second_gym[-1]))
