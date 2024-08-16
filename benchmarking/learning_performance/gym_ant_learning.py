import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Initialize the Gym-Ant environment
gym_env = gym.make('Ant-v4')

# Print environment details
print("Environment details:")
print(f"Action space: {gym_env.action_space}")
print(f"Observation space: {gym_env.observation_space}")

# Create the PPO model
gym_model = PPO('MlpPolicy', gym_env, verbose=1)

# Train the agent
print("Training the agent...")
gym_model.learn(total_timesteps=10000)

# Save the model
gym_model.save("ppo_gym_ant")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(gym_model, gym_env, n_eval_episodes=10)
print(f"Gym-Ant Mean reward: {mean_reward} +/- {std_reward}")

# Print final environment state after evaluation
obs, info = gym_env.reset()
print("Initial observation:", obs)
for _ in range(10):  # Run 10 steps
    action, _states = gym_model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = gym_env.step(action)
    print(f"Step: {_}, Observation: {obs}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
    if done or truncated:
        obs, info = gym_env.reset()  # Reset environment if done or truncated
