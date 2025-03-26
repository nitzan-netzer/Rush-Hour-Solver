import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rush_hour_env import RushHourEnv

# STEP 1: Create your custom environment instance
env = RushHourEnv(num_of_vehicle=4)

# STEP 2: Check if the environment follows Gym API correctly
check_env(env, warn=True)

# STEP 3: Initialize the PPO (Proximal Policy Optimization) model
model = PPO(
    "MlpPolicy",     # Neural network policy
    env,             # Custom Rush Hour env
    verbose=1        # Print training output
    # tensorboard_log is removed
)

# STEP 4: Train the model
total_timesteps = 100_000
model.learn(total_timesteps=total_timesteps)

# STEP 5: Save the model
model.save("ppo_rush_hour_model")

# STEP 6: Test the model
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        print("🚗 Great! Red car escaped!")
        break
