import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from rush_hour_env import RushHourEnv
from gymnasium.envs.registration import register
import numpy as np

# Register the custom environment
print("Registering custom environment...")
register(
    id='RushHourCustom-v0',
    entry_point=RushHourEnv,
    kwargs={"num_of_vehicle": 4},
)

# Flatten observation wrapper


class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(36,), dtype=np.int16)

    def observation(self, obs):
        return np.array([ord(cell) if cell else 0 for row in obs for cell in row], dtype=np.int16)


# Create and wrap environment
print("Creating environment instance...")
raw_env = gym.make("RushHourCustom-v0")
env = FlattenObsWrapper(raw_env)

# Initialize the model
print("Initializing DQN model...")
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1,
)

# Train the model
print("Starting training...")
model.learn(total_timesteps=50000)
print("Training completed.")

# Save the model
print("Saving the trained model to 'dqn_rush_hour'...")
model.save("dqn_rush_hour")

# Evaluate the model
print("Evaluating the model...")
eval_env = Monitor(FlattenObsWrapper(gym.make("RushHourCustom-v0")))
print("Wrapped eval env")

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

print("Starting test run with rendering:")
obs, _ = eval_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = eval_env.step(action)
    eval_env.render()  # <- comment this line temporarily if needed
print("Test run complete.")
