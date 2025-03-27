from pathlib import Path
import time
from environments.rush_hour_env import RushHourEnv, test_boards  # uses pre-split boards
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from logging.custom_logger import RushHourCSVLogger  # for episode logging

# === Setup logging ===
log_dir = Path("logs/logs") / "rush_hour"
log_dir.mkdir(parents=True, exist_ok=True)
run_id = f"run_{int(time.time())}_1"
log_file = log_dir / f"{run_id}.csv"

# === Create training environment ===
env = RushHourEnv(num_of_vehicle=4)  # uses training boards by default
check_env(env, warn=True)

# === Create PPO model ===
model = PPO("MlpPolicy", env, verbose=1)

# === Train the model (no early stopping) ===
callback = RushHourCSVLogger(log_path=log_file)
model.learn(total_timesteps=100_000, callback=callback)

# === Save the model ===
model_path = "ppo_rush_hour_model"
model.save(model_path)
print(f"💾 Model saved to: {model_path}")

# === Load the model for evaluation ===
print("\n🚀 Evaluating model on unseen test boards...")
test_env = RushHourEnv(num_of_vehicle=4, boards=test_boards)
model = PPO.load(model_path, env=test_env)

# === Evaluate ===
solved = 0
total_rewards = 0
total_steps = 0
test_episodes = 50

for i in range(test_episodes):
    obs, _ = test_env.reset()
    episode_reward = 0
    for step in range(100):  # max steps
        action, _ = model.predict(obs)
        obs, reward, done, _, info = test_env.step(action)
        episode_reward += reward
        if done:
            solved += 1
            total_steps += step + 1
            break
    total_rewards += episode_reward

# === Print results ===
print("\n📊 Test Evaluation Results:")
print(f"✅ Solved {solved}/{test_episodes}")
print(f"🏆 Success rate: {solved / test_episodes * 100:.2f}%")
print(f"📈 Avg reward: {total_rewards / test_episodes:.2f}")
if solved > 0:
    print(f"⏱️ Avg steps to solve: {total_steps / solved:.2f}")
else:
    print("⚠️ No puzzles solved in test set.")
