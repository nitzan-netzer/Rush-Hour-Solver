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
test_episodes = 50
RushHourEnv.evaluate_model(model,test_env,test_episodes)
