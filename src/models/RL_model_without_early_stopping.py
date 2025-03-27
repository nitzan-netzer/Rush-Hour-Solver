import setup_path # NOQA

from pathlib import Path
import time
from environments.rush_hour_env import RushHourEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from logs_utils.custom_logger import RushHourCSVLogger  # for episode logging

def train_and_save_model_without(model_path="models/ppo_rush_hour_model_es.zip", log_file="logs/rush_hour/run_latest.csv"):
    """Train and save the PPO model with logging and early stopping."""
    # === Setup logging ===
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)


    # === Create training environment ===
    env = RushHourEnv(num_of_vehicle=4)  # uses training boards by default
    check_env(env, warn=True)

    # === Create PPO model ===
    model = PPO("MlpPolicy", env, verbose=1)

    # === Train the model (no early stopping) ===
    callback = RushHourCSVLogger(log_path=log_file)
    model.learn(total_timesteps=30_000, callback=callback)

    # === Save the model ===
    model_path = "models/ppo_rush_hour_model"
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")

    # === Load the model for evaluation ===
    print("\n🚀 Evaluating model on unseen test boards...")
    test_env = RushHourEnv(num_of_vehicle=4, train=False)
    model = PPO.load(model_path, env=test_env)

    # === Evaluate ===
    test_episodes = 50
    RushHourEnv.evaluate_model(model,test_env,test_episodes)

if __name__ == "__main__":
    train_and_save_model_without()