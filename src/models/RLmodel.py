import setup_path  # NOQA
from pathlib import Path

from environments.rush_hour_env import RushHourEnv
from environments.evaluate import evaluate_model
from environments.rewards import valid_moves_reward, per_steps_reward, reward_function_no_repetition
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from logs_utils.custom_logger import RushHourCSVLogger
from models.early_stopping import EarlyStoppingSuccessRateCallback


def train_and_save_model(model_path="models_zip/ppo_rush_hour_model_es.zip", log_file="logs/rush_hour/run_latest.csv"):
    """Train and save the PPO model with logging and early stopping."""

    # === Setup logging ===
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # === Create training environment ===
    print("🚀 Creating training environment...")
    # Uses default training boards
    env = RushHourEnv(num_of_vehicle=4, rewards=per_steps_reward)
    check_env(env, warn=True)

    # === Create PPO model ===
    print("🧠 Initializing PPO model...")
    model = PPO("MlpPolicy", env, verbose=1)

    # === Callbacks ===
    csv_logger = RushHourCSVLogger(log_path=log_file)
    early_stop = EarlyStoppingSuccessRateCallback(
        window_size=200, success_threshold=0.9, patience=20, verbose=1
    )

    # === Train with early stopping and logging ===
    print("📚 Training the model with early stopping and logging...")
    model.learn(total_timesteps=500_000, callback=[csv_logger, early_stop])

    # === Save model ===
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")

    # === Load model for test evaluation ===
    print("\n🚀 Evaluating on test boards...")
    test_env = RushHourEnv(num_of_vehicle=4, train=False)
    model = PPO.load(model_path, env=test_env)

    # === Run evaluation ===
    evaluate_model(model, test_env, episodes=50)


if __name__ == "__main__":
    train_and_save_model()
