from pathlib import Path
import time
from environments.rush_hour_env import RushHourEnv, test_boards
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from logs_utils.custom_logger import RushHourCSVLogger
from models.early_stopping import EarlyStoppingSuccessRateCallback


def train_and_save_model(model_path="models/ppo_rush_hour_model_es.zip", log_file="logs_csv/rush_hour/run_latest.csv"):
    """Train and save the PPO model with logging and early stopping."""

    # === Setup logging ===
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # === Create training environment ===
    print("🚀 Creating training environment...")
    env = RushHourEnv(num_of_vehicle=4)  # Uses default training boards
    check_env(env, warn=True)

    # === Create PPO model ===
    print("🧠 Initializing PPO model...")
    model = PPO("MlpPolicy", env, verbose=1)

    # === Callbacks ===
    csv_logger = RushHourCSVLogger(log_path=log_file)
    early_stop = EarlyStoppingSuccessRateCallback(
        window_size=100, success_threshold=0.9
    )

    # === Train with early stopping and logging ===
    print("📚 Training the model with early stopping and logging...")
    model.learn(total_timesteps=300_000, callback=[csv_logger, early_stop])

    # === Save model ===
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")

    # === Load model for test evaluation ===
    print("\n🚀 Evaluating on test boards...")
    test_env = RushHourEnv(num_of_vehicle=4, boards=test_boards)
    model = PPO.load(model_path, env=test_env)

    # === Run evaluation ===
    evaluate_model(model, test_env, test_episodes=50)


def evaluate_model(model, env, test_episodes=50):
    """Evaluate the trained model on test boards."""
    solved, total_steps, total_rewards = 0, 0, 0

    for i in range(test_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(100):
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            if done:
                solved += 1
                total_steps += step + 1
                break
        total_rewards += episode_reward

    print("\n📊 Test Evaluation Results:")
    print(f"✅ Solved {solved}/{test_episodes}")
    print(f"🏆 Success rate: {solved / test_episodes * 100:.2f}%")
    print(f"📈 Avg reward: {total_rewards / test_episodes:.2f}")
    if solved:
        print(f"⏱️ Avg steps to solve: {total_steps / solved:.2f}")
    else:
        print("⚠️ No puzzles solved in the test set.")
