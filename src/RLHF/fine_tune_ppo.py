import setup_path  # NOQA: Adds parent directory to Python path
from environments.rush_hour_env import RushHourEnv
from RLHF.preference_reward_wrapper import PreferenceRewardWrapper
from stable_baselines3 import PPO
from GUI.visualizer import run_visualizer
from utils.config import VIDEO_DIR, LOG_FILE_PATH
from utils.custom_logger import RushHourCSVLogger
from environments.evaluate import evaluate_model
from pathlib import Path
import time

# Create log directory if it doesn't exist
log_dir = Path(LOG_FILE_PATH).parent
log_dir.mkdir(parents=True, exist_ok=True)

# Generate unique log file name with timestamp
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"PPO_RLHF_finetuned_{timestamp}.csv"

# 1. Load your existing environment
env = RushHourEnv(num_of_vehicle=6, train=True)

# Create test environment for evaluation
test_env = RushHourEnv(num_of_vehicle=6, train=False)

# 2. Wrap it with the preference reward
wrapped_env = PreferenceRewardWrapper(
    env, "models_zip/preference_model_20250527_154637.keras")

# 3. Load your previous best PPO model
model = PPO.load(
    "models_zip/PPO_MLP_full_run_1748012332_rlhf_finetuned.zip", env=wrapped_env)

# Setup custom logger
csv_logger = RushHourCSVLogger(log_path=str(log_file))

# 4. Fine-tune it with the new reward signal
print("🚀 Starting fine-tuning with custom logging...")
model.learn(
    total_timesteps=1_000_000,
    callback=csv_logger,
    progress_bar=True
)

# 5. Save the updated model
model_save_path = f"models_zip/PPO_MLP_full_run_1748012332_rlhf_finetuned_{timestamp}"
model.save(model_save_path)
print(f"💾 Model saved to: {model_save_path}")

# 6. Evaluate the model
print("\n📊 Evaluating model performance...")
evaluate_model(model, test_env, episodes=100)

# 7. Record a demo video
print("\n🎥 Recording demo video...")
run_visualizer(model, env, record=True, output_video=VIDEO_DIR /
               f"PPO_MLP_full_run_1748012332_rlhf_finetuned_{timestamp}_demo.mp4")
