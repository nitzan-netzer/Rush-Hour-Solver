import time
from pathlib import Path

import setup_path  # NOQA
from environments.rush_hour_env import RushHourEnv
from RLHF.preference_reward_wrapper import PreferenceRewardWrapper
from RLHF.fine_tune_logger import FineTuneLogger  # ✅ NEW LOGGER
from stable_baselines3 import PPO
from GUI.visualizer import run_visualizer
from utils.config import VIDEO_DIR, LOG_FILE_PATH
from environments.evaluate import evaluate_model
from utils.analyze_logs import analyze_logs
from utils.config import NUM_VEHICLES


class RLHFFineTuner:
    def __init__(self, base_model_path, reward_model_path, log_file_path):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_file_path).parent
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / \
            f"PPO_RLHF_finetuned_{self.timestamp}.csv"
        self.model_save_path = f"models_zip/PPO_MLP_full_run_rlhf_finetuned_{self.timestamp}"

        self.env = RushHourEnv(num_of_vehicle=NUM_VEHICLES, train=True)
        self.test_env = RushHourEnv(num_of_vehicle=NUM_VEHICLES, train=False)
        self.wrapped_env = PreferenceRewardWrapper(self.env, reward_model_path)

        self.model = PPO.load(base_model_path, env=self.wrapped_env)
        self.logger = FineTuneLogger(
            log_path=str(self.log_file))  # ✅ NEW LOGGER

    def train(self, timesteps=1_000_000):
        print("🚀 Starting fine-tuning with CSV logging...")
        self.model.learn(
            total_timesteps=timesteps,
            callback=self.logger,
            progress_bar=True
        )
        self.model.save(self.model_save_path)
        print(f"💾 Model saved to: {self.model_save_path}")

    def evaluate(self, episodes=100):
        print("\n📊 Evaluating model performance...")
        evaluate_model(self.model, self.test_env, episodes=episodes)

    def record_video(self):
        print("\n🎥 Recording demo video...")
        run_visualizer(self.model, self.env, record=True,
                       output_video=VIDEO_DIR / f"{self.model_save_path.split('/')[-1]}_demo.mp4")

    def analyze_logs(self):
        print("\n📈 Analyzing training logs...")
        analyze_logs([str(self.log_file)])


if __name__ == "__main__":
    base_model_path = "models_zip/PPO_MLP_full_4vehicles_run_1749058532.zip"
    reward_model_path = "models_zip/preference_model_20250605_124526.keras"
    tuner = RLHFFineTuner(base_model_path, reward_model_path, LOG_FILE_PATH)

    tuner.train(timesteps=1_000_000)
    tuner.evaluate(episodes=200)
    tuner.analyze_logs()
    tuner.record_video()
