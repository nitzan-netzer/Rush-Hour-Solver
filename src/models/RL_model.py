import setup_path  # NOQA
import torch
from pathlib import Path

from environments.rush_hour_env import RushHourEnv
from environments.rush_hour_image_env import RushHourImageEnv
from environments.evaluate import evaluate_model
from environments.rewards import basic_reward, per_steps_reward

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from utils.custom_logger import RushHourCSVLogger
from models.early_stopping import EarlyStoppingSuccessRateCallback
from models.cnn_policy import RushHourCNN
from utils.config import MODEL_PATH, LOG_FILE_PATH, NUM_VEHICLES

torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)


class RLModel:
    def __init__(self, model_class, env, model_path, log_file,
                 early_stopping=True, cnn=False):
        self.env = env
        self.model_class = model_class
        self.model_path = model_path
        self.log_file = log_file
        self.early_stopping = early_stopping
        self.cnn = cnn

        if self.cnn and model_class.__name__ == "PPO":
            policy = ActorCriticCnnPolicy
            policy_kwargs = {
                "features_extractor_class": RushHourCNN,
                "features_extractor_kwargs": {"features_dim": 64}
            }
        else:
            policy = "MlpPolicy"
            policy_kwargs = {}

        self.model = model_class(
            policy,
            env=self.env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=1e-4,
            n_steps=256,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5
        )

    def train(self):
        print("📚 Training with memory-safe settings...")
        callbacks = [RushHourCSVLogger(log_path=self.log_file)]

        if self.early_stopping:
            callbacks.append(
                EarlyStoppingSuccessRateCallback(
                    window_size=100, success_threshold=0.9, verbose=1)
            )

        print("🟡 Beginning PPO.learn()")
        torch.cuda.empty_cache()

        self.model.learn(
            total_timesteps=500_000 if self.cnn else 500_000,
            callback=callbacks,
            progress_bar=True
        )
        print("🟢 Training complete")

    def save(self):
        self.model.save(self.model_path)
        print(f"💾 Model saved to {self.model_path}")

    def evaluate(self, test_env, episodes=None):
        print("🚀 Evaluating on test boards...")
        model = self.model_class.load(self.model_path, env=test_env)
        evaluate_model(model, test_env, episodes)
