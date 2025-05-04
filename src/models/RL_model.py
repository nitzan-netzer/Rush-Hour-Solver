import setup_path  # NOQA
import multiprocessing
import torch
from pathlib import Path

from environments.rush_hour_env import RushHourEnv
from environments.rush_hour_image_env import RushHourImageEnv
from environments.evaluate import evaluate_model

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from utils.custom_logger import RushHourCSVLogger
from models.early_stopping import EarlyStoppingSuccessRateCallback

from utils.config import MODEL_PATH, LOG_FILE_PATH, NUM_VEHICLES

# Use all CPU cores for PyTorch intra-op threads
N_CPU = multiprocessing.cpu_count()
torch.set_num_threads(N_CPU)


class RLModel:
    def __init__(self, model_class, env, model_path, log_file,
                 early_stopping=True, cnn=False):
        self.env = env
        self.model_class = model_class
        self.model_name = model_class.__name__
        self.model_path = model_path
        self.log_file = log_file
        self.early_stopping = early_stopping
        self.cnn = cnn
        self.setup_logging()

        print(f"🧠 Initializing {self.model_name} model...")

        # Select policy and any CNN-specific kwargs
        if self.cnn and self.model_name == "PPO":
            from models.cnn_policy import RushHourCNN
            from stable_baselines3.common.policies import ActorCriticCnnPolicy
            policy = ActorCriticCnnPolicy
            policy_kwargs = dict(
                features_extractor_class=RushHourCNN,
                features_extractor_kwargs=dict(features_dim=256)
            )
        else:
            policy = "MlpPolicy"
            policy_kwargs = None

        # Instantiate the RL model
        self.model = model_class(
            policy,
            self.env,
            verbose=1,
            policy_kwargs=policy_kwargs
        )

    def setup_logging(self):
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        cpu_count = multiprocessing.cpu_count()       
        print(f"→ multiprocessing.cpu_count() reports {cpu_count} cores")
        csv_logger = RushHourCSVLogger(log_path=self.log_file)
        callbacks = [csv_logger]
        if self.early_stopping:
            print("📚 Training with early stopping and logging...")
            early_stop = EarlyStoppingSuccessRateCallback(
                window_size=100,
                success_threshold=0.9,
                verbose=1
            )
            callbacks.append(early_stop)
            total_timesteps = 50_000
        else:
            print("📚 Training without early stopping (logging only)...")
            total_timesteps = 300_000

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )

    def save(self):
        self.model.save(self.model_path)
        print(f"💾 Model saved to: {self.model_path}")

    def evaluate(self, test_env, episodes=None):
        print("\n🚀 Evaluating on test boards...")
        model = self.model_class.load(self.model_path, env=test_env)
        evaluate_model(model, test_env, episodes)


def run(num_of_vehicle,
        model_class,
        early_stopping=False,
        cnn=False):
    print("🚀 Creating training environment...")

    # Vectorized training environment across all CPU cores
    if cnn:
        train_env = make_vec_env(
            lambda: RushHourImageEnv(num_of_vehicle=num_of_vehicle, train=True),
            n_envs=N_CPU
        )
        test_env = RushHourImageEnv(num_of_vehicle=num_of_vehicle, train=False)
    else:
        train_env = make_vec_env(
            lambda: RushHourEnv(num_of_vehicle=num_of_vehicle, train=True),
            n_envs=N_CPU
        )
        test_env = RushHourEnv(num_of_vehicle=num_of_vehicle, train=False)

    # Validate the single-threaded test environment
    check_env(test_env, warn=True)

    # Initialize, train, save, and evaluate
    model = RLModel(
        model_class,
        train_env,
        model_path=MODEL_PATH,
        log_file=LOG_FILE_PATH,
        early_stopping=early_stopping,
        cnn=cnn
    )
    model.train()
    model.save()
    model.evaluate(test_env)

    return MODEL_PATH


if __name__ == "__main__":
    # Example: PPO with CNN and early stopping
    run(
        num_of_vehicle=NUM_VEHICLES,
        model_class=PPO,
        early_stopping=True,
        cnn=True
    )
    # To try other algorithms:
    # run(NUM_VEHICLES, DQN,      early_stopping=True, cnn=False)
    # run(NUM_VEHICLES, A2C,      early_stopping=True, cnn=False)
