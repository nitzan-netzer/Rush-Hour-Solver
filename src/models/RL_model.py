import setup_path  # NOQA
from pathlib import Path

from environments.rush_hour_env import RushHourEnv
from environments.rush_hour_image_env import RushHourImageEnv
from environments.evaluate import evaluate_model
# You can change to another reward here
from environments.rewards import basic_reward

from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib.ppo_mask import MaskablePPO as PPO
from stable_baselines3.common.env_checker import check_env

from utils.custom_logger import RushHourCSVLogger
from models.early_stopping import EarlyStoppingRewardCallback

from models.cnn_policy import RushHourCNN
from stable_baselines3.common.policies import ActorCriticCnnPolicy

from utils.config import MODEL_PATH, LOG_FILE_PATH, NUM_VEHICLES



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

        # Select policy and CNN-specific kwargs if needed
        if self.cnn and self.model_name == "PPO":
            policy = ActorCriticCnnPolicy
            policy_kwargs = dict(
                features_extractor_class=RushHourCNN,
                features_extractor_kwargs=dict(features_dim=128)
            )
        else:
            policy = "MlpPolicy"
            policy_kwargs = None

        self.model = model_class(
            policy,
            self.env,
            verbose=0,
            policy_kwargs=policy_kwargs
        )

    def setup_logging(self):
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        print("📚 Training with memory-safe settings...")
        if self.cnn:
            window_size = 100
        else:
            window_size = 50
        csv_logger = RushHourCSVLogger(log_path=self.log_file)
        callbacks = [csv_logger]

        if self.early_stopping:
            early_stop = EarlyStoppingRewardCallback(
                window_size=window_size,
                reward_threshold=950,
                verbose=1
            )
            callbacks.append(early_stop)
            total_timesteps = 2_000_000  
        else:
            total_timesteps = 2_000_000  
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


def run(num_of_vehicle, model_class, early_stopping=False, cnn=False):
    print("🚀 Creating memory-optimized training environment...")

    if cnn:
        train_env = RushHourImageEnv(
            num_of_vehicle=num_of_vehicle, train=True,
            image_size=(128, 128), rewards=basic_reward
        )
        test_env = RushHourImageEnv(
            num_of_vehicle=num_of_vehicle, train=False,
            image_size=(128, 128), rewards=basic_reward
        )
    else:
        train_env = RushHourEnv(
            num_of_vehicle=num_of_vehicle, train=True,
            rewards=basic_reward
        )
        test_env = RushHourEnv(
            num_of_vehicle=num_of_vehicle, train=False,
            rewards=basic_reward
        )

    check_env(test_env, warn=True)

    model = RLModel(
        model_class=model_class,
        env=train_env,
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
    # Try other setups:
    # run(NUM_VEHICLES, DQN, early_stopping=True, cnn=False)
    # run(NUM_VEHICLES, A2C, early_stopping=True, cnn=False)
