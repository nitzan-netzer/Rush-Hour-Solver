import setup_path # NOQA
from pathlib import Path

from environments.rush_hour_env import RushHourEnv
from environments.evaluate import evaluate_model
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from utils.custom_logger import RushHourCSVLogger
from models.early_stopping import EarlyStoppingSuccessRateCallback

from utils.config import MODEL_PATH,LOG_FILE,NUM_VEHICLES

class PPOModel:
    def __init__(self,env, model_path=MODEL_PATH, log_file=LOG_FILE,enable_early_stopping=True):
        self.env = env
        self.model_path = model_path
        self.log_file = log_file
        self.enable_early_stopping = enable_early_stopping
        self.setup_logging()

        # === Create PPO model ===
        print("🧠 Initializing PPO model...")
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def setup_logging(self):
        # === Setup logging ===
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        # === Callbacks ===
        csv_logger = RushHourCSVLogger(log_path=self.log_file)
        if self.enable_early_stopping:
            print("📚 Training the model with early stopping and logging...")

            early_stop = EarlyStoppingSuccessRateCallback(
                window_size=100, success_threshold=0.9)
            self.model.learn(total_timesteps=50_000, callback=[csv_logger, early_stop])

        else:
            print("📚 Training the model without early stopping and logging...")
            self.model.learn(total_timesteps=300_000, callback=[csv_logger])

    def save(self):
        self.model.save(self.model_path)
        print(f"💾 Model saved to: {self.model_path}")
    
    def evaluate(self,test_env,episodes=None):
        # === Load model for test evaluation ===
        print("\n🚀 Evaluating on test boards...")
        model = PPO.load(self.model_path, env=test_env)
        evaluate_model(model,test_env,episodes)

def run(num_of_vehicle,enable_early_stopping=False):
    print("🚀 Creating training environment...")
    train_env = RushHourEnv(num_of_vehicle=num_of_vehicle,train=True)
    check_env(train_env, warn=True)

    test_env = RushHourEnv(num_of_vehicle=num_of_vehicle,train=False)
    check_env(test_env, warn=True)

    ppo_model = PPOModel(train_env,enable_early_stopping=enable_early_stopping)
    ppo_model.train()
    ppo_model.save()
    ppo_model.evaluate(test_env)

    return MODEL_PATH

if __name__ == "__main__":
    run(num_of_vehicle=NUM_VEHICLES,enable_early_stopping=True)
