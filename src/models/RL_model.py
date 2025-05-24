import setup_path # NOQA
from pathlib import Path

from environments.rush_hour_env import RushHourEnv
from environments.evaluate import evaluate_model
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from utils.custom_logger import RushHourCSVLogger
from models.early_stopping import EarlyStoppingSuccessRateCallback

from utils.config import MODEL_PATH,LOG_FILE_PATH,NUM_VEHICLES

class RLModel:
    def __init__(self,model_class,env,model_path,log_file,early_stopping=True):
        self.env = env
        self.model_class = model_class  # PPO, DQN, A2C, etc.
        self.model_name = model_class.__name__
        self.model_path = model_path 
        self.log_file = log_file
        self.early_stopping = early_stopping
        self.setup_logging()

        # === Create PPO model ===
        print(f"🧠 Initializing {self.model_name} model...")
        self.model = model_class("MlpPolicy", self.env, verbose=1)

    def setup_logging(self):
        # === Setup logging ===
        log_dir = Path(self.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        # === Callbacks ===
        csv_logger = RushHourCSVLogger(log_path=self.log_file)
        if self.early_stopping:
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
    
    def evaluate(self,env,episodes=None):
        # === Load model for test evaluation ===
        model = self.model_class.load(self.model_path, env=env)
        result = evaluate_model(model,env,num_of_episodes=episodes)
        return result

def run(num_of_vehicle,model_class,early_stopping=False):
    print("🚀 Creating training environment...")
    train_env = RushHourEnv(num_of_vehicle=num_of_vehicle,train=True)
    check_env(train_env, warn=True)

    test_env = RushHourEnv(num_of_vehicle=num_of_vehicle,train=False)
    check_env(test_env, warn=True)

    model = RLModel(model_class,train_env,model_path=MODEL_PATH,log_file=LOG_FILE_PATH,early_stopping=early_stopping)
    model.train()
    model.save()
    model.evaluate(test_env)

    return MODEL_PATH

if __name__ == "__main__":
    run(num_of_vehicle=NUM_VEHICLES,model_class=PPO,early_stopping=True)
    #run(num_of_vehicle=NUM_VEHICLES,model_class=DQN,early_stopping=True)
    #run(num_of_vehicle=NUM_VEHICLES,model_class=A2C,early_stopping=True)
