import time
from models.RL_model import RLModel
from stable_baselines3 import PPO, DQN
from environments.rush_hour_env import RushHourEnv
from environments.rush_hour_image_env import RushHourImageEnv
from utils.analyze_logs import analyze_logs
from GUI.visualizer import run_visualizer

from utils.config import MODEL_DIR, LOG_DIR, VIDEO_DIR, VIDEO_PATH, NUM_VEHICLES


def init_model(model_class, early_stopping=True, cnn=False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Unique run ID
    run_id = f"run_{int(time.time())}"
    model_name = f"{model_class.__name__}_{'CNN' if cnn else 'MLP'}_{'early' if early_stopping else 'full'}_{run_id}.zip"
    model_path = MODEL_DIR / model_name
    log_file = LOG_DIR / f"{model_name}.csv"

    # Choose environment
    if cnn:
        train_env = RushHourImageEnv(num_of_vehicle=NUM_VEHICLES, train=True)
    else:
        train_env = RushHourEnv(num_of_vehicle=NUM_VEHICLES, train=True)

    # Create model wrapper
    model = RLModel(
        model_class,
        train_env,
        model_path=model_path,
        log_file=log_file,
        early_stopping=early_stopping,
        cnn=cnn,
    )
    return model


def train_model(model):
    print("🚀 Training the model...")
    model.train()
    model.save()
    return model.model_path


def evaluate_model(model):
    print("🚀 Evaluating the model...")
    if model.cnn:
        test_env = RushHourImageEnv(num_of_vehicle=NUM_VEHICLES, train=False)
    else:
        test_env = RushHourEnv(num_of_vehicle=NUM_VEHICLES, train=False)
    model.evaluate(test_env)


def analyze_training_logs(logs_files):
    print("\n📊 Analyzing training logs...")
    analyze_logs(logs_files)
    print("✅ Log analysis completed. Plots displayed.")


def visualize_and_save(model_path, video_path=VIDEO_PATH):
    print("\n🎥 Generating and saving visualization...")
    run_visualizer(model_path, record=True, output_video=str(video_path))
    print(f"✅ Video saved at: {video_path}")


def main():
    logs_files = []
    models_path = []

    for model_class in [PPO, DQN]:
        cnn = model_class == PPO  # Enable CNN only for PPO

        for early_stopping in [True, False]:
            model = init_model(model_class, early_stopping, cnn)
            train_model(model)
            print("✅ Step 1: Model training completed.")
            evaluate_model(model)
            print("✅ Step 2: Model evaluation completed.")
            logs_files.append(model.log_file)
            models_path.append(model.model_path)

    analyze_training_logs(logs_files)
    print("✅ Step 3: Log analysis completed.")

    visualize_and_save(models_path[0])  # Visualize first model
    print("✅ Step 4: Visualization completed.")
    print("\n🎉 All steps completed successfully!")


if __name__ == "__main__":
    main()
