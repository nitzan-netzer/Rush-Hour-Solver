import os
import time
import shutil
from pathlib import Path
from models.RLmodel import train_and_save_model
from models.RL_model_without_early_stopping import train_and_save_model_without
from environments.rush_hour_env import RushHourEnv
from environments.evaluate import evaluate_model
from logs_utils.analyze_logs import analyze_logs
from GUI.visualizer import run_visualizer  # ✅ Correct import
from stable_baselines3 import PPO, DQN


# === Config ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
MODEL_DIR = Path("models_zip/")  # Directory to store models
LOG_DIR = Path("logs/rush_hour/")  # Directory to store logs
VIDEO_DIR = Path("videos/")  # Directory to store videos
VIDEO_PATH = VIDEO_DIR / "rush_hour_demo.mp4"  # Path to save the video
# Symlink or copy for the latest log
LATEST_LOG_FILE = "logs/rush_hour/run_latest.csv"
NUM_VEHICLES = 4


# === Step 1: Train and Save Model ===
def train_model(enable_early_stopping=True):
    """Train and save the RL model with logging and early stopping."""
    print("🚀 Training the model...")

    # Generate a unique run ID based on timestamp
    run_id = f"run_{int(time.time())}"
    model_name = f"dqn_rush_hour_with_early_stopping_{run_id}.zip"
    model_path = MODEL_DIR / model_name
    log_file = LOG_DIR / f"{run_id}.csv"

    # Train and save the model
    if enable_early_stopping:
        train_and_save_model(model_path=str(
            model_path), log_file=str(log_file))
    else:
        train_and_save_model_without(model_path=str(
            model_path), log_file=str(log_file))
    # Create/update symlink or fallback copy for the latest log
    latest_log_path = Path(LATEST_LOG_FILE)
    if latest_log_path.exists() or latest_log_path.is_symlink():
        latest_log_path.unlink()  # Remove existing symlink or file

    try:
        os.symlink(log_file, LATEST_LOG_FILE)  # Try creating symlink
        print(f"🔗 Symlink created: {LATEST_LOG_FILE} → {log_file}")
    except OSError:
        shutil.copy(log_file, LATEST_LOG_FILE)  # Fallback for Windows
        print(
            f"🔗 Symlink not created, copied latest log to: {LATEST_LOG_FILE}")

    print(f"✅ Model saved as {model_path}")
    return model_path


# === Step 2: Evaluate the Model ===
def run_model_evaluation(model_path):
    """Evaluate the trained model on test boards."""
    print("\n📊 Evaluating model on test boards...")
    test_env = RushHourEnv(num_of_vehicle=NUM_VEHICLES, train=True)
    model = DQN.load(str(model_path), env=test_env)

    # Use the evaluate_model function from RLmodel
    evaluate_model(model, test_env)
    print("✅ Model evaluation completed.")


# === Step 3: Analyze Training Logs ===
def analyze_training_logs():
    """Analyze the logs and generate plots."""
    print("\n📊 Analyzing training logs...")
    latest_log_path = Path(LATEST_LOG_FILE)

    if not latest_log_path.exists():
        print(f"❌ Log file not found: {LATEST_LOG_FILE}")
        return

    analyze_logs(str(latest_log_path))  # Passes the correct path dynamically
    print("✅ Log analysis completed. Plots displayed.")


# === Step 4: Visualize and Save Video ===
def visualize_and_save(model_path, video_path):
    """Visualize the trained model and save a video."""
    print("\n🎥 Generating and saving visualization...")

    # Run visualizer with recording enabled
    run_visualizer(model_path, record=True, output_video=str(video_path))
    print(f"✅ Video saved at: {video_path}")


# === Main Execution Flow ===
if __name__ == "__main__":
    # Create necessary directories if they don't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Train the model and get the model path
    model_path = train_model(enable_early_stopping=True)
    print("✅ Step 1: Model training completed.")

    # Step 2: Evaluate the model
    run_model_evaluation(model_path)
    print("✅ Step 2: Model evaluation completed.")

    # Step 3: Analyze training logs
    analyze_training_logs()
    print("✅ Step 3: Log analysis completed.")

    # Step 4: Visualize and save a demo video
    visualize_and_save(model_path, VIDEO_PATH)
    print("✅ Step 4: Visualization completed.")

    print("\n🎉 All steps completed successfully!")
