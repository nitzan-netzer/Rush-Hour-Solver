import os
import time
import shutil
from models.ppo_model import run
from utils.analyze_logs import analyze_logs
from GUI.visualizer import run_visualizer  # ✅ Correct import

# === Config ===
from utils.config import MODEL_DIR,LOG_DIR,VIDEO_DIR,VIDEO_PATH,NUM_VEHICLES,LOG_FILE
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs


# === Step 1: Train and Save Model ===
def train_model(enable_early_stopping=True):
    """Train and save the RL model with logging and early stopping."""
    print("🚀 Training the model...")

    # Generate a unique run ID based on timestamp
    run_id = f"run_{int(time.time())}"
    model_name = f"ppo_rush_hour_{run_id}.zip"
    model_path = MODEL_DIR / model_name
    log_file = LOG_DIR / f"{run_id}.csv"

    # Train and save the model
    if enable_early_stopping:
        train_and_save_model(model_path=str(model_path), log_file=str(log_file))
    else:
        train_and_save_model_without(model_path=str(model_path), log_file=str(log_file))
    # Create/update symlink or fallback copy for the latest log
    latest_log_path = LOG_FILE
    if latest_log_path.exists() or latest_log_path.is_symlink():
        latest_log_path.unlink()  # Remove existing symlink or file

    try:
        os.symlink(log_file, LOG_FILE)  # Try creating symlink
        print(f"🔗 Symlink created: {LOG_FILE} → {log_file}")
    except OSError:
        shutil.copy(log_file, LOG_FILE)  # Fallback for Windows
        print(
            f"🔗 Symlink not created, copied latest log to: {LOG_FILE}")

    print(f"✅ Model saved as {model_path}")
    return model_path


# === Step 2: Analyze Training Logs ===
def analyze_training_logs():
    """Analyze the logs and generate plots."""
    print("\n📊 Analyzing training logs...")
    latest_log_path = Path(LATEST_LOG_FILE)

    if not latest_log_path.exists():
        print(f"❌ Log file not found: {LATEST_LOG_FILE}")
        return

    analyze_logs(str(latest_log_path))  # Passes the correct path dynamically
    print("✅ Log analysis completed. Plots displayed.")


# === Step 3: Visualize and Save Video ===
def visualize_and_save(model_path, video_path):
    """Visualize the trained model and save a video."""
    print("\n🎥 Generating and saving visualization...")

    # Run visualizer with recording enabled
    run_visualizer(model_path,record=True, output_video=str(video_path))
    print(f"✅ Video saved at: {video_path}")

def create_folders():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

def main():
    create_folders()
    # Step 1: Train the model and get the model path
    model_path = run(num_of_vehicle=NUM_VEHICLES,enable_early_stopping=True)
    print("✅ Step 1: Model training completed.")

    # Step 2: Analyze training logs
    analyze_training_logs()
    print("✅ Step 2: Log analysis completed.")

    # Step 3: Visualize and save a demo video
    visualize_and_save(model_path, VIDEO_PATH)
    print("✅ Step 2: Visualization completed.")

    print("\n🎉 All steps completed successfully!")

if __name__ == "__main__":
    main()
   