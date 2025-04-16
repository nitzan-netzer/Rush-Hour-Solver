import time
from models.RL_model import RLModel
from stable_baselines3 import PPO, DQN
from environments.rush_hour_env import RushHourEnv
from utils.analyze_logs import analyze_logs
from GUI.visualizer import run_visualizer  

# === Config ===
from utils.config import MODEL_DIR,LOG_DIR,VIDEO_DIR,VIDEO_PATH,NUM_VEHICLES

# === Initialize Model ===
def init_model(model_class,early_stopping=True):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Generate a unique run ID based on timestamp
    run_id = f"run_{int(time.time())}"
    model_name = f"{model_class.__name__}_{run_id}.zip"
    model_path = MODEL_DIR / model_name
    log_file = LOG_DIR / f"{model_name}.csv"


    train_env = RushHourEnv(num_of_vehicle=NUM_VEHICLES,train=True)
    model = RLModel(model_class,train_env,model_path=model_path,log_file=log_file,early_stopping=early_stopping)

    return model
# === Step 1: Train and Save Model ===
def train_model(model):
    """Train and save the RL model with logging and early stopping."""
    print("🚀 Training the model...")
    model.train()
    model.save()

    return model.model_path

# === Step 2: Evaluate Model ===
def evaluate_model(model):
    """Evaluate the trained model."""
    print("🚀 Evaluating the model...")
    test_env = RushHourEnv(num_of_vehicle=NUM_VEHICLES,train=False)
    model.evaluate(test_env)

# === Step 3: Analyze Training Logs ===
def analyze_training_logs(logs_files):
    """Analyze the logs and generate plots."""
    print("\n📊 Analyzing training logs...")
    analyze_logs(logs_files)
    print("✅ Log analysis completed. Plots displayed.")


# === Step 4: Visualize and Save Video ===
def visualize_and_save(model_path,video_path=VIDEO_PATH):
    """Visualize the trained model and save a video."""
    print("\n🎥 Generating and saving visualization...")
    # Run visualizer with recording enabled
    run_visualizer(model_path,record=True, output_video=str(video_path))
    print(f"✅ Video saved at: {video_path}")


def main():
    logs_files = []
    models_path = []

    for model_class in [PPO, DQN]:
        # === Initialize Model ===
        model = init_model(model_class,early_stopping=True)
        # Step 1: Train the model and get the model path
        train_model(model)
        print("✅ Step 1: Model training completed.")

        # Step 2: Evaluate the model
        evaluate_model(model)
        print("✅ Step 2: Model evaluation completed.")
        logs_files.append(model.log_file)
        models_path.append(model.model_path)
    
    # Step 3: Analyze training logs
    analyze_training_logs(logs_files)
    print("✅ Step 3: Log analysis completed.")

    # Step 4: Visualize and save a demo video
    visualize_and_save(models_path[0]) # TODO: change to the best model
    print("✅ Step 4: Visualization completed.")

    print("\n🎉 All steps completed successfully!")

if __name__ == "__main__":
    main()
   