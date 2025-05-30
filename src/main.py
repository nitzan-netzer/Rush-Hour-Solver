import time
from pathlib import Path
from stable_baselines3 import PPO, DQN, A2C
from models.RL_model import RLModel
from environments.rush_hour_env import RushHourEnv
from environments.rush_hour_image_env import RushHourImageEnv
from environments.rewards import basic_reward
from utils.analyze_logs import analyze_logs
from GUI.visualizer import run_visualizer
from utils.config import MODEL_DIR, LOG_DIR, VIDEO_PATH, NUM_VEHICLES


def load_model_from_path(model_path, env):
    name = model_path.name.lower()
    if "ppo" in name:
        return PPO.load(model_path, env=env)
    elif "dqn" in name:
        return DQN.load(model_path, env=env)
    elif "a2c" in name:
        return A2C.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown model type in filename: {model_path.name}")


def analyze_training_logs(logs_files):
    print("\n📊 Analyzing training logs...")
    analyze_logs(logs_files)
    print("✅ Log analysis completed. Plots displayed.")


def visualize_all_models(models_paths):
    print("\n🎥 Generating and saving visualizations for all models...")
    for i, model_path in enumerate(models_paths):
        video_path = VIDEO_PATH.parent / f"{model_path.stem}_demo.mp4"

        is_cnn = "cnn" in model_path.name.lower()
        env = RushHourImageEnv(NUM_VEHICLES, train=False, rewards=basic_reward, image_size=(128, 128)) \
            if is_cnn else RushHourEnv(NUM_VEHICLES, train=False, rewards=basic_reward)

        print(
            f"🎬 Visualizing model {i + 1}/{len(models_paths)}: {model_path.name}")
        model = load_model_from_path(model_path, env)
        run_visualizer(model, env, record=True, output_video=str(video_path))
        print(f"✅ Video saved at: {video_path}")


def main():
    start_time = time.time()
    logs_files = []
    models_paths = []
    running_times = []

    runs_to_train = [
        # PPO CNN
        (PPO, True, True),   # PPO-CNN + EarlyStopping
        (PPO, True, False),  # PPO-CNN + No EarlyStopping
        # PPO MLP
        (PPO, False, True),  # PPO-MLP + EarlyStopping
        (PPO, False, False),  # PPO-MLP + No EarlyStopping
        # DQN MLP
        (DQN, False, True),  # DQN-MLP + EarlyStopping
        (DQN, False, False),  # DQN-MLP + No EarlyStopping
    ]

    for model_class, cnn, early_stopping in runs_to_train:
        run_id = f"run_{int(time.time())}"
        model_name = f"{model_class.__name__}_{'CNN' if cnn else 'MLP'}_{'early' if early_stopping else 'full'}_{run_id}.zip"
        model_path = MODEL_DIR / model_name
        log_file = LOG_DIR / f"{model_name}.csv"

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Training: {model_class.__name__} | "
              f"{'CNN' if cnn else 'MLP'} | "
              f"{'EarlyStopping' if early_stopping else 'FullTraining'} ===")

        # Create environment (must be done before RLModel to get obs_space)
        env = RushHourImageEnv(NUM_VEHICLES, train=True, rewards=basic_reward, image_size=(128, 128)) \
            if cnn else RushHourEnv(NUM_VEHICLES, train=True, rewards=basic_reward)

        model = RLModel(
            model_class=model_class,
            env=env,
            model_path=model_path,
            log_file=log_file,
            early_stopping=early_stopping,
            cnn=cnn
        )

        model.train()
        model.save()

        # Create matching test env and evaluate
        test_env = RushHourImageEnv(NUM_VEHICLES, train=False, rewards=basic_reward, image_size=(128, 128)) \
            if cnn else RushHourEnv(NUM_VEHICLES, train=False, rewards=basic_reward)
        model.evaluate(test_env)

        models_paths.append(model.model_path)
        logs_files.append(model.log_file)
        end_time = time.time()
        running_times.append(end_time - start_time)

    analyze_training_logs(logs_files)
    visualize_all_models(models_paths)
    print("\n🎉 All steps completed successfully!")
    for i in range(len(running_times)):
        print(f"Running time {i} --- {running_times[i]}")


if __name__ == "__main__":
    main()
