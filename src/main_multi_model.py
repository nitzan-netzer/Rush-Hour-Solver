import time

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO

from environments.rush_hour_env import RushHourEnv
from GUI.visualizer import run_visualizer
from models.RL_model import RLModel
from utils.analyze_logs import analyze_logs
# === Config ===
from utils.config import (LOG_DIR, MODEL_DIR, NUM_VEHICLES, VIDEO_DIR,
                          VIDEO_PATH,LOG_DIR_FIGURES)


def create_env(num_of_vehicles:int,file_path:str):
    env = RushHourEnv(num_of_vehicle=num_of_vehicles,file_path=file_path)
    return env

def create_model(env: RushHourEnv,model_class: RLModel,model_name:str,early_stopping:bool=True):  
    model_path = MODEL_DIR / model_name
    log_file = LOG_DIR / f"{model_name}.csv"
    model = RLModel(model_class, env, model_path=model_path,
                    log_file=log_file, early_stopping=early_stopping)

    return model

def train_model(env: RushHourEnv,model: RLModel):
    """Train and save the RL model with logging and early stopping."""
    print("🚀 Training the model...")
    start_time = time.time()
    env.set_train()
    model.train()
    model.save()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"🚀 Training completed in {training_time:.2f} seconds")
    return model.model_path

def evaluate_model(env: RushHourEnv,model: RLModel):
    """Evaluate the trained model."""
    print("🚀 Evaluating the model with train set ...")
    env.set_train()
    episodes = min(len(env.train_boards),len(env.test_boards))
    result_train = model.evaluate(env,episodes=episodes)
    print("\n🚀 Evaluating the model with test set ...")
    env.set_test()
    result_test = model.evaluate(env)

    return result_train, result_test
def analyze_training_logs(logs_files:list[str]):
    """Analyze the logs and generate plots."""
    print("\n📊 Analyzing training logs...")
    analyze_logs(logs_files)
    print("✅ Log analysis completed. Plots displayed.")


def visualize_and_save(test_env: RushHourEnv,model_path:str, video_path:str=VIDEO_PATH):
    """Visualize the trained model and save a video."""
    print("\n🎥 Generating and saving visualization...")
    # Run visualizer with recording enabled
    run_visualizer(test_env,model_path, record=True, output_video=str(video_path))
    print(f"✅ Video saved at: {video_path}")

def get_model_name(model_class,file_path,num_of_boards):
    return f"{model_class.__name__}_{file_path[9:-5]}_{num_of_boards}"

def show_results(results_train, results_test):
    # Get all model names
    model_names = list(results_train.keys())
    x = np.arange(len(model_names))
    width = 0.35  # Width of bars

    # Prepare data for plotting
    train_success = [results_train[model][0] for model in model_names]
    test_success = [results_test[model][0] for model in model_names]
    train_rewards = [results_train[model][1] for model in model_names]
    test_rewards = [results_test[model][1] for model in model_names]
    train_steps = [results_train[model][2] for model in model_names]
    test_steps = [results_test[model][2] for model in model_names]

    # Create timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Plot success rates
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_success, width, label='Train Success Rate')
    plt.bar(x + width/2, test_success, width, label='Test Success Rate')
    plt.title('Success Rate Comparison')
    plt.xlabel('Models')
    plt.ylabel('Success Rate (%)')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOG_DIR_FIGURES / f'success_rate_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_rewards, width, label='Average Train Rewards')
    plt.bar(x + width/2, test_rewards, width, label='Average Test Rewards')
    plt.title('Average Rewards Comparison')
    plt.xlabel('Models')
    plt.ylabel('Average Reward')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOG_DIR_FIGURES / f'rewards_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot steps
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_steps, width, label='Average Train Steps')
    plt.bar(x + width/2, test_steps, width, label='Average Test Steps')
    plt.title('Average Steps Comparison')
    plt.xlabel('Models')
    plt.ylabel('Average Steps')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOG_DIR_FIGURES / f'steps_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Figures saved in {LOG_DIR_FIGURES} with timestamp {timestamp}")

def main(num_of_vehicles,file_path,num_of_boards,models):
   
    early_stopping = True
    env = create_env(num_of_vehicles,file_path)
    results_train = {}
    results_test = {}
    logs_files = []
    for model_class in models:
       for num_of_board in num_of_boards:
            model_name = get_model_name(model_class,file_path,num_of_board)
            print(f"\n-----{model_name}-----")
            env.update_num_of_boards(num_of_board)
            model = create_model(env,model_class,model_name,early_stopping)
            train_model(env,model)
            print("✅ Step 1: Model training completed.")
            result_train, result_test = evaluate_model(env,model)
            results_train[model_name] = result_train
            results_test[model_name] = result_test
            print("✅ Step 2: Model evaluation completed.")
            logs_files.append(model.log_file)
    show_results(results_train,results_test)
    """
    # Step 3: Analyze training logs
    analyze_training_logs(logs_files)
    print("✅ Step 3: Log analysis completed.")
    visualize_and_save(env,models_path[-1])
    print("\n🎉 All steps completed successfully!")
    """

def create_directories():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR_FIGURES.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    create_directories()
    num_of_vehicles = 6
    file_path = "database/10000_cards_4_cars_1_trucks.json"
    num_of_boards = (1,10,100,500,1000,5000,9000)
    #num_of_boards = (100,250,500)
    #num_of_boards = (1,)
    models= [PPO,]
    main(num_of_vehicles,file_path,num_of_boards,models)

