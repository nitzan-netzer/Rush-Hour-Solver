import setup_path  # NOQA: Adds parent directory to Python path
from environments.rush_hour_env import RushHourEnv
from RLHF.preference_reward_wrapper import PreferenceRewardWrapper
from stable_baselines3 import PPO
from GUI.visualizer import run_visualizer
from utils.config import VIDEO_DIR

# 1. Load your existing environment
env = RushHourEnv(num_of_vehicle=6, train=True)

# 2. Wrap it with the preference reward
wrapped_env = PreferenceRewardWrapper(env, "preference_model.keras")

# 3. Load your previous best PPO model
model = PPO.load("models_zip/PPO_MLP_full_run_1748012332.zip", env=wrapped_env)

# 4. Fine-tune it with the new reward signal
model.learn(total_timesteps=1_000_000)

# 5. Save the updated model
model.save("models_zip/PPO_MLP_full_run_1748012332_rlhf_finetuned")

run_visualizer(model, env, record=True, output_video=VIDEO_DIR /
               "PPO_MLP_full_run_1748012332_rlhf_finetuned_demo.mp4")
