from pathlib import Path
import time
from rush_hour_env import RushHourEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from custom_logger import RushHourCSVLogger  # changed from custom_callback

# === Setup logging ===
log_dir = Path("logs_csv") / "rush_hour"
log_dir.mkdir(parents=True, exist_ok=True)
run_id = f"run_{int(time.time())}_1"
log_file = log_dir / f"{run_id}.csv"

# === Create env and model ===
env = RushHourEnv(num_of_vehicle=4)
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)

# === Create logger callback ===
callback = RushHourCSVLogger(log_path=log_file)

# === Train the model ===
model.learn(total_timesteps=100_000, callback=callback)

# === Save the model ===
model.save("ppo_rush_hour_model")
