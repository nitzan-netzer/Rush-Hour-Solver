from pathlib import Path
import time
# 🚀 CHANGED: import test_boards from env
from rush_hour_env import RushHourEnv, test_boards
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from custom_logger import RushHourCSVLogger
# 🚀 CHANGED: import early stopping
from early_stopping import EarlyStoppingSuccessRateCallback

# === Setup logging ===
log_dir = Path("logs_csv") / "rush_hour"
log_dir.mkdir(parents=True, exist_ok=True)
run_id = f"run_{int(time.time())}_1"
log_file = log_dir / f"{run_id}.csv"

# === Create training environment ===
env = RushHourEnv(num_of_vehicle=4)  # uses default training boards
check_env(env, warn=True)

# === Create PPO model ===
model = PPO("MlpPolicy", env, verbose=1)

# === Callbacks ===
csv_logger = RushHourCSVLogger(log_path=log_file)
early_stop = EarlyStoppingSuccessRateCallback(
    window_size=100, success_threshold=0.9)

# === Train with early stopping and logging ===
# 🚀 CHANGED: add early stopping
model.learn(total_timesteps=300_000, callback=[csv_logger, early_stop])

# === Save model ===
model_path = "ppo_rush_hour_model_es"
model.save(model_path)
print(f"💾 Model saved to: {model_path}")

# === Load model for test evaluation ===
print("\n🚀 Evaluating on test boards...")
# 🚀 CHANGED: test boards
test_env = RushHourEnv(num_of_vehicle=4, boards=test_boards)
model = PPO.load(model_path, env=test_env)

# === Run evaluation ===
solved = 0
total_steps = 0
total_rewards = 0
test_episodes = 50

for i in range(test_episodes):
    obs, _ = test_env.reset()
    episode_reward = 0
    for step in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _, info = test_env.step(action)
        episode_reward += reward
        if done:
            solved += 1
            total_steps += step + 1
            break
    total_rewards += episode_reward

print("\n📊 Test Evaluation Results:")
print(f"✅ Solved {solved}/{test_episodes}")
print(f"🏆 Success rate: {solved / test_episodes * 100:.2f}%")
print(f"📈 Avg reward: {total_rewards / test_episodes:.2f}")
if solved:
    print(f"⏱️ Avg steps to solve: {total_steps / solved:.2f}")
else:
    print("⚠️ No puzzles solved in test set.")
