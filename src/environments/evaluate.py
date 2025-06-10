from stable_baselines3 import PPO
import setup_path  # NOQA
from environments.rush_hour_env import RushHourEnv
from environments.rush_hour_image_env import RushHourImageEnv
from utils.config import MODEL_PATH, NUM_VEHICLES
import os


def evaluate_model(model, env, episodes=None):
    """Evaluate the trained model on test boards."""
    solved, total_steps, total_rewards = 0, 0, 0
    if episodes is None:
        episodes = len(env.boards)
    for i in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not done and not truncated:

            action_mask = info.get("action_mask")
            action, _ = model.predict(obs, action_masks=action_mask)
            obs, reward, done,truncated, info = env.step(action)

            episode_reward += reward
            if done:
                solved += 1
                total_steps += env.num_steps
                break
        total_rewards += episode_reward

    print("\n📊 Test Evaluation Results:")
    print(f"✅ Solved {solved}/{episodes}")
    print(f"🏆 Success rate: {solved / episodes * 100:.2f}%")
    print(f"📈 Avg reward: {total_rewards / episodes:.2f}")
    if solved:
        print(f"⏱️ Avg steps to solve: {total_steps / solved:.2f}")
    else:
        print("⚠️ No puzzles solved in the test set.")


def main():
    model_path = "models_zip/PPO_CNN_full_run_1748719838.zip"
    use_image_env = True
    image_size = 128
    episodes = 200

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    print(f"📂 Loading model from: {model_path}")
    model = PPO.load(model_path)

    print("🧩 Initializing test environment...")
    if use_image_env:
        env = RushHourImageEnv(num_of_vehicle=NUM_VEHICLES, image_size=(
            image_size, image_size), train=False)
    else:
        env = RushHourEnv(num_of_vehicle=NUM_VEHICLES, train=False)

    evaluate_model(model, env, episodes=episodes)


if __name__ == "__main__":
    main()
