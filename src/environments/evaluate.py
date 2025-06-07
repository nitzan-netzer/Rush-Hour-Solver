def evaluate_model(model,env, episodes=None):
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

if __name__ == "__main__":
    import setup_path
    from utils.config import NUM_VEHICLES
    from environments.rush_hour_env import RushHourEnv
    from environments.rewards import basic_reward
    from sb3_contrib.ppo_mask import MaskablePPO as PPO
    model_path = r"models_zip\MaskablePPO_MLP_8x8"
    test_env = RushHourEnv(NUM_VEHICLES, train=False, rewards=basic_reward)
    model=  PPO.load(model_path, env=test_env)
    evaluate_model(model, test_env)
