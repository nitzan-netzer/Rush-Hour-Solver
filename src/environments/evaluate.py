
def evaluate_model(model,env, episodes=50):
    """Evaluate the trained model on test boards."""
    solved, total_steps, total_rewards = 0, 0, 0

    for i in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, reward, done,truncated, _ = env.step(action)
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

