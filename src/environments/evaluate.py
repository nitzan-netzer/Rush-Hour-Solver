
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
            #action_mask = info.get("action_mask")
            #action, _ = model.predict(obs, action_masks=action_mask)
            action, _ = model.predict(obs)
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

