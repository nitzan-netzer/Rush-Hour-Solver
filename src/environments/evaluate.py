
def evaluate_model(model,env, num_of_episodes=None):
    """Evaluate the trained model on test boards."""
    solved, total_steps, total_rewards = 0, 0, 0
    if num_of_episodes is None:
        num_of_episodes = len(env.boards)
    env.max_steps = 50 # TODO: remove this
    for i in range(num_of_episodes):
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

    success_rate, avg_reward, avg_steps = get_results(num_of_episodes,solved, total_steps, total_rewards)
    print_results(num_of_episodes, solved, success_rate, avg_reward, avg_steps)

    return (success_rate, avg_reward, avg_steps)

def get_results(num_of_episodes, solved, total_steps, total_rewards):
    success_rate = solved / num_of_episodes * 100
    avg_reward = total_rewards / num_of_episodes
    if solved > 0:
        avg_steps = total_steps / solved
    else:
        avg_steps = 0
    return success_rate, avg_reward, avg_steps

def print_results(num_of_episodes, solved, success_rate, avg_reward, avg_steps):
    print(f"✅ Solved {solved}/{num_of_episodes}")
    print(f"🏆 Success rate: {success_rate:.2f}%")
    print(f"📈 Avg reward: {avg_reward:.2f}")
    if solved:
        print(f"⏱️ Avg steps to solve: {avg_steps :.2f}")
    else:
        print("⚠️ No puzzles solved in the set.")
