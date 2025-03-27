def basic_reward(valid_move,done):
    reward = -1  # Encourage shorter solutions
    if not valid_move:
        reward -= 5
    if done:
        reward += 1000
    return reward