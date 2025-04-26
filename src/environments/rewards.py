def basic_reward(valid_move, done, truncated):
    reward = -1  # Encourage shorter solutions
    if not valid_move:
        reward -= 5
    if done:
        reward += 1000
    if truncated:
        reward -= 100
    return reward


def reward_function_no_repetition(valid_move, done, truncated, last_state=None, current_state=None):
    reward = 0

    if not valid_move:
        reward -= 2
    else:
        reward += 1

    if last_state is not None and current_state is not None and last_state == current_state:
        reward -= 3  # discourage doing nothing

    if done:
        reward += 1000

    if truncated:
        reward -= 100

    return reward


def scaled_reward(valid_move, done, truncated):
    # small per-step penalty
    reward = -0.1
    if not valid_move:
        reward -= 0.5       # discourage invalid moves
    if done:
        reward += 10        # scaled success bonus
    if truncated:
        reward -= 2         # truncated episode penalty
    return reward
