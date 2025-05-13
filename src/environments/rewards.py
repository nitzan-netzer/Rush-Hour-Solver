def basic_reward(state_history, current_state, vehicle, valid_move, done, truncated, board, steps, max_steps=5):
    """
    Basic reward: small penalty per step, heavy penalty for invalid moves, reward for solving.
    """
    reward = -1  # Penalize every step
    if not valid_move:
        reward -= 5
    if done:
        reward += 1000
    if truncated:
        reward -= 100
    return reward


def valid_moves_reward(state_history, current_state, vehicle, valid_move, done, truncated, board, steps, max_steps=5):
    """
    Encourages valid moves; penalizes invalid ones more.
    """
    reward = -1  # Base step penalty

    possible_moves = vehicle.get_possible_moves(board) if vehicle else []

    if valid_move not in possible_moves:
        reward -= 10
    else:
        reward += 5

    if done:
        reward += 1000
    if truncated:
        reward -= 100

    return reward


def per_steps_reward(state_history, current_state, vehicle, valid_move, done, truncated, board, steps, max_steps=5):
    """
    Rewards solutions that take fewer steps, and penalizes invalid moves.
    """
    reward = -1 + (1 - steps / max_steps)  # Encourage shorter episodes

    possible_moves = vehicle.get_possible_moves(board) if vehicle else []

    if valid_move not in possible_moves:
        reward -= 10
    else:
        reward += 5

    if done:
        reward += 1000 - 2 * steps
    if truncated:
        reward -= 100

    return reward


def reward_function_no_repetition(state_history, current_state, vehicle, valid_move, done, truncated, board, steps, max_steps=5):
    """
    Penalizes repeating board states and rewards exploring new ones.
    """
    reward = 0

    if tuple(current_state.flatten()) in state_history:
        reward -= 5
    else:
        reward += 1

    if done:
        reward += 1000 - 3 * steps
    if truncated:
        reward -= 100

    return reward
