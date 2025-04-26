def basic_reward(state_history, current_state, vehicle, valid_move, done, truncated, board, steps, max_steps=5):
    reward -= 1  # Encourage shorter solutions
    if not valid_move:
        reward -= 5
    if done:
        reward += 1000
    if truncated:
        reward -= 100
    return reward


def valid_moves_reward(state_history, current_state, vehicle, valid_move, done, truncated, board, steps, max_steps=5):

    reward -= 1

    possible_moves = vehicle.get_possible_moves(board)

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

    reward -= 1 - (steps / max_steps)

    possible_moves = vehicle.get_possible_moves(board)

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
    if tuple(current_state) in state_history:
        reward -= 5  # Heavy penalty for revisiting states
    else:
        reward += 1  # Reward for exploring new states

    if done:
        reward += 1000 - 3 * steps

    if truncated:
        reward -= 100

    return reward
