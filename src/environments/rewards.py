def basic_reward(valid_move, done, truncated):
    reward = -1  # Encourage shorter solutions
    if not valid_move:
        reward -= 5
    if done:
        reward += 1000
    if truncated:
        reward -= 100
    return reward


def new_reward(vehicle, move_str, done, truncated, board):
    reward = -1

    possible_moves = vehicle.get_possible_moves(board)

    if move_str not in possible_moves:
        reward -= 10
    else:
        reward += 5

    if done:
        reward += 1000

    if truncated:
        reward -= 100

    return reward
