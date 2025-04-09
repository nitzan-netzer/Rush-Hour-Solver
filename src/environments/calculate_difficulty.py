"""
This module contains the function to calculate the difficulty of a stage
based on the number of mechanisms
"""
from collections import deque
from copy import deepcopy


def calculate_difficulty(board):
    """
    Calculates the difficulty of the stage based on the number of mechanisms (vehicles)
    blocking the red car.

    Args:
        board (Board): The game board object.

    Returns:
        int: The difficulty score of the stage.
    """
    red_car = None

    # Find the RedCar
    for vehicle in board.vehicles:
        if vehicle.letter == "X":  # Assuming "X" represents the RedCar
            red_car = vehicle
            break

    if not red_car:
        raise ValueError("RedCar not found on the board!")

    # Start calculating the difficulty
    return _calculate_difficulty_recursive(board, red_car)


def _calculate_difficulty_recursive(board, vehicle, visited=None):
    """
    Recursively calculates the difficulty based on vehicles blocking the given vehicle.

    Args:
        board (Board): The game board object.
        vehicle (Vehicle): The vehicle currently being evaluated for blocks.
        visited (set): A set of already-visited vehicles to prevent infinite recursion.

    Returns:
        int: The difficulty score for this vehicle.
    """
    if visited is None:
        visited = set()

    # Add the current vehicle to the visited set to avoid revisiting
    visited.add(vehicle.letter)

    difficulty = 0

    # Determine the cells in front of the vehicle based on its direction
    if vehicle.direction == "RL":  # Right-Left
        for col in range(
            vehicle.col + vehicle.length, board.col
        ):  # Check cells to the right
            if board.board[vehicle.row, col] != "":
                blocking_letter = board.board[vehicle.row, col]
                blocking_vehicle = next(
                    v for v in board.vehicles if v.letter == blocking_letter
                )
                if blocking_vehicle.letter not in visited:
                    difficulty += 1  # Add a point for this blocking vehicle
                    difficulty += _calculate_difficulty_recursive(
                        board, blocking_vehicle, visited
                    )
                break  # Stop after the first blocking vehicle
    elif vehicle.direction == "UD":  # Up-Down
        for row in range(vehicle.row + vehicle.length, board.row):  # Check cells below
            if board.board[row, vehicle.col] != "":
                blocking_letter = board.board[row, vehicle.col]
                blocking_vehicle = next(
                    v for v in board.vehicles if v.letter == blocking_letter
                )
                if blocking_vehicle.letter not in visited:
                    difficulty += 1  # Add a point for this blocking vehicle
                    difficulty += _calculate_difficulty_recursive(
                        board, blocking_vehicle, visited
                    )
                break  # Stop after the first blocking vehicle

    return difficulty


# ---- BFS Shortest Path Length Algorithm ----

def get_board_hash(board):
    """Generate a hashable representation of the board for visited state tracking."""
    return tuple(board.get_board_flatten())


def shortest_solution_path_length(start_board):
    """
    Use BFS to calculate the shortest number of steps needed to solve the board.

    Args:
        start_board (Board): Initial board state.

    Returns:
        int: Number of steps in the shortest solution, or -1 if no solution found.
    """
    visited = set()
    queue = deque([(deepcopy(start_board), 0)])

    while queue:
        board, depth = queue.popleft()

        if board.game_over():
            return depth

        board_hash = get_board_hash(board)
        if board_hash in visited:
            continue
        visited.add(board_hash)

        for letter, move in board.get_all_moves():
            next_board = deepcopy(board)
            vehicle = next_board.get_vehicle_by_letter(letter)
            if next_board.move_vehicle(vehicle, move):
                queue.append((next_board, depth + 1))

    return -1  # No solution found


# ----  heuristics recursive blocking ----

def classify_stage_by_recursive_score(score):
    """
    Converts a recursive blocking score into a complexity label.

    Args:
        score (int): Recursive blocker score.

    Returns:
        str: 'easy', 'medium', or 'hard'
    """
    if score <= 5:
        return "easy"
    elif score <= 10:
        return "medium"
    else:
        return "hard"


def recursive_blocking_score(board):
    """
    Recursively calculates a difficulty score based on vehicles blocking
    the red car and the blockers of those blockers (depth-based blocking).

    Args:
        board (Board): The game board object.

    Returns:
        int: A heuristic difficulty score (higher = harder).
    """
    red_car = board.get_vehicle_by_letter("X")
    if not red_car:
        raise ValueError("Red car (X) not found on the board.")

    return _calculate_difficulty_recursive(board, red_car, visited=set())


# def _count_recursive_blockers(board, vehicle, visited):
#     """
#     Recursive helper that counts the blockers in front of a vehicle.

#     Args:
#         board (Board): Current board.
#         vehicle (Vehicle): Vehicle being evaluated.
#         visited (set): Set of vehicle letters already visited.

#     Returns:
#         int: Recursive blocker count.
#     """
#     if vehicle.letter in visited:
#         return 0
#     visited.add(vehicle.letter)

#     difficulty = 0
#     row, col = vehicle.row, vehicle.col
#     length = vehicle.length

#     if vehicle.direction == "RL":
#         for c in range(col + length, board.col):
#             blocking_letter = board.board[row, c]
#             if blocking_letter != "":
#                 blocker = board.get_vehicle_by_letter(blocking_letter)
#                 if blocker and blocker.letter not in visited:
#                     difficulty += 1
#                     difficulty += _count_recursive_blockers(
#                         board, blocker, visited)
#                 break  # Stop at the first blocker
#     elif vehicle.direction == "UD":
#         for r in range(row + length, board.row):
#             blocking_letter = board.board[r, col]
#             if blocking_letter != "":
#                 blocker = board.get_vehicle_by_letter(blocking_letter)
#                 if blocker and blocker.letter not in visited:
#                     difficulty += 1
#                     difficulty += _count_recursive_blockers(
#                         board, blocker, visited)
#                 break

#     return difficulty
