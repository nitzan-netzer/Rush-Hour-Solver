"""
This module contains the function to calculate the difficulty of a stage
based on the number of mechanisms
"""


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
