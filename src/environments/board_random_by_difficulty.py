import random
from copy import deepcopy
from tqdm import tqdm

from board_random import BoardRandom
from vehicles import Car, Truck
from calculate_difficulty import recursive_blocking_score, classify_stage_by_recursive_score

DIRECTIONS = ["UD", "RL"]
DIFFICULTY_LABELS = ("easy", "medium", "hard")


def generate_boards_by_difficulty(
    num_boards,
    difficulty,
    num_cars=4,
    num_trucks=2,
    num_steps=25,
    max_attempts=500
):
    """
    Generate Rush Hour boards filtered by difficulty using the recursive blocking score.

    Args:
        num_boards (int): Number of desired boards.
        difficulty (str): One of ['easy', 'medium', 'hard'].
        num_cars (int): Number of additional cars.
        num_trucks (int): Number of additional trucks.
        num_steps (int): Number of random moves to mix the board.
        max_attempts (int): How many attempts to try before giving up.

    Returns:
        List[Board]: List of boards matching the target difficulty.
    """
    assert difficulty in DIFFICULTY_LABELS, f"Invalid difficulty: {difficulty}"
    boards = []
    attempts = 0

    with tqdm(total=num_boards, desc=f"Generating {difficulty} boards") as pbar:
        while len(boards) < num_boards and attempts < max_attempts:
            attempts += 1
            board = BoardRandom()
            board.reset()

            # Add random trucks
            for _ in range(num_trucks):
                truck = Truck(random.choice(DIRECTIONS), random.choice("OPQR"))
                board.add_random_vehicle(truck)

            # Add random cars
            for _ in range(num_cars):
                car = Car(random.choice(DIRECTIONS),
                          random.choice("ABCDEFGHJKLMN"))
                board.add_random_vehicle(car)

            # Shuffle with moves
            for _ in range(num_steps):
                board.random_move()

            # Score and check
            score = recursive_blocking_score(board)
            label = classify_stage_by_recursive_score(score)
            if label == difficulty:
                boards.append(deepcopy(board))
                pbar.update(1)

    return boards


if __name__ == "__main__":
    boards = generate_boards_by_difficulty(
        num_boards=5,
        difficulty="medium",
        num_cars=5,
        num_trucks=3,
        num_steps=30
    )

    # Print and score them
    for i, board in enumerate(boards):
        print(f"\n🧩 Board {i+1}")
        print(board)
        print("Score:", recursive_blocking_score(board))
