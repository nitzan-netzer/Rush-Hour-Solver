"""
Generate cards with random vehicles and random moves.
"""
import os
import random
from copy import deepcopy
from datetime import datetime

from tqdm import tqdm

from board_random import BoardRandom
from board_to_image import car_colors, save_board_to_image, truck_colors
from calculate_difficulty import calculate_difficulty
from vehicles import Car, Truck

DIRECTIONS = ["UD", "RL"]


def shuffle_colors():
    """
    Shuffle the colors of the cars and trucks.
    """
    random.shuffle(car_colors)
    random.shuffle(truck_colors)


def cards_generator(
    num_cards: int,
    num_cars: int,
    num_trucks: int,
    num_step: int,
    threshold: int,
):
    """
    Generate cards with random vehicles and random moves.
    The board is guaranteed to have a red car at in start position (2, 4-5).
    before random moves, it guarantee board is can be solved.
    """
    boards: list[BoardRandom] = []
    board = BoardRandom()
    with tqdm(total=num_cards, desc="Generating Boards") as pbar:
        while len(boards) < num_cards:
            board.reset()
            shuffle_colors()
            failed = False
            for i in range(num_trucks):
                direction = random.choice(DIRECTIONS)
                truck_color = truck_colors[i]
                board.add_random_vehicle(Truck(direction, truck_color))

            for i in range(num_cars):
                direction = random.choice(DIRECTIONS)
                car_color = car_colors[i]
                board.add_random_vehicle(Car(direction, car_color))

            for _ in range(num_step):
                if not board.random_move():
                    failed = True
                    break

            # Ensure the red car not win the game immediately
            if not failed:
                difficulty = calculate_difficulty(board)
                if difficulty > threshold:
                    boards.append(deepcopy(board))
                    pbar.update(1)

    return boards


def save(boards, path, save_images=False):
    """
    Save the board to a JSON file.
    """
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    filename = rf"{path}/cards_{timestamp}.json"
    BoardRandom.save_multiple_boards(boards, filename)
    if save_images:
        for i, board in enumerate(boards):
            save_board_to_image(board, rf"{path}/board-{i}.png")


def main():
    """
    the main function to generate cards.
    """
    num_cards = 10
    difficulty = "easy"
    match difficulty:
        case "easy":
            num_cars = 2
            num_trucks = 1
            num_step = 10
            threshold = 1
            path = "database/easy"
            os.makedirs("database/easy", exist_ok=True)
        case "medium":
            num_cars = 4
            num_trucks = 2
            num_step = 25
            threshold = 2
            path = "database/medium"
        case "hard":
            num_cars = 6
            num_trucks = 3
            num_step = 50
            threshold = 3
            path = "database/hard"
        case _:
            print("Invalid difficulty")
            return
    boards = cards_generator(num_cards, num_cars, num_trucks, num_step, threshold)
    save(boards, path)


if __name__ == "__main__":
    main()
