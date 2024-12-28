"""
Generate cards with random vehicles and random moves.
"""
import os
import random

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
    path: str,
):
    """
    Generate cards with random vehicles and random moves.
    The board is guaranteed to have a red car at in start position (2, 4-5).
    before random moves, it guarantee board is can be solved.
    """
    count = 0
    board = BoardRandom()
    while count < num_cards:
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
                count += 1
                save_board_to_image(board, rf"{path}/board-{count}.png")


def main():
    """ "
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
            os.makedirs("database/easy", exist_ok=True)
            path = "database/easy"
        case "medium":
            num_cars = 4
            num_trucks = 2
            num_step = 25
            threshold = 2
            os.makedirs("database/medium", exist_ok=True)
            path = "database/medium"
        case "hard":
            num_cars = 6
            num_trucks = 3
            num_step = 50
            threshold = 3
            os.makedirs("database/hard", exist_ok=True)
            path = "database/hard"
        case _:
            print("Invalid difficulty")
            return
    cards_generator(num_cards, num_cars, num_trucks, num_step, threshold, path)


if __name__ == "__main__":
    main()
