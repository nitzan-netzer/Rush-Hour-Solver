"""
Generate cards with random vehicles and random moves.
"""
import os
import random

from board import Board
from board_to_image import car_colors, save_board_to_image, truck_colors
from calculate_difficulty import calculate_difficulty
from vehicles import Car, Truck


def shuffle_colors():
    """
    Shuffle the colors of the cars and trucks.
    """
    random.shuffle(car_colors)
    random.shuffle(truck_colors)


def cards_generator(
    num_of_cards: int,
    num_of_cars: int,
    num_of_trucks: int,
    num_of_step: int,
    threshold: int,
):
    """
    Generate cards with random vehicles and random moves.
    The board is guaranteed to have a red car at in start position (2, 4-5).
    before random moves, it guarantee board is can be solved.
    """
    count = 0
    board = Board()
    while count < num_of_cards:
        board.reset()
        shuffle_colors()

        for i in range(num_of_trucks):
            direction = random.choice(["UD", "RL"])
            truck_color = truck_colors[i]
            board.add_random_vehicle(Truck(direction, truck_color))

        for i in range(num_of_cars):
            direction = random.choice(["UD", "RL"])
            car_color = car_colors[i]
            board.add_random_vehicle(Car(direction, car_color))

        for _ in range(num_of_step):
            board.random_move()

        # Ensure the red car not win the game immediately
        difficulty = calculate_difficulty(board)
        if difficulty > threshold:
            count += 1
            save_board_to_image(board, rf"database\easy\board-{count}.png")


def main():
    """ "
    the main function to generate cards.
    """
    num_of_cards = 10
    mode = "easy"
    if mode == "easy":
        num_of_cars = 2
        num_of_trucks = 1
        num_of_step = 10
        threshold = 1
        os.makedirs("database/easy", exist_ok=True)
    else:
        num_of_cars = 6
        num_of_trucks = 2
        num_of_step = 50
        threshold = 3
        os.makedirs("database/hard", exist_ok=True)

    cards_generator(num_of_cards, num_of_cars, num_of_trucks, num_of_step, threshold)


if __name__ == "__main__":
    main()
