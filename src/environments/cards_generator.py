"""
Generate cards with random vehicles and random moves.
"""
import json
import os
import random
from copy import deepcopy
from datetime import datetime

from tqdm import tqdm

import setup_path  # NOQA
from environments.board_random import BoardRandom
from GUI.board_to_image import (car_colors, save_board_to_image,
                                         truck_colors)
from environments.calculate_difficulty import calculate_difficulty
from environments.vehicles import Car, Truck

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
    shuffle: bool = False
):
    """
    Generate cards with random vehicles and random moves.
    The board is guaranteed to have a red car at in start position (2, 4-5).
    before random moves, it guarantee board is can be solved.
    """
    boards: list[BoardRandom] = []
    board = BoardRandom()
    hashset = set()

    with tqdm(total=num_cards, desc="Generating Boards") as pbar:
        while len(boards) < num_cards:
            board.reset()
            if shuffle:
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
                board_hash = board.get_hash()
                if difficulty > threshold and not board_hash in hashset:
                    boards.append(deepcopy(board))
                    pbar.update(1)

    return boards


def save(boards, path, save_images=False):
    """
    Save the board to a JSON file.
    """
    filename = f"{path}.json"
    BoardRandom.save_multiple_boards(boards, filename)
    if save_images:
        os.makedirs(path, exist_ok=True)
        for i, board in enumerate(boards):
            save_board_to_image(board, rf"{
                path}/board-{i}.png")
            if i == 10:
                break


def save_config(num_cards:int ,num_cars:int, num_trucks:int,vehicles_letter:list[str]):
    """
    Save the configuration to a JSON file.
    """
    filename = f"{num_cards}_cards_{num_cars}_cars_{num_trucks}_trucks"
    path = f"database/{filename}"
    config_path = f"database/config_{filename}.json"
    config = {
        "path": path+".json",
        "num_cards": num_cards,
        "num_cars": num_cars,
        "num_trucks": num_trucks,
        "vehicles_letter": vehicles_letter,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    return path
def main():
    num_cards = 1000
    num_trucks = 1
    num_step = 20
    threshold = 1
    for num_cars in range(2, 5):
        boards = cards_generator(
            num_cards, num_cars, num_trucks, num_step, threshold)
        vehicles_letter = boards[0].get_all_vehicles_letter()
        path = save_config(num_cards, num_cars, num_trucks, vehicles_letter)
        save(boards, path, save_images=True)    

if __name__ == "__main__":
    main()
