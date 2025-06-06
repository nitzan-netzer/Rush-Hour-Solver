from sklearn.model_selection import train_test_split
import setup_path  # NOQA
from environments.board import Board
import json
import os
from pathlib import Path


def initialize_boards(input_folder=None):
    """
    Initialize all board JSONs from the database folder, excluding non-board files.
    """
    if input_folder is None:
        file_path = Path(__file__)
        input_folder = file_path.parent.parent.parent / "database"
        input_folder = str(input_folder)

    boards = []
    for file in os.listdir(input_folder):
        if not file.endswith(".json"):
            continue
        if "trajectory" in file or "feedback" in file or "config" in file:
            continue  # skip non-board data
        json_boards_path = os.path.join(input_folder, file)
        boards.extend(Board.load_multiple_boards(json_boards_path))

    return train_test_split(boards, test_size=0.2, random_state=42)


def load_specific_board_file(filename="1000_cards_4_cars_1_trucks.json", input_folder="database"):
    """
    Load a specific board JSON file and return a train/test split.
    """
    json_path = Path(input_folder) / filename
    if not json_path.exists():
        raise FileNotFoundError(f"Board file not found: {json_path}")

    boards = Board.load_multiple_boards(str(json_path))
    return train_test_split(boards, test_size=0.2, random_state=42)
