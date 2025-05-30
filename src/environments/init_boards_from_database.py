from sklearn.model_selection import train_test_split
import setup_path  # NOQA
from environments.board import Board
import json
import os
from pathlib import Path


def initialize_boards(input_folder=None):
    if input_folder is None:
        file_path = Path(__file__)
        input_folder = file_path.parent.parent.parent / "database"
        input_folder = str(input_folder)
    boards = []
    for file in os.listdir(input_folder):
        if file.endswith(".json") and not file.startswith("_"):
            json_boards_path = os.path.join(input_folder, file)
            boards.extend(Board.load_multiple_boards(json_boards_path))

    train_boards, test_boards = train_test_split(
        boards, test_size=0.2, random_state=42)
    return train_boards, test_boards