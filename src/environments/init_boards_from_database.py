from sklearn.model_selection import train_test_split
import setup_path  # NOQA
from environments.board import Board
import json
import os
from pathlib import Path


def initialize_boards(input_folder="database"):
    boards = []
    for file in os.listdir(input_folder):
        if not file.endswith(".json"):
            continue
        if "trajectory" in file or "feedback" in file or "config" in file:
            continue  # skip non-board data
        json_boards_path = os.path.join(input_folder, file)
        boards.extend(Board.load_multiple_boards(json_boards_path))
    return train_test_split(boards, test_size=0.2, random_state=42)
