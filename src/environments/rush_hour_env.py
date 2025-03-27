from gymnasium import Env, spaces
import numpy as np
from random import choice
from environments.board import Board

# 🚀 CHANGED: Add train/test split
from sklearn.model_selection import train_test_split

# 🚀 CHANGED: Load and split boards once globally
all_boards = Board.load_multiple_boards("database/example-1000.json")
train_boards, test_boards = train_test_split(
    all_boards, test_size=0.2, random_state=42)


class RushHourEnv(Env):
    def __init__(self, num_of_vehicle: int, boards=None):  # 🚀 CHANGED: add boards argument
        self.num_of_vehicle = num_of_vehicle
        # 🚀 CHANGED: use custom or default to train_boards
        self.boards = boards or train_boards
        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(36,), dtype=np.uint8
        )
        self.state = None
        self.board = None

    def reset(self, seed=None, options=None):
        self.board = choice(self.boards)  # 🚀 CHANGED: use selected board pool
        self.state = self.board.get_board_flatten().astype(np.uint8)
        return self.state, self._get_info()

    def step(self, action):
        reward = -1  # Encourage shorter solutions

        vehicle_str, move_str = self.parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(vehicle_str)

        if vehicle is None:
            reward -= 10
            done = False
        else:
            valid_move = self.board.move_vehicle(vehicle, move_str)
            if not valid_move:
                reward -= 5

            done = self.board.game_over()
            if done:
                reward += 1000

        self.state = self.board.get_board_flatten().astype(np.uint8)
        return self.state, reward, done, False, self._get_info()

    def render(self):
        print(self.board)

    def _get_info(self):
        non_empty_cells = np.count_nonzero(self.board.board != "")
        red_car_escaped = self.board.game_over()  # 🚀 CHANGED: include escape flag
        return {
            "non_empty_cells": non_empty_cells,
            "red_car_escaped": red_car_escaped,  # 🚀 CHANGED
        }

    def parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        vehicle_str = ["X", "A", "B", "O"][vehicle]
        return vehicle_str, move_str
