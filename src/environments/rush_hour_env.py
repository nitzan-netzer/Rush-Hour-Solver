import json
from copy import deepcopy
from random import choice

import numpy as np
from gymnasium import Env, spaces
from sklearn.model_selection import train_test_split

import setup_path  # NOQA
from environments.board import Board
from environments.rewards import basic_reward
from .init_boards_from_database import initialize_boards


class RushHourEnv(Env):
    train_boards, test_boards = initialize_boards()

    def __init__(self, num_of_vehicle: int, rewards=basic_reward, train=True):
        super().__init__()
        self.boards = RushHourEnv.train_boards if train else RushHourEnv.test_boards
        self.max_steps = 200 if train else 100

        self.num_of_vehicle = num_of_vehicle
        self.get_reward = rewards

        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(36,), dtype=np.uint8
        )

        self.state = None
        self.board = None
        self.vehicles_letter = self.board.get_all_vehicles_letter()
        self.num_steps = 0
        self.state_history = []

    def reset(self, board=None, seed=None):
        self.board = deepcopy(
            choice(self.boards)) if board is None else deepcopy(board)
        self.num_steps = 0
        self.state = self.board.get_board_flatten().astype(np.uint8)
        self.state_history = [tuple(self.state)]  # For no-repetition reward
        return self.state, self._get_info()

    def step(self, action):
        vehicle_str, move_str = self.parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(vehicle_str)

        valid_move = False
        done = False

        if vehicle:
            valid_move = self.board.move_vehicle(vehicle, move_str)
            done = self.board.game_over()

        self.num_steps += 1
        truncated = self.num_steps >= self.max_steps

        current_state = self.board.get_board_flatten().astype(np.uint8)

        reward = self.get_reward(
            self.state_history,
            current_state,
            vehicle,
            valid_move,
            done,
            truncated,
            self.board,
            self.num_steps,
            max_steps=self.max_steps
        )

        self.state_history.append(tuple(current_state))
        self.state = current_state

        return self.state, reward, done, truncated, self._get_info()

    def render(self):
        print(self.board)

    def _get_info(self):
        non_empty_cells = np.count_nonzero(self.board.board != "")
        red_car_escaped = self.board.game_over()
        return {
            "non_empty_cells": non_empty_cells,
            "red_car_escaped": red_car_escaped,
        }

    def parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        vehicle_str = self.vehicles_letter[vehicle]
        return vehicle_str, move_str


if __name__ == "__main__":
    env = RushHourEnv(num_of_vehicle=6)
    obs, _ = env.reset()
    env.render()
