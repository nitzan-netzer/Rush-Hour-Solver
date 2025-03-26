import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
from random import choice
from board import Board

# Load boards from JSON
boards = Board.load_multiple_boards("database/example-1000.json")


class RushHourEnv(Env):
    def __init__(self, num_of_vehicle: int):
        self.num_of_vehicle = num_of_vehicle
        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            # board flattened using ASCII values
            low=0, high=255, shape=(36,), dtype=np.uint8
        )
        self.state = None
        self.board = None

    def reset(self, seed=None, options=None):
        self.board = choice(boards)
        self.state = self.board.get_board_flatten().astype(np.uint8)
        return self.state, self._get_info()

    def step(self, action):
        # encourage shorter solution
        reward = -1

        vehicle_str, move_str = self.parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(vehicle_str)

        # Validate vehicle exists - if not charge with -10 points
        if vehicle is None:
            reward -= 10
            done = False
        else:
            # Validate move - if not charge with -5 points
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
        return {"non_empty_cells": non_empty_cells}

    def parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        vehicle_str = ["X", "A", "B", "O"][vehicle]
        return vehicle_str, move_str
