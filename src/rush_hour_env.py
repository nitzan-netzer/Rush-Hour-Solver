import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from board import Board
from random import choice

# Load boards from JSON
boards = Board.load_multiple_boards("database/example-1000.json")


class RushHourEnv(Env):
    def __init__(self, num_of_vehicle: int):
        self.num_of_vehicle = num_of_vehicle
        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            0, num_of_vehicle, [36,], dtype=np.int16)
        self.state = None

    def reset(self, seed=None, options=None):
        self.board = choice(boards)
        self.state = self.board.board
        return self.state, self._get_info()

    def step(self, action):
        reward = -1
        vehicle_str, move_str = self.parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(vehicle_str)
        is_move = self.board.move_vehicle(vehicle, move_str)
        self.state = self.board.board
        done = self.board.game_over()
        if done:
            reward += 1000
        return self.state, reward, done, False, self._get_info()

    def render(self):
        for row in self.state:
            print(" ".join(cell if cell else "." for cell in row))
        print()

    def _get_info(self):
        non_empty_cells = np.count_nonzero(self.state != "")
        return {"non_empty_cells": non_empty_cells}

    def parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        # Extend this list if needed
        vehicle_str = ["X", "A", "B", "O"][vehicle]
        return vehicle_str, move_str
