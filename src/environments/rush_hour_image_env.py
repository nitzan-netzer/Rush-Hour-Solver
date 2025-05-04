
from copy import deepcopy
from random import choice

import numpy as np
from gymnasium import Env, spaces

import setup_path  # NOQA
from GUI.board_to_image import generate_board_image
from .init_boards_from_database import initialize_boards




class RushHourImageEnv(Env):
    train_boards, test_boards = initialize_boards()

    def __init__(self, num_of_vehicle: int, image_size=(300, 300), train=True):
        super().__init__()

        if train:
            self.boards = RushHourImageEnv.train_boards
            self.max_steps = 2000
        else:
            self.boards = RushHourImageEnv.test_boards
            self.max_steps = 100

        self.image_size = image_size
        self.num_of_vehicle = num_of_vehicle

        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(*image_size, 3), dtype=np.float32
        )

        self.state = None
        self.board = None
        self.vehicles_letter = ["A", "B", "C",
                                "D", "O", "X"]  # TODO: make dynamic
        self.num_steps = 0
        self.reward = 0

    def reset(self, board=None, seed=None):
        if board is None:
            self.board = deepcopy(choice(self.boards))
        else:
            self.board = deepcopy(board)
        self.num_steps = 0
        self.reward = 0

        self.state = self._board_to_image(self.board)
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
        self.reward = self._compute_reward(valid_move, done, truncated, self.reward)

        self.state = self._board_to_image(self.board)
        return self.state, self.reward, done, truncated, self._get_info()

    def render(self):
        img = generate_board_image(
            self.board, scale=self.image_size[0] // 6, draw_letters=True)
        img.show()

    def _board_to_image(self, board):
        img = generate_board_image(
            board, scale=self.image_size[0] // 6, draw_letters=False)
        img = img.resize(self.image_size)
        img_array = np.asarray(img).astype(np.float32) / \
            255.0  # Normalize to [0,1]
        return img_array

    def parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        vehicle_str = self.vehicles_letter[vehicle]
        return vehicle_str, move_str

    def _compute_reward(self, valid_move, done, truncated, reward):
        reward -= 1
        if not valid_move:
            reward -= 5
        if done:
            reward += 1000
        if truncated:
            reward -= 100
        return reward

    def _get_info(self):
        non_empty_cells = np.count_nonzero(self.board.board != "")
        red_car_escaped = self.board.game_over()
        return {
            "non_empty_cells": non_empty_cells,
            "red_car_escaped": red_car_escaped,
        }


if __name__ == "__main__":
    env = RushHourImageEnv(num_of_vehicle=6, train=True)
    obs, _ = env.reset()
    print("Observation shape:", obs.shape)
    env.render()
