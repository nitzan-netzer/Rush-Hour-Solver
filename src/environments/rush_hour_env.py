import os
import json
import numpy as np
from copy import deepcopy
from random import choice

from gymnasium import Env, spaces
from sklearn.model_selection import train_test_split

import setup_path  # NOQA
from environments.board import Board
from environments.rewards import basic_reward
from GUI.board_to_image import generate_board_image


def initialize_boards(input_folder="database"):
    boards = []
    for file in os.listdir(input_folder):
        if file.endswith(".json") and not file.startswith("_"):
            json_boards_path = os.path.join(input_folder, file)
            boards.extend(Board.load_multiple_boards(json_boards_path))
    return train_test_split(boards, test_size=0.2, random_state=42)


class RushHourImageEnv(Env):
    train_boards, test_boards = initialize_boards()

    def __init__(self, num_of_vehicle: int, image_size=(128, 128), train=True, rewards=basic_reward):
        super().__init__()

        self.boards = RushHourImageEnv.train_boards if train else RushHourImageEnv.test_boards
        self.max_steps = 2000 if train else 100

        self.image_size = image_size
        self.num_of_vehicle = num_of_vehicle
        self.get_reward = rewards

        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(*image_size, 3), dtype=np.float32
        )

        self.state = None
        self.board = None
        self.vehicles_letter = []
        self.num_steps = 0
        self.total_reward = 0
        self.state_history = []

    def reset(self, board=None, seed=None):
        self.board = deepcopy(choice(self.boards)) if board is None else deepcopy(board)
        self.vehicles_letter = self.board.get_all_vehicles_letter()
        self.num_steps = 0
        self.total_reward = 0
        self.state_history = []

        self.state = self._board_to_image(self.board)
        self.state_history.append(tuple(self.state.flatten()))
        return self.state, self._get_info()

    def step(self, action):
        vehicle_str, move_str = self.parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(vehicle_str) if vehicle_str else None

        valid_move = False
        done = False

        if vehicle:
            valid_move = self.board.move_vehicle(vehicle, move_str)
            done = self.board.game_over()

        self.num_steps += 1
        truncated = self.num_steps >= self.max_steps

        current_state = self._board_to_image(self.board)
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
        self.state = current_state
        self.total_reward += reward
        self.state_history.append(tuple(current_state.flatten()))
        return self.state, reward, done, truncated, self._get_info()

    def _board_to_image(self, board):
        img = generate_board_image(board, scale=self.image_size[0] // 6, draw_letters=False)
        img = img.resize(self.image_size)
        return np.asarray(img).astype(np.float32) / 255.0

    def parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        try:
            vehicle_str = self.vehicles_letter[vehicle]
        except IndexError:
            return None, None
        return vehicle_str, move_str

    def _get_info(self):
        return {
            "red_car_escaped": self.board.game_over(),
            "action_mask": self.board.get_all_valid_actions(),  # ✅ Action mask support
            "total_reward": self.total_reward
        }

    def render(self):
        img = generate_board_image(self.board, scale=self.image_size[0] // 6, draw_letters=True)
        img.show()


if __name__ == "__main__":
    env = RushHourImageEnv(num_of_vehicle=6, train=True)
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Valid action indices:", np.nonzero(info["action_mask"])[0])
    env.render()
