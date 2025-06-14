from copy import deepcopy
from random import choice

import numpy as np
from gymnasium import Env, spaces

import setup_path  # NOQA
from GUI.board_to_image import generate_board_image
from environments.init_boards_from_database import initialize_boards, load_specific_board_file
from environments.rewards import basic_reward


class RushHourImageEnv(Env):
    train_boards, test_boards = load_specific_board_file(
        filename="1000_cards_2_cars_1_trucks.json", input_folder="database")

    def __init__(self, num_of_vehicle: int, image_size=(84, 84), train=True, rewards=basic_reward):
        super().__init__()

        self.boards = RushHourImageEnv.train_boards if train else RushHourImageEnv.test_boards
        self.max_steps = 200 if train else 100

        self.image_size = image_size
        self.num_of_vehicle = num_of_vehicle
        self.get_reward = rewards

        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(*image_size, 3), dtype=np.float32
        )

        self.state = None
        self.board = None
        self.vehicles_letter = sorted(
            list({v.letter for b in self.boards for v in b.vehicles}))

        self.num_steps = 0
        self.state_history = []

        print(f"num_vehicles: {self.num_of_vehicle}")

    def reset(self, board=None, seed=None):
        self.board = deepcopy(
            choice(self.boards)) if board is None else deepcopy(board)
        self.vehicles_letter = self.board.get_all_vehicles_letter()
        self.num_steps = 0
        self.state_history = []

        self.state = self._board_to_image(self.board)
        self.state_history.append(tuple(self.state.flatten()))
        return self.state, self._get_info()

    def step(self, action):
        vehicle_str, move_str = self.parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(
            vehicle_str) if vehicle_str else None

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
        self.state_history.append(tuple(current_state.flatten()))
        return self.state, reward, done, truncated, self._get_info()

    def render(self):
        img = generate_board_image(
            self.board, scale=self.image_size[0] // 6, draw_letters=True)
        img.show()

    def _board_to_image(self, board):
        img = generate_board_image(
            board, scale=self.image_size[0] // 6, draw_letters=False)
        img = img.resize(self.image_size)
        img_array = np.asarray(img).astype(np.float32) / 255.0
        return np.clip(img_array, 0, 1)

    def parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        try:
            vehicle_str = self.vehicles_letter[vehicle]
        except IndexError: # if send wrong num_of_vehicle 
            return None, None
        return vehicle_str, move_str

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
