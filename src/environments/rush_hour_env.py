from gymnasium import Env, spaces
from copy import deepcopy
from random import choice
import numpy as np
from environments.board import Board
from environments.rewards import basic_reward
from .init_boards_from_database import initialize_boards, load_specific_board_file


class RushHourEnv(Env):
    train_boards, test_boards = load_specific_board_file()

    def __init__(self, num_of_vehicle: int = 6, min_vehicles: int = 4, rewards=basic_reward, train=True):
        super().__init__()

        self.max_vehicles = num_of_vehicle
        self.min_vehicles = min_vehicles
        self.get_reward = rewards

        self.boards = RushHourEnv.train_boards if train else RushHourEnv.test_boards
        self.max_steps = 200 if train else 100

        self.board = None
        self.state = None
        self.num_steps = 0
        self.state_history = []

        self.vehicles_letter = []
        self.action_space = spaces.Discrete(self.max_vehicles * 4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(36,), dtype=np.uint8)
        print(f"num_vehicles: {self.max_vehicles}")

    def reset(self, board=None, seed=None):
        # Select a valid board with enough vehicles
        while True:
            b = deepcopy(choice(self.boards)
                         ) if board is None else deepcopy(board)
            vehicles = b.get_all_vehicles_letter()
            if self.min_vehicles <= len(vehicles) <= self.max_vehicles:
                break

        self.board = b
        self.vehicles_letter = vehicles
        self.num_steps = 0
        self.state = self.board.get_board_flatten().astype(np.uint8)
        self.state_history = [tuple(self.state)]

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

    def parse_action(self, action):
        num_vehicles = len(self.vehicles_letter)
        max_action = num_vehicles * 4
        if action >= max_action:
            # Return dummy no-op if action is invalid (SB3 might sample full space)
            return self.vehicles_letter[0], "U"  # safe fallback

        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        vehicle_str = self.vehicles_letter[vehicle]
        return vehicle_str, move_str

    def render(self):
        print(self.board)

    def _get_info(self):
        return {
            "non_empty_cells": np.count_nonzero(self.board.board != ""),
            "red_car_escaped": self.board.game_over()
        }
