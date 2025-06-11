import json
from copy import deepcopy
from random import choice

import numpy as np
from gymnasium import Env, spaces

import setup_path  # NOQA
from environments.board import Board
from environments.rewards import basic_reward
from environments.init_boards_from_database import initialize_boards


class RushHourEnv(Env):
    train_boards, test_boards = initialize_boards()

    def __init__(self, num_of_vehicle: int, rewards=basic_reward, train=True):
        super().__init__()
        self.boards = RushHourEnv.train_boards if train else RushHourEnv.test_boards
        self.max_steps = 100 if train else 50
        num_of_vehicle = len(self.boards[0].get_all_vehicles_letter())
        self.num_of_vehicle = num_of_vehicle
        self.get_reward = rewards
        size = self.boards[0].row * self.boards[0].col

        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size,), dtype=np.uint8
        )

        self.state = None
        self.board = None
        self.vehicles_letter = None
        self.num_steps = 0
        self.total_reward = 0
        self.state_history = []

    def reset(self, board=None, seed=None,options=None):
        self.board = deepcopy(choice(self.boards)) if board is None else deepcopy(board)
        self.vehicles_letter = self.board.get_all_vehicles_letter()
        self.num_steps = 0
        self.total_reward = 0
        self.state = self.board.get_board_flatten().astype(np.uint8)
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
        self.total_reward += reward
        self.state = current_state
        return self.state, reward, done, truncated, self._get_info()

    def _get_info(self):
        """
        Ensures that info contains the 'action_mask' required by MaskablePPO.
        """
        action_mask = self.board.get_all_valid_actions()
     
        return {
            "red_car_escaped": self.board.game_over(),
            "action_mask": action_mask,
            "total_reward": self.total_reward

        }

    def render(self):
        print(self.board)

    def parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        vehicle_str = self.vehicles_letter[vehicle]

        return vehicle_str, move_str
    
    def get_current_board(self):
        return deepcopy(self.board)



    def reset_number_of_vehicles(self, num_of_vehicle):
        """
        Reset the environment with a specific number of vehicles.
        """
        board = choice(self.boards)
        count = 0
        while board.num_of_vehicles != num_of_vehicle :
            count += 1
            board = choice(self.boards)
            if count > 100:
                raise ValueError(f"No board found with {num_of_vehicle} vehicles.")
        return self.reset(board)

if __name__ == "__main__":
    env = RushHourEnv(num_of_vehicle=16)
   
    obs, info = env.reset_number_of_vehicles(4)
    env.render()
   
    obs, info = env.reset_number_of_vehicles(5)
    env.render()
   
    obs, info = env.reset_number_of_vehicles(6)
    env.render()

    obs, info = env.reset_number_of_vehicles(7)
    env.render()
   
   