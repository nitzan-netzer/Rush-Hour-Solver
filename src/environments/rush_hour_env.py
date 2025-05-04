import json
from copy import deepcopy
from random import choice

import numpy as np
from gymnasium import Env, spaces
from sklearn.model_selection import train_test_split

import setup_path  # NOQA
from environments.board import Board
from environments.rewards import basic_reward
import os
from .init_boards_from_database import initialize_boards

class RushHourEnv(Env):
    train_boards, test_boards = initialize_boards()
    
    def __init__(self,num_of_vehicle:int ,rewards=basic_reward,train=True): 
        if train:
            self.boards = RushHourEnv.train_boards
            self.max_steps = 2000 
        else:
            self.boards = RushHourEnv.test_boards
            self.max_steps = 100

        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(36,), dtype=np.uint8
        )
        self.state = None
        self.board = None
        self.vehicles_letter = ["A", "B", "C", "D", "O", "X"] # TODO: make this dynamic
        self.get_reward = rewards
        self.reward = 0

    def reset(self,board=None,seed=None):
        if board is None:
            self.board =  deepcopy(choice(self.boards))
        else:
            self.board = deepcopy(board)
        self.state = self.board.get_board_flatten().astype(np.uint8)
        self.num_steps = 0
        self.reward = 0
        #self.vehicles_letter = self.board.get_all_vehicles_letter()
        return self.state, self._get_info()
    

    def step(self, action):
        vehicle_str, move_str = self.parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(vehicle_str)
        if vehicle is None:
            valid_move = False
            done = False
        else:
            valid_move = self.board.move_vehicle(vehicle, move_str)
            done = self.board.game_over()
        
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        self.reward = self.get_reward(valid_move,done,truncated, self.reward)

        self.state = self.board.get_board_flatten().astype(np.uint8)
        return self.state, self.reward, done,truncated, self._get_info()

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
        vehicle_str = self.vehicles_letter[vehicle]
        return vehicle_str, move_str

if __name__ == "__main__":
    env = RushHourEnv(num_of_vehicle=6)
    env.reset()
    env.render()
 

