import os
import random
from copy import deepcopy
from random import choice

import numpy as np
from gymnasium import Env, spaces
from sklearn.model_selection import train_test_split

import setup_path  # NOQA
from environments.board import Board
from environments.rewards import basic_reward


def initialize_boards(input_folder="database"):
    boards = []

    for file in os.listdir(input_folder):
        if file.endswith(".json") and not file.startswith("_"):
            json_boards_path = os.path.join(input_folder, file)
            boards.extend(Board.load_multiple_boards(json_boards_path))
    
    return boards

def initialize_boards_from_file(file_path):
    boards = Board.load_multiple_boards(file_path)
    return boards

class RushHourEnv(Env):
    
    def __init__(self,num_of_vehicle:int ,rewards=basic_reward,file_path=None): 
        if file_path is None:
            self.all_boards = initialize_boards()
            self.train_boards, self.test_boards = train_test_split(self.all_boards, test_size=0.2, random_state=42)
        else:
            self.all_boards = initialize_boards_from_file(file_path)
            self._train_boards, self.test_boards = train_test_split(self.all_boards, test_size=0.1, random_state=42)

    
        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(36,), dtype=np.uint8)
        self.get_reward = rewards

    def set_train(self):
        self.boards = self.train_boards
        self.max_steps = 1000 
    
    def set_test(self):
        self.boards = self.test_boards
        self.max_steps = 50

    def update_num_of_boards(self,num_of_boards,random_state=42):
        random.seed(random_state)
        boards = random.sample(self._train_boards,num_of_boards)
        self.train_boards = boards
        #self.train_boards, self.test_boards = train_test_split(boards, test_size=0.2, random_state=42)



    def reset(self,board:Board=None,seed=None):
        if board is None:
            self.board =  deepcopy(choice(self.boards))
        else:
            self.board = deepcopy(board)
        self.state = self.board.get_board_flatten().astype(np.uint8)
        self.num_steps = 0
        self.vehicles_letter = self.board.get_all_vehicles_letter()
        self.total_reward = 0
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

        reward = self.get_reward(valid_move,done,truncated)
        self.total_reward += reward
        self.state = self.board.get_board_flatten().astype(np.uint8)
        return self.state, reward, done,truncated, self._get_info()

    def render(self):
        print(self.board)

    def _get_info(self):
        non_empty_cells = np.count_nonzero(self.board.board != "")
        red_car_escaped = self.board.game_over()  # 🚀 CHANGED: include escape flag
        return {
            "non_empty_cells": non_empty_cells,
            "red_car_escaped": red_car_escaped,  # 🚀 CHANGED
            "total_reward": self.total_reward
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
 

