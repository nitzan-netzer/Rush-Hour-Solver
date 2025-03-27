import setup_path # NOQA

from gymnasium import Env, spaces
import numpy as np
from random import choice
from environments.board import Board
from environments.rewards import basic_reward

from sklearn.model_selection import train_test_split


class RushHourEnv(Env):
    train_boards, test_boards = train_test_split(
        Board.load_multiple_boards("database/example-1000.json"),
        test_size=0.2,
        random_state=42
    )
    def __init__(self, num_of_vehicle: int,rewards=basic_reward,train=True): 
        self.num_of_vehicle = num_of_vehicle
        if train:
            self.boards = RushHourEnv.train_boards
            self.max_steps = 500
        else:
            self.boards = RushHourEnv.test_boards
            self.max_steps = 100

        self.action_space = spaces.Discrete(num_of_vehicle * 4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(36,), dtype=np.uint8
        )
        self.state = None
        self.board = None
        self.get_reward = rewards
       
    def reset(self, seed=None):
        self.board = choice(self.boards)
        self.state = self.board.get_board_flatten().astype(np.uint8)
        self.num_steps = 0
        return self.state, self._get_info()
    

    def step(self, action):
        vehicle_str, move_str = self.parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(vehicle_str)
        valid_move = self.board.move_vehicle(vehicle, move_str)
        done = self.board.game_over()
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        reward = self.get_reward(valid_move,done,truncated)

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
        }

    def parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        vehicle_str = ["X", "A", "B", "O"][vehicle]
        return vehicle_str, move_str
    
    @staticmethod
    def evaluate_model(model,env, episodes=50):
        """Evaluate the trained model on test boards."""
        solved, total_steps, total_rewards = 0, 0, 0

        for i in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            while not done and not truncated:
                action, _ = model.predict(obs)
                obs, reward, done,truncated, _ = env.step(action)
                episode_reward += reward
                if done:
                    solved += 1
                    total_steps += env.num_steps
                    break
            total_rewards += episode_reward

        print("\n📊 Test Evaluation Results:")
        print(f"✅ Solved {solved}/{episodes}")
        print(f"🏆 Success rate: {solved / episodes * 100:.2f}%")
        print(f"📈 Avg reward: {total_rewards / episodes:.2f}")
        if solved:
            print(f"⏱️ Avg steps to solve: {total_steps / solved:.2f}")
        else:
            print("⚠️ No puzzles solved in the test set.")

