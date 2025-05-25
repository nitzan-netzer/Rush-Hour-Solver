import gymnasium as gym
import numpy as np
from gymnasium import Env,spaces
from gymnasium.spaces import MultiDiscrete

from board import Board
from random import choice

boards = Board.load_multiple_boards(r"database/example-1000.json")

class RushHourEnv(Env):
    def __init__(self, num_of_vehicle: int):
        # Define action and observation space
        self.action_space = MultiDiscrete(
            [num_of_vehicle, 4]
        )  # Example: 10 possible moves
        self.observation_space = spaces.Box(0, num_of_vehicle, [36,], dtype=np.int16)
        self.state = None  # Initialize state variable
    def reset(self):
        """
        Initialize the board state.
        The red car (X) is always in row 3 and has a length of 2.
        """
        # Example board
        self.board = choice(boards)
        self.state = self.board.board
        return self.state, self._get_info()

    def step(self, action):
        """
        Process an action to update the state.
        For now, this is a placeholder. Implement game logic here.
        """
        reward = -1  # Example reward structure
        vehicle_str, move_str = parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(vehicle_str)
        is_move = self.board.move_vehicle(vehicle, move_str)
        """
        if is_move:
            print("moved")
        else:
            print("not moved")
        """
        self.state = self.board.board
        done = self.board.game_over()
        return self.state, reward, done, info

    def _get_info(self):
        """
        Provide additional information about the board.
        """
        non_empty_cells = np.count_nonzero(self.state != "")
        return {
            "non_empty_cells": non_empty_cells,  # Count of non-empty elements
        }

    def render(self):
        """
        Visualize the current state of the board.
        """
        for row in self.state:
            print(" ".join(cell if cell else "." for cell in row))
        print()


def parse_action(action):
    vehicle, move = action
    match int(move):
        case 0:
            move_str = "U"  # Up
        case 1:
            move_str = "D"  # Down
        case 2:
            move_str = "L"  # Left
        case 3:
            move_str = "R"  # Right

    match int(vehicle):
        case 0:
            vehicle_str = "X"
        case 1:
            vehicle_str = "A"
        case 2:
            vehicle_str = "B"
        case 3:
            vehicle_str = "O"

    return vehicle_str, move_str
gym.register(
    id="RushHourEnv-v0",
    entry_point=RushHourEnv,
    kwargs={"num_of_vehicle": 4},  
)

if __name__ == "_main_":
    env = gym.make("RushHourEnv-v0")
    state, info = env.reset()
    print("Initial State:")
    env.render()
    print("Info after reset:", info)
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward

    print("Total Reward:", total_reward)
    print("Final State:")
    env.render()


# Example usage
if __name__ == "__main__":
    num_of_cars = 3
    num_of_trucks = 1
    num_of_vehicle = num_of_cars + num_of_trucks
    env = RushHourEnv(num_of_vehicle)
    # Reset environment
    state, info = env.reset()
    print("Initial State:")
    env.render()
    print("Info after reset:", info)
    done = False
    total_reward = 0
    while not done:
        # Perform a step (placeholder logic)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        # print("State after step:")
        # env.render()
        # print("Reward:", reward)
        # print("Info after step:", info)
        total_reward += reward
    print("Total Reward:", total_reward)
    print("Final State:")
    env.render()
