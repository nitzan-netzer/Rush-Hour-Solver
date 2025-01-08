import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete
import numpy as np

class RushHourEnv(Env):
    def __init__(self):
        # Define action and observation space
        self.action_space = Discrete(10)  # Example: 10 possible moves
        self.state = None  # Initialize state variable

    def reset(self):
        """
        Initialize the board state.
        The red car (X) is always in row 3 and has a length of 2.
        """
        # Example board
        self.state = np.array([
            ["A", "A", "", "", "", ""],
            ["", "", "", "", "D", ""],
            ["", "X", "X", "", "D", ""],  
            ["", "B", "B", "B", "", ""],
            ["", "", "", "", "C", "C"],
            ["", "", "", "", "", ""],
        ], dtype=object)
        
        return self.state, self._get_info()

    def step(self, action):
        """
        Process an action to update the state.
        For now, this is a placeholder. Implement game logic here.
        """
        # Example reward and termination condition
        reward = -1
        done = False  # Update this logic based on game rules
        info = self._get_info()
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

# Example usage
if __name__ == "__main__":
    env = RushHourEnv()

    # Reset environment
    state, info = env.reset()
    print("Initial State:")
    env.render()
    print("Info after reset:", info)

    # Perform a step (placeholder logic)
    state, reward, done, info = env.step(0)
    print("State after step:")
    env.render()
    print("Reward:", reward)
    print("Info after step:", info)
