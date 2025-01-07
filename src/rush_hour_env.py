import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np

class RushHourEnv(Env):
    def __init__(self):
        # Define action and observation space
        self.action_space = Discrete(10)  # Example: 10 possible moves
        self.observation_space = Box(low=0, high=1, shape=(10,), dtype=np.float32)  # Example state space
        
        self.state = None  # Initialize state variable

    def reset(self):
        # Initialize the state
        self.state = np.zeros(10)  # Set the initial state
        return self.state, self._get_info()  # Return state and info dictionary

    def step(self, action):
        # Update state based on action
        reward = -1  # Example reward structure
        done = False  # Example: check if game is over
        info = self._get_info()  # Fetch additional information
        return self.state, reward, done, info

    def _get_info(self):
        # Return additional info about the environment
        return {
            "state_sum": np.sum(self.state),  # Example: sum of state values
            "non_zero_count": np.count_nonzero(self.state),  # Example: count of non-zero elements
        }

    def render(self):
        # Visualize the environment (optional)
        print(self.state)

# Registering the environment
gym.register(
    id="RushHour-v0",
    entry_point="RushHourEnv", 
)

# Example usage
if __name__ == "__main__":
    env = RushHourEnv()

    # Reset environment
    state, info = env.reset()
    print("Initial State:", state)
    print("Info after reset:", info)

    # Perform a step
    state, reward, done, info = env.step(0)
    print("State after step:", state)
    print("Reward:", reward)
    print("Info after step:", info)
