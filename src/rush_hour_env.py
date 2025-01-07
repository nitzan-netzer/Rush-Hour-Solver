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
        return self.state, {}  # Return state and empty info dictionary

    def step(self, action):
        # Update state based on action
        reward = -1  # Example reward structure
        done = False  # Example: check if game is over
        info = {}
        return self.state, reward, done, info

    def render(self):
        # Visualize the environment (optional)
        print(self.state)

# Registering the environment
gym.register(
    id="RushHour-v0",
    entry_point="RushHourEnv", 
)