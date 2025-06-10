from gymnasium import Env, spaces
from copy import deepcopy
from random import choice
import numpy as np
from gymnasium import Env, spaces


from environments.board import Board
from environments.rewards import basic_reward
from environments.init_boards_from_database import initialize_boards, load_specific_board_file



class RushHourEnv(Env):
    train_boards, test_boards = initialize_boards()

    def __init__(self, num_of_vehicle: int = 6, min_vehicles: int = 4, rewards=basic_reward, train=True):
        super().__init__()
        self.boards = RushHourEnv.train_boards if train else RushHourEnv.test_boards
        self.max_steps = 100 if train else 50
        num_of_vehicle = len(self.boards[0].get_all_vehicles_letter())
        self.num_of_vehicle = num_of_vehicle
        self.get_reward = rewards
        size = self.boards[0].row * self.boards[0].col

        self.boards = RushHourEnv.train_boards if train else RushHourEnv.test_boards
        self.max_steps = 200 if train else 100

        self.board = None
        self.state = None
        self.num_steps = 0
        self.state_history = []
        self.vehicles_letter = []

        self.action_space = spaces.Discrete(self.max_vehicles * 4)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size,), dtype=np.uint8
        )

     def reset(self, board=None, seed=None,options=None):
        self.board = deepcopy(choice(self.boards)) if board is None else deepcopy(board)
        self.vehicles_letter = self.board.get_all_vehicles_letter()
        self.num_steps = 0
        self.total_reward = 0
        self.state = self.board.get_board_flatten().astype(np.uint8)
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

if __name__ == "__main__":
    env = RushHourEnv(num_of_vehicle=16)
    my_board = env.boards[0]
    obs, info = env.reset(my_board)
    print("Valid action indices:")
    print(np.nonzero(info["action_mask"])[0])
    env.render()