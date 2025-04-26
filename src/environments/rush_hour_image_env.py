import numpy as np
from gymnasium import Env, spaces
from random import choice
from copy import deepcopy
from environments.board import Board
from GUI.board_to_image import generate_board_image


class RushHourImageEnv(Env):
    def __init__(self, boards, num_vehicles, image_size=(300, 300)):
        """
        Args:
            boards (list): List of Board objects to sample from.
            num_vehicles (int): Number of vehicles (determines action space).
            image_size (tuple): Output image size (width, height).
        """
        super().__init__()
        self.boards = boards
        self.num_vehicles = num_vehicles
        self.image_size = image_size

        # Action space: num_vehicles * 4 (each vehicle: [U,D,L,R])
        self.action_space = spaces.Discrete(num_vehicles * 4)

        # Observation space: image (H, W, C) - normalized [0,1]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(*image_size, 3), dtype=np.float32)

        self.board = None
        self.state = None
        self.num_steps = 0
        self.max_steps = 100

        # Vehicles letters (adapt this if needed)
        self.vehicles_letter = ["A", "B", "C", "D", "O", "X"]

    def reset(self, seed=None, options=None):
        self.board = deepcopy(choice(self.boards))
        self.num_steps = 0
        self.state = self._board_to_image(self.board)
        return self.state, self._get_info()

    def step(self, action):
        vehicle_str, move_str = self._parse_action(action)
        vehicle = self.board.get_vehicle_by_letter(vehicle_str)

        valid_move = False
        done = False

        if vehicle:
            valid_move = self.board.move_vehicle(vehicle, move_str)
            done = self.board.game_over()

        self.num_steps += 1
        truncated = self.num_steps >= self.max_steps
        reward = self._compute_reward(valid_move, done, truncated)

        self.state = self._board_to_image(self.board)
        return self.state, reward, done, truncated, self._get_info()

    def render(self):
        img = generate_board_image(
            self.board, scale=self.image_size[0] // 6, draw_letters=True)
        img.show()

    def _board_to_image(self, board):
        img = generate_board_image(
            board, scale=self.image_size[0] // 6, draw_letters=False)
        img = img.resize(self.image_size)
        img_array = np.asarray(img).astype(np.float32) / 255.0  # Normalize
        return img_array

    def _parse_action(self, action):
        vehicle = action // 4
        move = action % 4
        move_str = ["U", "D", "L", "R"][move]
        vehicle_str = self.vehicles_letter[vehicle]
        return vehicle_str, move_str

    def _compute_reward(self, valid_move, done, truncated):
        reward = -1  # Encourage shorter solutions
        if not valid_move:
            reward -= 5
        if done:
            reward += 1000
        if truncated:
            reward -= 100
        return reward

    def _get_info(self):
        return {
            "steps": self.num_steps,
        }


# Example usage:
if __name__ == "__main__":
    from environments.board import Board

    boards = Board.load_multiple_boards(
        "database/1000_cards_3_cars_1_trucks.json")
    env = RushHourImageEnv(boards, num_vehicles=6)

    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    env.render()

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        env.render()
