"""
vehicles.py

This module defines classes for vehicles, including Car, RedCar, and Truck.
Each class includes attributes for position, direction, and length,
along with methods for handling movement and state.
"""

from abc import ABC
from typing import List


class Vehicle(ABC):
    """
    Represents a generic vehicle on the board.

    Attributes:
        length (int): The number of squares the vehicle occupies.
        direction (str): Whether the vehicle is horizontal or vertical.
        letter (str): A single-character letter representing the vehicle on the board.
        row (int): The topmost row position of the vehicle.
        col (int): The leftmost column position of the vehicle.
    """

    def __init__(self, length: int, direction: str, letter: str, row: int, col: int):
        self._length = length
        self._direction = direction
        self._letter = letter
        self.row = row
        self.col = col

    @property
    def length(self) -> int:
        return self._length

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def letter(self) -> str:
        return self._letter

    def change_direction(self):
        """
        Change the vehicle's direction from horizontal to vertical or vice versa.
        """
        if self.direction == "RL":
            self._direction = "UD"
        else:
            self._direction = "RL"

    def get_possible_moves(self, board) -> List[str]:
        """
        Determine possible move directions based on the vehicle's direction and the board's layout.

        Args:
            board: An object that provides an `empty_space(row, col)` method
                   returning True if the given position is free.

        Returns:
            A list of strings representing possible move directions.
            For a horizontal vehicle: ['L', 'R']
            For a vertical vehicle: ['U', 'D']
        """
        moves = []
        if self.direction == "RL":
            can_move_right = board.empty_space(self.row, self.col + self.length)
            can_move_left = board.empty_space(self.row, self.col - 1)

            if can_move_left:
                moves.append("L")
            if can_move_right:
                moves.append("R")
        else:  # Direction.UP_DOWN
            can_move_up = board.empty_space(self.row - 1, self.col)
            can_move_down = board.empty_space(self.row + self.length, self.col)

            if can_move_up:
                moves.append("U")
            if can_move_down:
                moves.append("D")

        return moves


class Car(Vehicle):
    """
    A Car is a type of Vehicle with length 2.
    """

    def __init__(self, direction: str, letter: str, row: int = -1, col: int = -1):
        super().__init__(2, direction, letter, row, col)


class RedCar(Car):
    """
    A specific Car instance representing the red car.
    Typically placed at a known position.
    """

    def __init__(self):
        super().__init__("RL", "X", 2, 4)


class Truck(Vehicle):
    """
    A Truck is a type of Vehicle with length 3.
    """

    def __init__(self, direction: str, letter: str, row: int = -1, col: int = -1):
        super().__init__(3, direction, letter, row, col)


def create_vehicle(vehicle_data: dict) -> Vehicle:
    """
    Create a vehicle object based on the provided data.
    Args:
        vehicle_data (dict): A dictionary containing vehicle data.
        The dictionary must contain a 'type' key with a value of 'Car', 'Truck', or 'RedCar'.
        For 'Car' and 'Truck' types, the dictionary must also contain 'direction' and 'letter' keys.

    Returns:
      Vehicle: A vehicle object based on the provided data.

    Raises:
        ValueError: If the vehicle type is unknown.
    """
    if vehicle_data["type"] == "RedCar":
        return RedCar()
    elif vehicle_data["type"] == "Car":
        return Car(vehicle_data["direction"], vehicle_data["letter"])
    elif vehicle_data["type"] == "Truck":
        return Truck(vehicle_data["direction"], vehicle_data["letter"])
    else:
        raise ValueError(f"Unknown vehicle type: {vehicle_data['type']}")
