"""
This module defines the `Board` class, which represents the game board for a rush hour puzzle game.
The board manages the placement and movement of vehicles, ensuring they follow rules and constraints.
"""

import random

import numpy as np

from vehicles import RedCar


class Board:
    """
    Represents the game board for the vehicle puzzle game.

    Attributes:
        vehicles (list): A list of vehicles currently on the board.
        row (int): The number of rows on the board.
        col (int): The number of columns on the board.
        board (numpy.ndarray): A 2D array representing the board state.
    """

    def __init__(self, row: int = 6, col: int = 6):
        """
        Initializes a new game board with a default size of 6x6 and adds the red car.

        Args:
            row (int): The number of rows on the board. Default is 6.
            col (int): The number of columns on the board. Default is 6.
        """
        self.row = row
        self.col = col
        self.reset()


    def reset(self):
        """
        Resets the board by removing all vehicles except the red car.
        """
        self.vehicles = []
        self.board = np.empty((self.row, self.col), dtype=str)
        self.add_vehicle(RedCar(), 2, 4)


    def add_vehicle(self, vehicle, row: int, col: int):
        """
        Adds a vehicle to the board at the specified position.

        Args:
            vehicle: The vehicle to add.
            row (int): The starting row of the vehicle.
            col (int): The starting column of the vehicle.
        """
        if vehicle.direction == "RL":
            self.board[row, col:col + vehicle.length] = vehicle.symbol
        else:
            self.board[row:row + vehicle.length, col] = vehicle.symbol
        vehicle.row = row
        vehicle.col = col
        self.vehicles.append(vehicle)

    def add_random_vehicle(self, vehicle):
        """
        Attempts to add a vehicle at a random position on the board. If placement fails,
        it changes the vehicle's direction after 100 attempts and gives up after 200 attempts.

        Args:
            vehicle: The vehicle to add.

        Returns:
            bool: True if the vehicle was added successfully, False otherwise.
        """
        count = 0
        row = random.choice(range(self.row))
        col = random.choice(range(self.col))
        while not self.check_add_vehicle(vehicle, row, col):
            row = random.choice(range(self.row))
            col = random.choice(range(self.col))
            count += 1
            if count == 100:
                if vehicle.direction == "RL":
                    vehicle.direction = "UD"
                else:
                    vehicle.direction = "RL"
            if count > 200:
                print("Cannot add car")
                return False
        self.add_vehicle(vehicle, row, col)
        return True

    def check_add_vehicle(self, vehicle, row: int, col: int):
        """
        Checks if a vehicle can be added at the specified position, ensuring no conflicts
        with other vehicles or board boundaries.

        Args:
            vehicle: The vehicle to check.
            row (int): The starting row.
            col (int): The starting column.

        Returns:
            bool: True if the vehicle can be placed, False otherwise.
        """
        # Direction-based uniqueness check
        if vehicle.direction == "RL":
            for v in self.vehicles:
                if v.direction == "RL" and v.row == row:
                    return False
        else:
            for v in self.vehicles:
                if v.direction == "UD" and v.col == col:
                    return False

        # Check boundaries and collisions
        if vehicle.direction == "RL":
            if col + vehicle.length > self.col:
                return False
            if np.any(self.board[row, col:col + vehicle.length] != ""):
                return False
        else:
            if row + vehicle.length > self.row:
                return False
            if np.any(self.board[row:row + vehicle.length, col] != ""):
                return False

        return True

    def move_vehicle(self, vehicle, move: str):
        """
        Moves a vehicle on the board in the specified direction if the move is valid.

        Args:
            vehicle: The vehicle to move.
            move (str): The direction to move ("L", "R", "U", "D").

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        if move not in vehicle.get_possible_moves(self):
            return False
        if vehicle.direction == "RL":
            if move == "L":
                vehicle.col -= 1
                self.board[vehicle.row, vehicle.col + vehicle.length] = ""
                self.board[vehicle.row, vehicle.col] = vehicle.symbol
            elif move == "R":
                vehicle.col += 1
                self.board[vehicle.row, vehicle.col - 1] = ""
                self.board[vehicle.row, vehicle.col + vehicle.length - 1] = vehicle.symbol
        else:
            if move == "U":
                vehicle.row -= 1
                self.board[vehicle.row + vehicle.length, vehicle.col] = ""
                self.board[vehicle.row, vehicle.col] = vehicle.symbol
            elif move == "D":
                vehicle.row += 1
                self.board[vehicle.row - 1, vehicle.col] = ""
                self.board[vehicle.row + vehicle.length - 1, vehicle.col] = vehicle.symbol
        return True

    def random_move(self):
        """
        Randomly selects a vehicle and moves it in a valid direction.
        """
        flag = True
        while flag:
            vehicle = random.choice(self.vehicles)
            moves = vehicle.get_possible_moves(self)
            if moves:
                move = random.choice(moves)
                self.move_vehicle(vehicle, move)
                flag = False

    def empty_space(self, row: int, col: int) -> bool:
        """
        Checks if a specific cell on the board is empty.

        Args:
            row (int): The row index.
            col (int): The column index.

        Returns:
            bool: True if the cell is empty, False otherwise.
        """
        if row < 0 or row >= self.row or col < 0 or col >= self.col:
            return False
        return self.board[row, col] == ""

    def check_win(self) -> bool:
        """
        Checks if the red car has reached the winning position.

        Returns:
            bool: True if the red car is at the winning position, False otherwise.
        """
        return self.board[2,5] != "X" and self.board[2,5] != ""

    def __str__(self):
        """
        Returns a string representation of the board.

        Returns:
            str: The board as a string.
        """
        return str(self.board)
