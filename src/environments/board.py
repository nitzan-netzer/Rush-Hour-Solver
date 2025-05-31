"""
This module defines the `Board` class, which represents the game board for a rush hour puzzle game.
The board manages the placement and movement of vehicles, ensuring they follow rules and constraints.
"""
import setup_path # NOQA

import json

import numpy as np

from environments.vehicles import RedCar, create_vehicle
from algorithms.utils import get_solution,get_total_steps
class Board:
    """
    Represents the game board for the vehicle puzzle game.

    Attributes:
        vehicles (list): A list of vehicles currently on the board.
        row (int): The number of rows on the board.
        col (int): The number of columns on the board.
        board (numpy.ndarray): A 2D array representing the board state.
    """

    def __init__(self, row: int = 6, col: int = 6, init_red_car=True):
        """
        Initializes a new game board with a default size of 6x6 and adds the red car.

        Args:
            row (int): The number of rows on the board. Default is 6.
            col (int): The number of columns on the board. Default is 6.
        """
        self.row = row
        self.col = col
        self.reset(init_red_car)

    def reset(self, init_red_car=True):
        """
        Resets the board by removing all vehicles except the red car.
        """
        self.vehicles = []
        self.board = np.empty((self.row, self.col), dtype=str)
        if init_red_car:
            self.add_vehicle(RedCar(), 2, 4)
        
        self.is_updated = False
        self.min_steps = 0
        self.heuristic = 0

    def update_heuristic_and_min_steps(self,func):
        if not self.is_updated:
            solution = func(self)
            solution_original = get_solution(solution)
            self.heuristic = self.get_heuristic()
            self.min_steps = get_total_steps(solution_original)
            self.is_updated = True

    def add_vehicle(self, vehicle, row: int, col: int):
        """
        Adds a vehicle to the board at the specified position.

        Args:
            vehicle: The vehicle to add.
            row (int): The starting row of the vehicle.
            col (int): The starting column of the vehicle.
        """
        if vehicle.direction == "RL":
            self.board[row, col: col + vehicle.length] = vehicle.letter
        else:
            self.board[row: row + vehicle.length, col] = vehicle.letter
        vehicle.row = row
        vehicle.col = col
        self.vehicles.append(vehicle)
        self.is_updated = False


    def check_add_vehicle(self, vehicle, row: int, col: int, uniqueness=False):
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
        if uniqueness:
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
            if np.any(self.board[row, col: col + vehicle.length] != ""):
                return False
        else:
            if row + vehicle.length > self.row:
                return False
            if np.any(self.board[row: row + vehicle.length, col] != ""):
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
                self.board[vehicle.row, vehicle.col] = vehicle.letter
            elif move == "R":
                vehicle.col += 1
                self.board[vehicle.row, vehicle.col - 1] = ""
                self.board[
                    vehicle.row, vehicle.col + vehicle.length - 1
                ] = vehicle.letter
        else:
            if move == "U":
                vehicle.row -= 1
                self.board[vehicle.row + vehicle.length, vehicle.col] = ""
                self.board[vehicle.row, vehicle.col] = vehicle.letter
            elif move == "D":
                vehicle.row += 1
                self.board[vehicle.row - 1, vehicle.col] = ""
                self.board[
                    vehicle.row + vehicle.length - 1, vehicle.col
                ] = vehicle.letter
        self.is_updated = False
        return True

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
        return self.board[2, 5] != "X" and self.board[2, 5] != ""

    def game_over(self) -> bool:
        return self.board[2, 5] == "X"

    def get_vehicle_by_letter(self, letter: str):
        """
        Returns the vehicle with the specified letter.

        Args:
            letter (str): The letter of the vehicle.

        Returns:
            Vehicle: The vehicle with the specified letter, or None if not found.
        """
        for vehicle in self.vehicles:
            if vehicle.letter == letter:
                return vehicle
        return None

    def __str__(self):
        """
        Saves the current board state as a string.

        Returns:
            str: The board state as a string.
        """
        str = ""
        for row in self.board:
            for cell in row:
                str += cell if cell != "" else "."
            str += "\n"
        return str

    def to_dict(self) -> dict:
        """
        Returns the current board state as a dictionary.

        Returns:
            dict: The board state as a dictionary.
        """
        json_board: dict = {
            "row": self.row, 
            "col": self.col,
            "heuristic": self.heuristic, 
            "min_steps": self.min_steps,
            "is_updated": self.is_updated,
            "vehicles": []
        }
        for vehicle in self.vehicles:
            json_board["vehicles"].append(
                {
                    "type": vehicle.__class__.__name__,
                    "letter": vehicle.letter,
                    "row": vehicle.row,
                    "col": vehicle.col,
                    "direction": vehicle.direction,
                 
                }
            )

        return json_board

    @staticmethod
    def from_dict(json_board: dict):
        """
        Loads a board state from a dictionary.

        Args:
            json_board (dict): The dictionary representing the board state.
        """
        row = json_board["row"]
        col = json_board["col"]
        board = Board(row, col, init_red_car=False)
        for vehicle_data in json_board["vehicles"]:
            vehicle = create_vehicle(vehicle_data)
            board.add_vehicle(
                vehicle, vehicle_data["row"], vehicle_data["col"])
            
        if "heuristic" in json_board:
            board.heuristic = json_board["heuristic"]
            board.min_steps = json_board["min_steps"]
            board.is_updated = json_board["is_updated"]
        else:
            board.is_updated = False
            board.min_steps = 0
            board.heuristic = 0
        return board

    def save(self, filename: str):
        """
        Saves the current board state to a file.

        Args:
            filename (str): The name of the file to save to.
        """
        json_board = self.to_dict()
        with open(filename, "w") as file:
            json.dump(json_board, file)

    @staticmethod
    def load(filename: str):
        """
        Loads a board state from a file.

        Args:
            filename (str): The name of the file to load from.
        """
        with open(filename, "r") as file:
            json_board = json.load(file)

        return Board.from_dict(json_board)

    @staticmethod
    def save_multiple_boards(boards, filename: str):
        """
        Saves multiple board states to a file.

        Args:
            boards (list): A list of board states to save.
            filename (str): The name of the file to save to.
        """
        json_boards = [board.to_dict() for board in boards]
        with open(filename, "w") as file:
            json.dump(json_boards, file)

    @staticmethod
    def load_multiple_boards(filename: str):
        """
        Loads multiple board states from a file.

        Args:
            filename (str): The name of the file to load from.
        """
        with open(filename, "r") as file:
            json_boards = json.load(file)

        return [Board.from_dict(json_board) for json_board in json_boards]

    def __eq__(self, other):
        """
        Compares two board states for equality.

        Args:
            other (Board): The other board state to compare.

        Returns:
            bool: True if the two board states are equal, False otherwise.
        """
        board_equal = np.array_equal(self.board, other.board)
        vehicles_len_equal = len(self.vehicles) == len(other.vehicles)

        return board_equal and vehicles_len_equal

    def get_all_moves(self):
        """
        Get all possible moves for all vehicles on the board.

        Returns:
            dict: A dictionary mapping vehicle letters to possible moves.
        """
        moves = []
        for vehicle in self.vehicles:
            if vehicle.direction == "RL":
                moves.append((vehicle.letter, "L"))
                moves.append((vehicle.letter, "R"))
            else:
                moves.append((vehicle.letter, "U"))
                moves.append((vehicle.letter, "D"))
        return tuple(moves)

    def get_board_flatten(self):
        """
        Get the board state as a flattened numpy array.

        Returns:
            numpy.ndarray: A flattened numpy array representing the board state.
        """
        arr = np.zeros((self.row * self.col), dtype=int)
        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):
                if cell != "":
                    arr[i * self.col + j] = ord(cell)
        return arr
    
    def get_hash(self):
        """
        Get a hash representation of the board state.

        Returns:
            int: A hash value representing the board state.
        """
        return hash(tuple(self.get_board_flatten()))
    
    def get_all_vehicles_letter(self):
        """"
        Get all vehicle letters on the board.
        """
    
        vehicles_str = [vehicle.letter for vehicle in self.vehicles]
        vehicles_str.sort()
        return vehicles_str

    def get_heuristic(self) -> int:
        """
        Calculate the heuristic value for the current board state.
        The heuristic is based on:
        1. Distance of red car from exit
        2. Number of blocking vehicles
        3. Number of moves needed to clear blocking vehicles
        """
        red_car = self.get_vehicle_by_letter("X")
        if not red_car:
            return float('inf')
        
        # Distance from red car to exit (column 5)
        distance_to_exit = 5 - (red_car.col + red_car.length)
    
        # Count blocking vehicles
        blocking_vehicles = 0
        for col in range(red_car.col + red_car.length, 6):
            if self.board[red_car.row, col] != "":
                blocking_vehicles += 1
        
        # Each blocking vehicle needs at least one move to clear
        return distance_to_exit + blocking_vehicles 