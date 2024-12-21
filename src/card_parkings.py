"""
Generate a board with all vehicles parked in the parking lot.
"""
import os

from board import Board
from board_to_image import save_board_to_image
from vehicles import Car, Truck


def main():
    """
    Generate a board with all vehicles parked in the parking lot.
    """
    board = Board()
    board.add_vehicle(Car("RL", "X"), 0, 0)
    board.add_vehicle(Car("RL", "A"), 0, 2)
    board.add_vehicle(Car("RL", "B"), 0, 4)
    board.add_vehicle(Car("RL", "C"), 1, 0)
    board.add_vehicle(Car("RL", "D"), 1, 2)
    board.add_vehicle(Car("RL", "E"), 1, 4)
    board.add_vehicle(Car("RL", "F"), 2, 0)
    board.add_vehicle(Car("RL", "G"), 2, 2)
    board.add_vehicle(Car("RL", "H"), 2, 4)
    board.add_vehicle(Car("RL", "I"), 3, 0)
    board.add_vehicle(Car("RL", "J"), 3, 2)
    board.add_vehicle(Car("RL", "K"), 3, 4)
    board.add_vehicle(Truck("RL", "O"), 4, 0)
    board.add_vehicle(Truck("RL", "P"), 4, 3)
    board.add_vehicle(Truck("RL", "Q"), 5, 0)
    board.add_vehicle(Truck("RL", "R"), 5, 3)

    path_parking = "database/parking"
    os.makedirs(path_parking, exist_ok=True)
    path_parking_with_latter = os.path.join(path_parking, "board-with-latter.png")
    path_parking_without_latter = os.path.join(path_parking, "board-without-latter.png")
    save_board_to_image(board, path_parking_with_latter, draw_letters=True)
    save_board_to_image(board, path_parking_without_latter, draw_letters=False)


if __name__ == "__main__":
    main()
