from board import Board
from board_to_image import save_board_to_image
from vehicles import Car, Truck


def main():
    """
    Generate a card with a red car at the exit.
    """
    board = Board()
    board.reset()
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

    path_board_parking_with_latter = "database/board-parking-with-latter.png"
    path_board_parking_without_latter = "database/board-parking-without-latter.png"
    save_board_to_image(board, path_board_parking_with_latter, draw_letters=True)
    save_board_to_image(board, path_board_parking_without_latter, draw_letters=False)


if __name__ == "__main__":
    main()