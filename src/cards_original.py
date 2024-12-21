import os

from board import Board
from board_to_image import save_board_to_image, save_board_to_video
from vehicles import Car, Truck

# fmt: off
sol1 = ["CL3", "OD3", "AR1", "PU1", "BU1", "RL2", "QD2", "XR5"]
sol2 = ["EL1", "PD2", "XR1", "AD1", "OL3", "CU2", "BU1", "XR5"]
sol3 = ["PU3", "OU2", "CR2", "AR3", "OD3", "XR1", "BU4", "XL1",
        "OU3", "CL3", "AL3", "PD3", "OD3", "XR5"]
sol4 = ["OD3", "XL1", "AU3", "XR1", "OU3", "RL2", "QL3", "PD3", "XR5"]
sol5 = ["EL3", "GD1", "FL3", "QD2", "AR1", "PU1", "RL1", "OD3", "XR5"]
# fmt: on


def card1():
    """
    Generate a card1 original.
    """
    board = Board()
    board.reset(init_red_car=False)
    board.add_vehicle(Car("RL", "X"), 2, 1)
    board.add_vehicle(Car("RL", "A"), 0, 0)
    board.add_vehicle(Car("UD", "B"), 4, 0)
    board.add_vehicle(Car("RL", "C"), 4, 4)
    board.add_vehicle(Truck("UD", "O"), 0, 5)
    board.add_vehicle(Truck("UD", "P"), 1, 0)
    board.add_vehicle(Truck("UD", "Q"), 1, 3)
    board.add_vehicle(Truck("RL", "R"), 5, 2)

    return board


def card2():
    """
    Generate a card2 original.
    """
    board = Board()
    board.reset(init_red_car=False)
    board.add_vehicle(Car("RL", "X"), 2, 0)
    board.add_vehicle(Car("UD", "A"), 0, 0)
    board.add_vehicle(Car("UD", "B"), 1, 3)
    board.add_vehicle(Car("UD", "C"), 2, 4)
    board.add_vehicle(Car("UD", "D"), 4, 2)
    board.add_vehicle(Car("RL", "E"), 4, 4)
    board.add_vehicle(Car("RL", "F"), 5, 0)
    board.add_vehicle(Car("RL", "G"), 5, 3)
    board.add_vehicle(Truck("RL", "O"), 0, 3)
    board.add_vehicle(Truck("UD", "P"), 1, 5)
    board.add_vehicle(Truck("RL", "Q"), 3, 0)

    return board


def card3():
    """
    Generate a card3 original.
    """
    board = Board()
    board.reset(init_red_car=False)
    board.add_vehicle(Car("RL", "X"), 2, 1)
    board.add_vehicle(Car("RL", "A"), 3, 1)
    board.add_vehicle(Car("UD", "B"), 4, 1)
    board.add_vehicle(Car("RL", "C"), 5, 2)
    board.add_vehicle(Truck("UD", "O"), 2, 3)
    board.add_vehicle(Truck("UD", "P"), 3, 5)

    return board


def card4():
    """
    Generate a card4 original.
    """
    board = Board()
    board.reset(init_red_car=False)
    board.add_vehicle(Car("RL", "X"), 2, 1)
    board.add_vehicle(Car("UD", "A"), 3, 2)
    board.add_vehicle(Car("UD", "B"), 4, 5)
    board.add_vehicle(Truck("UD", "O"), 0, 0)
    board.add_vehicle(Truck("UD", "P"), 0, 3)
    board.add_vehicle(Truck("RL", "Q"), 3, 3)
    board.add_vehicle(Truck("RL", "R"), 5, 2)

    return board


def card5():
    """
    Generate a card5 original.
    """
    board = Board()
    board.reset(init_red_car=False)
    board.add_vehicle(Car("RL", "X"), 2, 1)
    board.add_vehicle(Car("RL", "A"), 0, 0)
    board.add_vehicle(Car("UD", "B"), 0, 5)
    board.add_vehicle(Car("UD", "D"), 4, 0)
    board.add_vehicle(Car("RL", "E"), 4, 4)
    board.add_vehicle(Car("RL", "F"), 5, 4)
    board.add_vehicle(Car("UD", "G"), 2, 5)
    board.add_vehicle(Truck("UD", "O"), 0, 3)
    board.add_vehicle(Truck("UD", "P"), 1, 0)
    board.add_vehicle(Truck("UD", "Q"), 1, 4)
    board.add_vehicle(Truck("RL", "R"), 3, 1)

    return board


def main():
    original_path = "database/original/cards/"
    frame_path = "database/original/frames/"
    video_path = "database/original/videos/"
    os.makedirs(original_path, exist_ok=True)
    os.makedirs(frame_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    cards = (card1, card2, card3, card4, card5)
    solutions = (sol1, sol2, sol3, sol4, sol5)
    for card, sol in zip(cards, solutions):
        board = card()
        name = card.__name__
        card_path = original_path + name + ".png"
        card_video_path = video_path + name + "_solution.mp4"
        card_frame_path = frame_path + name
        save_board_to_image(board, card_path, draw_letters=True)
        save_board_to_video(
            board, sol, card_frame_path, card_video_path, draw_letters=False
        )


if __name__ == "__main__":
    main()
