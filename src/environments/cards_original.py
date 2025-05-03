"""
Generate the original cards and their solutions.
"""
import setup_path # NOQA
import os

from environments.board import Board
from GUI.board_to_image import save_board_to_image, save_board_to_video
from environments.vehicles import Car, Truck



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

    sol1 = ["CL3", "OD3", "AR1", "PU1", "BU1", "RL2", "QD2", "XR5"]

    return board, sol1


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

    sol2 = ["EL1", "PD2", "XR1", "AD1", "OL3", "CU2", "BU1", "XR5"]

    return board, sol2


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

    sol3 = ["PU3", "OU2", "CR2", "AR3", "OD3", "XR1", "BU4", "XL1",
            "OU3", "CL3", "AL3", "PD3", "OD3", "XR5"]
    return board, sol3


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
    
    sol4 = ["OD3", "XL1", "AU3", "XR1", "OU3", "RL2", "QL3", "PD3", "XR5"]
    return board, sol4


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

    sol5 = ["EL3", "GD1", "FL3", "QD2", "AR1", "PU1", "RL1", "OD3", "XR5"]
    return board, sol5

def card6():
    """
    Generate the initial setup for Card 6 and provide its solution moves.
    """
    board = Board()
    board.reset(init_red_car=False)

    board.add_vehicle(Car("RL", "X"), 2, 1)
    board.add_vehicle(Car("RL", "A"), 0, 0)
    board.add_vehicle(Car("UD", "B"), 0, 3)
    board.add_vehicle(Car("RL", "C"), 1, 0)
    board.add_vehicle(Car("RL", "D"), 3, 0)
    board.add_vehicle(Car("UD", "E"), 3, 2)
    board.add_vehicle(Car("UD", "F"), 4, 0)
    
    board.add_vehicle(Truck("UD", "O"), 1, 4)
    board.add_vehicle(Truck("UD", "P"), 1, 5)
    board.add_vehicle(Truck("UD", "Q"), 2, 3)
    board.add_vehicle(Truck("RL", "R"), 5, 3)

    sol6 = ["XL1", "EU3", "DR1", "FU1", "RL3", "PD2", "OD3", "QD1", "XR6"]
    return board, sol6

def card7():
    """
    Generate the initial setup for Card 7 and provide its solution moves.
    """
    board = Board()
    board.reset(init_red_car=False)

    board.add_vehicle(Car("RL", "X"), 2, 1)
    board.add_vehicle(Car("UP", "A"), 0, 1)
    board.add_vehicle(Car("RL", "B"), 0, 2)
    board.add_vehicle(Car("UP", "C"), 0, 4)
    board.add_vehicle(Car("UP", "D"), 0, 5)
    board.add_vehicle(Car("UP", "E"), 1, 3)
    board.add_vehicle(Car("UP", "F"), 2, 5)
    board.add_vehicle(Car("RL", "I"), 3, 2)
    board.add_vehicle(Car("UD", "H"), 4, 3)

    sol7 = ["FD1", "DD1", "CD3", "BR2", "EU1", "XR1", "AD3", "XL1", "ED1",
            "BL3", "DU1", "EU1", "XR5"]
    return board, sol7

def get_total_steps(solution):
    total_steps = 0
    for move in solution:
        total_steps += int(move[2])
    return total_steps

def main():
    """
    Generate the original cards and their solutions.
    """

    original_path = "database/original/cards/"
    video_path = "database/original/videos/"
    os.makedirs(original_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    cards = (card1, card2, card3, card4, card5, card6, card7)
    for card in cards:
            board, sol = card()
            name = card.__name__
            card_path = original_path + name + ".png"
            card_video_path = video_path + name + "_solution.mp4"
            board.save(card_path.replace(".png", ".json"))
            save_board_to_image(board, card_path, draw_letters=True)
            save_board_to_video(board, sol, card_video_path, draw_letters=False)
            print(f"{name} has {get_total_steps(sol)} steps and {len(sol)} moves")

if __name__ == "__main__":
    main()
