import random
from environments.board import Board
from GUI.board_to_image import save_board_to_image


def save_random_board_image(json_path, output_image_path, draw_letters=True):
    """
    Load a JSON dataset of boards, select a random board, and save it as an image.

    Args:
        json_path (str): Path to the JSON file containing multiple boards.
        output_image_path (str): Path where the output image will be saved.
        draw_letters (bool): Whether to draw letters on the vehicles in the image.
    """
    boards = Board.load_multiple_boards(json_path)
    random_board = random.choice(boards)
    save_board_to_image(random_board, output_image_path,
                        draw_letters=draw_letters)
    print(f"Original board saved as image to {output_image_path}")
    return random_board


def move_random_vehicle_and_save(board, output_image_path, draw_letters=True):
    """
    Selects a random vehicle from the board, makes a valid move, and saves the new board.

    Args:
        board (Board): The current board object.
        output_image_path (str): Path where the updated board image will be saved.
    """
    movable_vehicles = [
        v for v in board.vehicles if v.get_possible_moves(board)]

    if not movable_vehicles:
        print("No movable vehicles found.")
        return

    vehicle = random.choice(movable_vehicles)
    moves = vehicle.get_possible_moves(board)
    move = random.choice(moves)

    # Perform the move
    board.move_vehicle(vehicle, move)
    print(f"Moved vehicle '{vehicle.letter}' in direction '{move}'.")

    # Save the updated board
    save_board_to_image(board, output_image_path, draw_letters=draw_letters)
    print(f"Updated board saved as image to {output_image_path}")


if __name__ == "__main__":
    json_dataset_path = "database/1000_cards_3_cars_1_trucks.json"  # Adjust as needed
    original_image_file = "random_board_original.png"
    updated_image_file = "random_board_moved.png"

    # Step 1: Load and save the original board
    board = save_random_board_image(json_dataset_path, original_image_file)

    # Step 2: Move a vehicle and save the updated board
    move_random_vehicle_and_save(board, updated_image_file)
