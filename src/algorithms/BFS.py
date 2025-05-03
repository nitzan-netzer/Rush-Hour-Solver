from collections import deque
from copy import deepcopy

import setup_path  # NOQA
from environments.board import Board
from algorithms.utils import print_solution


def bfs(board: Board):
    """
    Solve the board using a breadth-first search algorithm.
    """
    # Initialize the queue with the current board state and empty path
    queue = deque([(board, [])])
    visited = set()
    visited.add(board.get_hash())
    
    while queue:
        current_board, path = queue.popleft()
        
        # Check if we've reached the goal state
        if current_board.game_over():
            return path

        # Generate all possible next moves
        for vehicle in current_board.vehicles:
            # Only try moves that are possible for this vehicle
            possible_moves = vehicle.get_possible_moves(current_board)
            for move in possible_moves:
                # Create new board only if move is possible
                new_board = deepcopy(current_board)
                vehicle_copy = new_board.get_vehicle_by_letter(vehicle.letter)
                if vehicle_copy and new_board.move_vehicle(vehicle_copy, move):
                    board_hash = new_board.get_hash()
                    if board_hash not in visited:
                        visited.add(board_hash)
                        new_path = path + [(vehicle.letter, move)]
                        queue.append((new_board, new_path))
    
    return None  # No solution found



if __name__ == "__main__":
    card = Board.load(f"database/original/cards/card1.json")
    solution = bfs(card)
    print_solution(solution)

