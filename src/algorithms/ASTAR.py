from copy import deepcopy
from heapq import heappop, heappush
from itertools import count

import setup_path  # NOQA
from algorithms.utils import print_solution
from environments.board import Board


def astar(board: Board):
    """
    Solve the board using A* search algorithm.
    """
    # Priority queue: (f_score, counter, g_score, board, path)
    # f_score = g_score + heuristic
    # g_score = number of moves so far
    # counter is used as a tiebreaker when f_scores are equal
    counter = count()
    open_set = []
    heappush(open_set, (board.get_heuristic(), next(counter), 0, board, []))
    
    # Keep track of visited states and their g_scores
    visited = {}
    visited[board.get_hash()] = 0
    
    while open_set:
        f_score, _, g_score, current_board, path = heappop(open_set)
        
        # Check if we've reached the goal state
        if current_board.game_over():
            return path
        
        # Skip if we found a better path to this state
        if g_score > visited.get(current_board.get_hash(), float('inf')):
            continue
        
        # Generate all possible next moves
        for vehicle in current_board.vehicles:
            possible_moves = vehicle.get_possible_moves(current_board)
            for move in possible_moves:
                # Create new board only if move is possible
                new_board = deepcopy(current_board)
                vehicle_copy = new_board.get_vehicle_by_letter(vehicle.letter)
                if vehicle_copy and new_board.move_vehicle(vehicle_copy, move):
                    new_g_score = g_score + 1
                    board_hash = new_board.get_hash()
                    
                    # Only proceed if this is a better path to this state
                    if new_g_score < visited.get(board_hash, float('inf')):
                        visited[board_hash] = new_g_score
                        new_path = path + [(vehicle.letter, move)]
                        new_f_score = new_g_score + new_board.get_heuristic()
                        heappush(open_set, (new_f_score, next(counter), new_g_score, new_board, new_path))
    
    return None  # No solution found


if __name__ == "__main__":
    card = Board.load(f"database/original/cards/card1.json")
    solution = astar(card)
    print_solution(solution)

