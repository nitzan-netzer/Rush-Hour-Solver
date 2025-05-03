import time

from ASTAR import astar
from BFS import bfs

from environments.board import Board
from algorithms.utils import print_solution

def run_and_time(algorithm,card):
    start_time = time.time()
    solution = algorithm(card)
    end_time = time.time()
    print(f"{algorithm.__name__} time: {end_time - start_time} seconds")
    return solution

def main():
    for i in range(1,8):
        print(f"card{i}:")
        card = Board.load(f"database/original/cards/card{i}.json")
        solution_bfs = run_and_time(bfs,card)
        solution_astar = run_and_time(astar,card)

        if solution_bfs != solution_astar:
           print(f"card{i} astar solution not equal to bfs solution")
           if len(solution_bfs) != len(solution_astar):
                print(f"card{i}bfs solution length not equal to astar solution length")
                print_solution(solution_bfs)
                print_solution(solution_astar)
    

if __name__ == "__main__":
    main()
