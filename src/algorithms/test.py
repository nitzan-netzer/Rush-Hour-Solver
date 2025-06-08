import time

from ASTAR import astar
from BFS import bfs

from environments.board import Board
from algorithms.utils import print_solution

def run_and_time(algorithm,card):
    start_time = time.time()
    solution = algorithm(card)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"{algorithm.__name__} time: {time_taken} seconds")
    return solution, time_taken

def main():
    total_time_bfs = 0
    total_time_astar = 0
    for i in range(1, 11):
        print(f"card{i}:")
        card = Board.load(f"database/original/cards/card{i}.json")
        solution_bfs, time_bfs = run_and_time(bfs,card)
        solution_astar, time_astar = run_and_time(astar,card)
        total_time_bfs += time_bfs
        total_time_astar += time_astar

        if solution_bfs != solution_astar:
           print(f"card{i} astar solution not equal to bfs solution")
           if len(solution_bfs) != len(solution_astar):
                print(f"card{i}bfs solution length not equal to astar solution length")
                print_solution(solution_bfs)
                print_solution(solution_astar)
    print("--------------------------------")
    print(f"average time BFS: {total_time_bfs/10} seconds")
    print(f"average time A*: {total_time_astar/10} seconds")

if __name__ == "__main__":
    main()
