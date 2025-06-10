
def get_solution(solution):
    """
    Get the solution path in a readable format.
    """
    solution_original = []
    if not solution:
        print("No solution found")
        return

    i = 0
    while i < len(solution):
        current_move = solution[i]
        count = 1
        
        # Count consecutive identical moves
        while i + count < len(solution) and solution[i + count] == current_move:
            count += 1
        solution_original.append(f"{current_move[0]}{current_move[1]}{count}")
        i += count
    return solution_original

def get_total_steps(solution_original):
    total_steps = 0
    for move in solution_original:
        total_steps += int(move[2])
    return total_steps

def get_total_moves(solution_original):
    return len(solution_original)

def print_solution(solution):
    """
    Print the solution path in a readable format.
    Args:
        solution: List of tuples containing (vehicle_letter, move_direction)
    """
    solution_original = get_solution(solution)
    print(f"Total steps: {get_total_steps(solution_original)}")
    print(f"Total moves: {get_total_moves(solution_original)}")
    print(solution_original)



