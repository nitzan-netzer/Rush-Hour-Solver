def print_solution(solution):
    """
    Print the solution path in a readable format.
    Args:
        solution: List of tuples containing (vehicle_letter, move_direction)
    """
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
            
        print(f"{current_move[0]}{current_move[1]}{count}", end="\t")
        i += count
    print(f"Total moves: {len(solution)}")