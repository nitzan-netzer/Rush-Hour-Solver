import random

from board import Board


class BoardRandom(Board):
    def __init__(self, row: int = 6, col: int = 6):
        super().__init__(row, col)

    def add_random_vehicle(self, vehicle):
        """
        Attempts to add a vehicle at a random position on the board. If placement fails,
        it changes the vehicle's direction after 100 attempts and gives up after 200 attempts.

        Args:
            vehicle: The vehicle to add.

        Returns:
            bool: True if the vehicle was added successfully, False otherwise.
        """
        count = 0
        row = random.choice(range(self.row))
        col = random.choice(range(self.col))
        while not self.check_add_vehicle(vehicle, row, col):
            row = random.choice(range(self.row))
            col = random.choice(range(self.col))
            count += 1
            if count == 100:
                vehicle.change_direction()
            if count > 200:
                print("Cannot add car")
                return False
        self.add_vehicle(vehicle, row, col)
        return True

    def random_move(self):
        """
        Randomly selects a vehicle and moves it in a valid direction.
        returns True if a move was made, False otherwise.
        """
        random.shuffle(self.vehicles)
        for vehicle in self.vehicles:
            moves = vehicle.get_possible_moves(self)
            if moves:
                move = random.choice(moves)
                self.move_vehicle(vehicle, move)
                return True
        return False
