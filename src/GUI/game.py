import setup_path # NOQA
import pygame
import sys
from environments.board import Board
from GUI.board_to_image import letter_to_color, GRAY, WHITE
from copy import deepcopy
from random import choice
from algorithms.BFS import bfs
from algorithms.ASTAR import astar

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 80
BOARD_SIZE = 6
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE
MARGIN = 50
TOTAL_SIZE = WINDOW_SIZE + 2 * MARGIN
BUTTON_HEIGHT = 40
BUTTON_WIDTH = 150
BUTTON_MARGIN = 20

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow for selected vehicle
BUTTON_COLOR = (200, 200, 200)
BUTTON_HOVER_COLOR = (180, 180, 180)

class Button:
    def __init__(self, x, y, width, height, text, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False

    def draw(self, screen, font):
        color = BUTTON_HOVER_COLOR if self.hovered else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        text_surface = font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                return self.action
        return None

class RushHourGame:
    def __init__(self, all_boards=None,initial_board=None):
        self.screen = pygame.display.set_mode((TOTAL_SIZE, TOTAL_SIZE + BUTTON_HEIGHT + BUTTON_MARGIN))
        pygame.display.set_caption("Rush Hour Game")
        self.clock = pygame.time.Clock()
        
        self.all_boards = all_boards

        # Initialize board
        if initial_board is None:
            self.new_level()
        else:
            self.initial_board = deepcopy(initial_board)
            self.restart_game()
            
        
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 48)
        
        # Initialize buttons
        button_y = TOTAL_SIZE + BUTTON_MARGIN
        self.buttons = [
            Button(MARGIN, button_y, BUTTON_WIDTH, BUTTON_HEIGHT, "Restart", self.restart_game),
            Button(TOTAL_SIZE - MARGIN - BUTTON_WIDTH, button_y, BUTTON_WIDTH, BUTTON_HEIGHT, "New Level", self.new_level),
            Button(TOTAL_SIZE - MARGIN - BUTTON_WIDTH * 2 - 10, button_y, BUTTON_WIDTH, BUTTON_HEIGHT, "Solve", self.solve_game)
        ]


    def restart_game(self):
        self.board = deepcopy(self.initial_board)
        self.selected_vehicle = None
        self.game_over = False
        self.steps = 0

    def new_level(self):
        if self.all_boards:
            new_board = choice(self.all_boards)
            self.board = deepcopy(new_board)
            self.initial_board = deepcopy(new_board)
            self.selected_vehicle = None
            self.game_over = False
            self.steps = 0

    def show_win_choice(self):
        # Create a semi-transparent overlay
        overlay = pygame.Surface((TOTAL_SIZE, TOTAL_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Black with 50% opacity
        
        # Create message box
        box_width = 400
        box_height = 200
        box_x = (TOTAL_SIZE - box_width) // 2
        box_y = (TOTAL_SIZE - box_height) // 2
        
        # Draw message box
        pygame.draw.rect(overlay, WHITE, (box_x, box_y, box_width, box_height))
        pygame.draw.rect(overlay, BLACK, (box_x, box_y, box_width, box_height), 2)
        
        # Draw win message
        text = self.big_font.render("You Win!", True, BLACK)
        text_rect = text.get_rect(center=(TOTAL_SIZE // 2, box_y + 40))
        
        # Draw steps
        steps_text = self.font.render(f"Total Steps: {self.steps}", True, BLACK)
        steps_rect = steps_text.get_rect(center=(TOTAL_SIZE // 2, box_y + 80))
        
        # Create choice buttons
        button_width = 120
        button_height = 40
        button_y = box_y + 120
        
        exit_button = Button(
            TOTAL_SIZE // 2 - button_width - 10,
            button_y,
            button_width,
            button_height,
            "Exit",
            lambda: sys.exit()
        )
        
        new_level_button = Button(
            TOTAL_SIZE // 2 + 10,
            button_y,
            button_width,
            button_height,
            "New Level",
            self.new_level
        )
        
        # Blit everything
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)
        self.screen.blit(steps_text, steps_rect)
        exit_button.draw(self.screen, self.font)
        new_level_button.draw(self.screen, self.font)
        pygame.display.flip()
        
        # Wait for user choice
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEMOTION:
                    exit_button.handle_event(event)
                    new_level_button.handle_event(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    action = exit_button.handle_event(event)
                    if action:
                        action()
                    action = new_level_button.handle_event(event)
                    if action:
                        action()
                        waiting = False

    def draw_board(self):
        self.screen.fill(WHITE)
        
        # Draw grid
        for i in range(BOARD_SIZE + 1):
            # Vertical lines
            pygame.draw.line(self.screen, BLACK, 
                           (MARGIN + i * CELL_SIZE, MARGIN),
                           (MARGIN + i * CELL_SIZE, MARGIN + WINDOW_SIZE))
            # Horizontal lines
            pygame.draw.line(self.screen, BLACK,
                           (MARGIN, MARGIN + i * CELL_SIZE),
                           (MARGIN + WINDOW_SIZE, MARGIN + i * CELL_SIZE))
        
        # Draw vehicles using colors from board_to_image
        for vehicle in self.board.vehicles:
            # Get base color from letter_to_color
            color = letter_to_color.get(vehicle.letter, GRAY)
            
            # Calculate rectangle coordinates
            if vehicle.direction == "RL":
                rect = pygame.Rect(
                    MARGIN + vehicle.col * CELL_SIZE,
                    MARGIN + vehicle.row * CELL_SIZE,
                    vehicle.length * CELL_SIZE,
                    CELL_SIZE
                )
            else:
                rect = pygame.Rect(
                    MARGIN + vehicle.col * CELL_SIZE,
                    MARGIN + vehicle.row * CELL_SIZE,
                    CELL_SIZE,
                    vehicle.length * CELL_SIZE
                )
            
            # Draw the vehicle
            pygame.draw.rect(self.screen, color, rect)
            
            # Draw highlight for selected vehicle
            if self.selected_vehicle and self.selected_vehicle.letter == vehicle.letter:
                pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect, 3)  # Draw yellow border
            
            # Draw vehicle letter
            text = self.font.render(vehicle.letter, True, WHITE)
            text_rect = text.get_rect(center=(
                MARGIN + vehicle.col * CELL_SIZE + CELL_SIZE // 2,
                MARGIN + vehicle.row * CELL_SIZE + CELL_SIZE // 2
            ))
            self.screen.blit(text, text_rect)
        
        # Draw exit as a stripe, moved right one tile and up half a tile
        exit_x = MARGIN + 6 * CELL_SIZE  # Move right one tile
        exit_y = MARGIN + 2 * CELL_SIZE  # Move up half a tile (remove + CELL_SIZE // 2 - 5)
        pygame.draw.rect(self.screen, BLACK,
                        (exit_x,
                         exit_y,
                         10,
                         CELL_SIZE))
        
        # Draw step counter (always at bottom)
        steps_text = self.font.render(f"Steps: {self.steps}", True, BLACK)
        steps_rect = steps_text.get_rect(center=(TOTAL_SIZE // 2, TOTAL_SIZE + BUTTON_MARGIN // 2))
        self.screen.blit(steps_text, steps_rect)
        
        # Draw buttons
        for button in self.buttons:
            button.draw(self.screen, self.font)

    def handle_click(self, pos):
        if self.game_over:
            return
            
        # Check if click is on a button
        for button in self.buttons:
            action = button.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {'pos': pos}))
            if action:
                action()
                return
        
        # If not on a button, handle board click
        x, y = pos
        if y < TOTAL_SIZE:  # Only handle clicks on the board area
            board_x = (x - MARGIN) // CELL_SIZE
            board_y = (y - MARGIN) // CELL_SIZE
            
            if 0 <= board_x < BOARD_SIZE and 0 <= board_y < BOARD_SIZE:
                cell_value = self.board.board[board_y, board_x]
                if cell_value != "":
                    self.selected_vehicle = self.board.get_vehicle_by_letter(cell_value)
                else:
                    self.selected_vehicle = None

    def handle_key(self, key):
        if self.selected_vehicle is None or self.game_over:
            return
            
        move = None
        if key == pygame.K_LEFT:
            move = "L"
        elif key == pygame.K_RIGHT:
            move = "R"
        elif key == pygame.K_UP:
            move = "U"
        elif key == pygame.K_DOWN:
            move = "D"
            
        if move and self.board.move_vehicle(self.selected_vehicle, move):
            self.steps += 1
            if self.board.game_over():
                self.game_over = True
                self.draw_board()  # Update the display
                self.show_win_choice()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    for button in self.buttons:
                        button.handle_event(event)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
            
            self.draw_board()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

    def solve_game(self,algorithm_solver=astar):
        """
        Solve the current game board using a breadth-first search algorithm and show the solution in real-time.
        """
        print("Solving game...")
        
        # Get solution using astar
        solution = algorithm_solver(self.board)
        
        if solution:
            # Create a surface for the solving message
            message_surface = pygame.Surface((TOTAL_SIZE, TOTAL_SIZE), pygame.SRCALPHA)
            message_surface.fill((0, 0, 0, 128))  # Semi-transparent black
            
            # Apply the solution path to the actual game board with delay
            for vehicle_letter, move in solution:
                vehicle = self.board.get_vehicle_by_letter(vehicle_letter)
                if vehicle:
                    self.board.move_vehicle(vehicle, move)
                    self.steps += 1
                    self.draw_board()
                    
                    # Draw solving message
                    message_surface.fill((0, 0, 0, 128))
                    text = self.font.render("Solving...", True, WHITE)
                    text_rect = text.get_rect(center=(TOTAL_SIZE//2, TOTAL_SIZE//2))
                    message_surface.blit(text, text_rect)
                    self.screen.blit(message_surface, (0, 0))
                    
                    pygame.display.flip()
                    pygame.time.delay(200)  # Delay between moves
            
            self.game_over = True
            self.draw_board()
            self.show_win_choice()
        else:
            print("No solution found!")

def play_game(all_boards,initial_board=None):
    game = RushHourGame(all_boards,initial_board)
    game.run()



if __name__ == "__main__":
    boards = Board.load_multiple_boards("database/50_cards_7_cars_3_trucks.json")
    card1 = Board.load("database/original/cards/card1.json")
    card2 = Board.load("database/original/cards/card2.json")
    card3 = Board.load("database/original/cards/card3.json")
    card4 = Board.load("database/original/cards/card4.json")
    card5 = Board.load("database/original/cards/card5.json")
    #play_game(boards,card1)
    play_game(boards)
