# Creating UI as described in the documentation of the project desing using pygame

import pygame
import time
import random
from stable_baselines3 import PPO
from board import Board
from board_to_image import generate_board_image
from rush_hour_env import RushHourEnv

# Constants
TILE_SIZE = 60
GRID_COLS = 6
GRID_ROWS = 8
PADDING = 20
TITLE_HEIGHT = 80
WINDOW_WIDTH = GRID_COLS * (TILE_SIZE + PADDING) + PADDING
WINDOW_HEIGHT = TITLE_HEIGHT + GRID_ROWS * (TILE_SIZE + PADDING) + PADDING
FONT_NAME = "arial"

# Load boards
all_boards = Board.load_multiple_boards("database/example-1000.json")
sample_boards = all_boards[:48]  # First 48 for levels

# Generate thumbnails (board_to_image returns PIL.Image)


def generate_thumbnails():
    thumbnails = []
    for board in sample_boards:
        # Generate a PIL image from the board (scale down for thumbnails)
        img = generate_board_image(board, scale=10, draw_letters=False)
        img = img.resize((TILE_SIZE, TILE_SIZE))  # TILE_SIZE is 60
        thumbnails.append(pygame.image.fromstring(
            img.tobytes(), img.size, img.mode))
    return thumbnails

# Solve the selected board


def solve_board(screen, model, board):
    env = RushHourEnv(num_of_vehicle=4, boards=[board])
    obs, _ = env.reset()

    draw_board(screen, env.board)
    time.sleep(1)

    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        draw_board(screen, env.board)
        time.sleep(0.3)
        if done:
            print("✅ Escaped!")
            font = pygame.font.SysFont(FONT_NAME, 48)
            text = font.render("✅ Escaped!", True, (0, 200, 0))
            screen.blit(text, (50, 50))
            pygame.display.flip()
            time.sleep(2)
            return

# Draw current board


def draw_board(screen, board_obj):
    screen.fill((240, 240, 240))
    board = board_obj.board
    for r in range(6):
        for c in range(6):
            letter = board[r, c]
            color = (200, 0, 0) if letter == "X" else (
                180, 180, 255) if letter != "" else (230, 230, 230)
            rect = pygame.Rect(c * 80, r * 80, 80, 80)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
    pygame.draw.line(screen, (255, 255, 255),
                     ((5 + 1) * 80, 2 * 80), ((5 + 1) * 80, 3 * 80), 5)
    pygame.display.flip()

# Main UI loop


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Rush Hour - Level Selector")

    font = pygame.font.SysFont(FONT_NAME, 18)
    big_font = pygame.font.SysFont(FONT_NAME, 28)
    model = PPO.load("ppo_rush_hour_model_es")
    thumbnails = generate_thumbnails()

    # Prepare level button rects
    buttons = []
    for idx in range(len(thumbnails)):
        col = idx % GRID_COLS
        row = idx // GRID_COLS
        x = PADDING + col * (TILE_SIZE + PADDING)
        y = TITLE_HEIGHT + row * (TILE_SIZE + PADDING)
        buttons.append(pygame.Rect(x, y, TILE_SIZE, TILE_SIZE))

    # "?" random button (top right)
    question_rect = pygame.Rect(WINDOW_WIDTH - 70, 10, 60, 60)

    running = True
    while running:
        screen.fill((255, 255, 255))
        title_text = big_font.render(
            "Select level or generate one", True, (0, 0, 0))
        screen.blit(title_text, (PADDING, 20))

        # Draw all thumbnails
        for idx, thumb in enumerate(thumbnails):
            screen.blit(thumb, buttons[idx].topleft)
            level_label = font.render("Level 1", True, (0, 0, 0))
            screen.blit(
                level_label, (buttons[idx].x, buttons[idx].y + TILE_SIZE + 2))

        # Draw "?" button
        pygame.draw.rect(screen, (0, 0, 0), question_rect, border_radius=10)
        qmark = big_font.render("?", True, (255, 255, 255))
        screen.blit(qmark, (question_rect.x + 20, question_rect.y + 10))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if question_rect.collidepoint(pos):
                    print("🎲 Random board")
                    solve_board(screen, model, random.choice(all_boards))
                for i, rect in enumerate(buttons):
                    if rect.collidepoint(pos):
                        print(f"▶️ Solving level {i+1}")
                        solve_board(screen, model, sample_boards[i])

    pygame.quit()


if __name__ == "__main__":
    main()
