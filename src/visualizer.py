# Visualize agent while training

import pygame
import time
from rush_hour_env import RushHourEnv
from stable_baselines3 import PPO

# Settings
TILE_SIZE = 80
BOARD_SIZE = 6
WINDOW_SIZE = TILE_SIZE * BOARD_SIZE

COLORS = {
    "": (220, 220, 220),
    "X": (255, 0, 0),
    "A": (0, 128, 255),
    "B": (0, 200, 100),
    "O": (160, 0, 255),
}


def draw_board(screen, board):
    screen.fill((240, 240, 240))
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            value = board.board[i, j]
            color = COLORS.get(value, (0, 0, 0))
            pygame.draw.rect(screen, color, pygame.Rect(
                j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(
                j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)
    pygame.display.flip()


def run_visualizer():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Rush Hour - Agent Demo")

    env = RushHourEnv(num_of_vehicle=4)
    model = PPO.load("ppo_rush_hour_model", env=env)

    obs, _ = env.reset()
    draw_board(screen, env.board)
    time.sleep(1)

    for i in range(100):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        draw_board(screen, env.board)
        time.sleep(0.3)
        if done:
            print("✅ Escaped!")
            time.sleep(1.5)
            break
        vehicle_str, move_str = env.parse_action(action)
        print(f"Step {i}: {vehicle_str} → {move_str}, reward: {reward}")
    pygame.quit()

    solved = 0
    for i in range(50):
        obs, _ = env.reset()
        for step in range(10):  # 3 moves + retries
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            if done:
                solved += 1
                break
    print(f"✅ Solved {solved}/50 boards")


if __name__ == "__main__":
    run_visualizer()
