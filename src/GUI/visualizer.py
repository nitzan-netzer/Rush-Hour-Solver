import setup_path # NOQA
import pygame
import time
import numpy as np
import cv2

from environments.rush_hour_env import RushHourEnv
from environments.evaluate import evaluate_model
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

TEXT_COLOR = (0, 0, 0)


def draw_board(screen, board, font):
    screen.fill((240, 240, 240))
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            value = board.board[i, j]
            color = COLORS.get(value, (0, 0, 0))
            pygame.draw.rect(
                screen, color, pygame.Rect(
                    j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            )
            pygame.draw.rect(
                screen, (0, 0, 0), pygame.Rect(j * TILE_SIZE,
                                               i * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1
            )
            if value:
                text = font.render(value, True, TEXT_COLOR)
                text_rect = text.get_rect(
                    center=(j * TILE_SIZE + TILE_SIZE // 2, i * TILE_SIZE + TILE_SIZE // 2))
                screen.blit(text, text_rect)
    pygame.display.flip()


def run_visualizer(model_path,record=False, output_video=r"videos\rush_hour_solution.mp4"):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Rush Hour - Agent Demo")
    font = pygame.font.SysFont(None, 36)

    test_env = RushHourEnv(num_of_vehicle=4, train=False)
    model = PPO.load(model_path, env=test_env)

    obs, _ = test_env.reset()
    draw_board(screen, test_env.board, font)
    time.sleep(1)

    out = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, 3,
                              (WINDOW_SIZE, WINDOW_SIZE))
        surface = pygame.display.get_surface()
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    for i in range(100):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                if out:
                    out.release()
                return

        action, _ = model.predict(obs)
        obs, reward, done, _, _ = test_env.step(action)

        vehicle_str, move_str = test_env.parse_action(action)
        print(f"Step {i}: {vehicle_str} → {move_str}, reward: {reward}")

        draw_board(screen, test_env.board, font)
        if record:
            surface = pygame.display.get_surface()
            frame = pygame.surfarray.array3d(surface)
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        time.sleep(0.2)

        if done:
            print("✅ Escaped!")
            text = font.render("✅ Escaped!", True, (0, 200, 0))
            screen.blit(text, (50, 50))
            pygame.display.flip()
            time.sleep(2)
            break

    pygame.quit()

    if out:
        out.release()
        print(f"✅ Video saved to {output_video}")

    # === Automated test loop ===
    evaluate_model(model,test_env)

if __name__ == "__main__":
    run_visualizer(model_path="models_zip/ppo_rush_hour.zip",record=True)
