import setup_path  # NOQA
import pygame
import time
import numpy as np
import cv2
from utils.config import MODEL_PATH
from environments.rush_hour_env import RushHourEnv
from environments.rush_hour_image_env import RushHourImageEnv
from stable_baselines3 import PPO
from GUI.board_to_image import letter_to_color

# === Settings ===
TILE_SIZE = 80
BOARD_SIZE = 6
WINDOW_SIZE = TILE_SIZE * BOARD_SIZE
TEXT_COLOR = (0, 0, 0)
COLORS = letter_to_color


def draw_board(screen, board, font):
    """Draw classic RushHourEnv board using Pygame."""
    screen.fill((240, 240, 240))
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            value = board.board[i, j]
            color = COLORS.get(value, (0, 0, 0))
            pygame.draw.rect(screen, color, pygame.Rect(
                j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(
                j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)
            if value:
                text = font.render(value, True, TEXT_COLOR)
                text_rect = text.get_rect(center=(j * TILE_SIZE + TILE_SIZE // 2,
                                                  i * TILE_SIZE + TILE_SIZE // 2))
                screen.blit(text, text_rect)
    pygame.display.flip()


def draw_image_obs(obs):
    """Draw normalized image-based observation (RushHourImageEnv)."""
    img = (obs * 255).astype(np.uint8)
    img_resized = cv2.resize(img, (WINDOW_SIZE, WINDOW_SIZE), interpolation=cv2.INTER_NEAREST)
    return img_resized


import os

def run_visualizer(model_path, record=False, output_video="videos/rush_hour_solution.mp4", cnn=None):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Rush Hour - Agent Demo")
    font = pygame.font.SysFont(None, 36)

    # === Auto-detect CNN model from file name if not passed ===
    if cnn is None:
        cnn = "CNN" in os.path.basename(model_path)

    # === Load environment accordingly ===
    if cnn:
        env = RushHourImageEnv(num_of_vehicle=6, train=False)
    else:
        env = RushHourEnv(num_of_vehicle=6, train=False)

    model = PPO.load(model_path, env=env)  # Will now match the env type
    obs, _ = env.reset()

    if not cnn:
        draw_board(screen, env.board, font)
    else:
        frame = draw_image_obs(obs)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame.surfarray.blit_array(screen, np.transpose(frame_rgb, (1, 0, 2)))
        pygame.display.flip()

    time.sleep(1)

    # === Setup video recording ===
    out = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, 3, (WINDOW_SIZE, WINDOW_SIZE))
        surface = pygame.display.get_surface()
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    for i in range(50):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                if out:
                    out.release()
                return

        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        print(f"Step {i}: reward: {reward}")

        if not cnn:
            draw_board(screen, env.board, font)
        else:
            frame = draw_image_obs(obs)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pygame.surfarray.blit_array(screen, np.transpose(frame_rgb, (1, 0, 2)))
            pygame.display.flip()

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


def main():
    run_visualizer(model_path=MODEL_PATH, record=True, cnn=True)  # Set cnn=False for MLP model


if __name__ == "__main__":
    main()
