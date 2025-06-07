import setup_path  # NOQA
import pygame
import time
import numpy as np
import cv2
from GUI.board_to_image import letter_to_color
from utils.config import BOARD_SIZE

# === Settings ===
TILE_SIZE = 80
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
    img_resized = cv2.resize(
        img, (WINDOW_SIZE, WINDOW_SIZE), interpolation=cv2.INTER_NEAREST)
    return img_resized


def run_visualizer(model, env, record=False, output_video="videos/rush_hour_solution.mp4"):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Rush Hour - Agent Demo")
    font = pygame.font.SysFont(None, 36)

    obs, info = env.reset()

    # Draw initial state
    if hasattr(env, "board"):  # Classic vector env
        draw_board(screen, env.board, font)
    else:  # Image-based env
        frame = draw_image_obs(obs)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pygame.surfarray.blit_array(screen, np.transpose(frame_rgb, (1, 0, 2)))
        pygame.display.flip()

    time.sleep(1)

    # === Setup video recording ===
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
    
    done = False
    truncated = False
    while not done and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                if out:
                    out.release()
                return

        action_mask = info.get("action_mask")
        action, _ = model.predict(obs, action_masks=action_mask)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {env.num_steps}: reward: {reward}")

        if hasattr(env, "board"):
            draw_board(screen, env.board, font)
        else:
            frame = draw_image_obs(obs)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pygame.surfarray.blit_array(
                screen, np.transpose(frame_rgb, (1, 0, 2)))
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
if __name__ == "__main__":
    from utils.config import NUM_VEHICLES
    from environments.rush_hour_env import RushHourEnv
    from environments.rewards import basic_reward
    from sb3_contrib.ppo_mask import MaskablePPO as PPO
    model_path = r"models_zip\MaskablePPO_MLP_8x8"
    env = RushHourEnv(NUM_VEHICLES, train=False, rewards=basic_reward)
    model=  PPO.load(model_path, env=env)

    run_visualizer(model,env,record=True, output_video="videos/rush_hour_solution.mp4")