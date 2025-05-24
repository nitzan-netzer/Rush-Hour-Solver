# Modified `generate_trajectories_cnn_with_images_and_video.py` to align with project structure

import setup_path  # NOQA
import os
import json
from stable_baselines3 import PPO
from environments.rush_hour_env import RushHourEnv
from GUI.board_to_image import save_board_to_video, generate_board_image
from copy import deepcopy
from pathlib import Path
import imageio

# =========== CONFIGURATION ===========
MODEL_PATH = Path("models_zip/PPO_MLP_full_run_1748012332.zip")
TRAJECTORIES_DIR = Path("database/trajectories_mlp_policy")
NUM_BOARDS = 20
SAVE_VIDEO = True
SCALE = 52  # Ensures 6*scale is divisible by 16 for video compatibility

TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

# =========== LOAD ENVIRONMENT & MODEL ===========
env = RushHourEnv(num_of_vehicle=6, train=True)
model = PPO.load(MODEL_PATH)

# =========== MAIN CODE ===========
for board_idx, board in enumerate(env.boards[:NUM_BOARDS]):
    print(f"Generating trajectories for board {board_idx}")
    for run_idx in range(2):
        obs, _ = env.reset(board=board)
        state = deepcopy(obs)
        done = False
        truncated = False
        step_data = []
        video_moves = []
        env_board = deepcopy(env.board)
        num_steps = 0

        while not done and not truncated:
            action, _ = model.predict(state, deterministic=(run_idx == 0))
            next_state, reward, done, truncated, info = env.step(action)

            vehicle_idx = action // 4
            move_idx = action % 4
            move_str = ["U", "D", "L", "R"][move_idx]
            vehicle_letter = env.vehicles_letter[vehicle_idx]
            video_moves.append((vehicle_letter, move_str, "1"))

            step_data.append({
                "step_num": num_steps,
                "state": state.tolist(),
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated)
            })

            state = next_state
            num_steps += 1

        red_car_escaped = info.get("red_car_escaped", False)
        print(
            f"\U0001F3C1 Board {board_idx}, Run {run_idx}: Red car escaped? {'YES' if red_car_escaped else 'NO'}")
        print(f"Steps taken: {num_steps}")

        json_out = {
            "board_index": board_idx,
            "run_index": run_idx,
            "steps": step_data,
            "red_car_escaped": red_car_escaped
        }

        json_filename = TRAJECTORIES_DIR / \
            f"trajectory_{board_idx}_agent{run_idx + 1}.json"
        with open(json_filename, "w") as f:
            json.dump(json_out, f, indent=2)

        if SAVE_VIDEO:
            video_filename = TRAJECTORIES_DIR / \
                f"trajectory_{board_idx}_agent{run_idx + 1}_video.mp4"
            # Inline video generation with scale control to avoid touching the main image module
            frames = []
            img = generate_board_image(
                env_board, scale=SCALE, draw_letters=False)
            frames.append(img.copy())
            for move in video_moves:
                letter, direction, times = move
                vehicle = env_board.get_vehicle_by_letter(letter)
                if vehicle is None:
                    print(
                        f"⚠️ Warning: Vehicle '{letter}' not found on board. Skipping move.")
                    continue
                for _ in range(int(times)):
                    env_board.move_vehicle(vehicle, direction)
                    img = generate_board_image(
                        env_board, scale=SCALE, draw_letters=False)
                    frames.append(img.copy())
            imageio.mimsave(video_filename, [frame.convert(
                "RGB") for frame in frames], fps=4)
            print(f"Video saved as {video_filename}")
