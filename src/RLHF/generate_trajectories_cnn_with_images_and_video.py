import setup_path  # NOQA
import os
import json
from pathlib import Path
from copy import deepcopy
import imageio
import random
from stable_baselines3 import PPO
from environments.rush_hour_env import RushHourEnv
from environments.board import Board
from GUI.board_to_image import generate_board_image
from utils.config import NUM_VEHICLES


class TrajectoryGenerator:
    def __init__(self,
                 model_path: Path,
                 output_dir: Path,
                 num_boards: int = 20,
                 scale: int = 52,
                 max_steps: int = 50,
                 max_invalid_moves: int = 20,
                 save_video: bool = True):
        self.model_path = model_path
        self.output_dir = output_dir
        self.num_boards = num_boards
        self.scale = scale
        self.max_steps = max_steps
        self.max_invalid_moves = max_invalid_moves
        self.save_video = save_video

        self.env = RushHourEnv(num_of_vehicle=NUM_VEHICLES, train=False)
        self.model = self.load_model()
        all_boards = self.env.boards.copy()
        random.shuffle(all_boards)

        if len(all_boards) < self.num_boards:
            raise ValueError(
                f"Not enough boards available. Requested {self.num_boards} but only have {len(all_boards)}")
        self.boards = all_boards[:self.num_boards]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        print(f"✅ Loaded model from {self.model_path}")
        return PPO.load(self.model_path)

    def run(self):
        for board_idx, board in enumerate(self.boards):
            print(f"\n=== Generating Trajectories for Board {board_idx} ===")
            board_copy = deepcopy(board)

            self.run_episode(board_idx, 0, board)
            self.run_episode(board_idx, 1, board_copy)

    def run_episode(self, board_idx: int, run_idx: int, board: Board):
        obs, _ = self.env.reset(board=board)
        state = deepcopy(obs)
        step_data = []
        all_moves = []
        num_steps = 0
        consecutive_invalid_moves = 0
        initial_board = deepcopy(self.env.board)
        last_info = {"red_car_escaped": False}
        reward = -10
        done, truncated = False, False

        # ✅ Set different random seeds for stochastic behavior
        seed_offset = 1000 if run_idx == 0 else 2000
        self.model.set_random_seed(board_idx + seed_offset)

        while not done and not truncated and num_steps < self.max_steps:
            if consecutive_invalid_moves >= self.max_invalid_moves:
                print("⚠️ Too many invalid moves. Aborting run.")
                break

            # ✅ Always sample stochastically for both agents
            action, _ = self.model.predict(state, deterministic=False)
            vehicle_idx = action // 4
            move_idx = action % 4
            move_str = ["U", "D", "L", "R"][move_idx]
            vehicle_letter = None
            valid_move = False
            info = {"red_car_escaped": False}

            if vehicle_idx >= len(self.env.vehicles_letter):
                reward = -10
                next_state = state
                consecutive_invalid_moves += 1
            else:
                vehicle_letter = self.env.vehicles_letter[vehicle_idx]
                next_state, reward, done, truncated, info = self.env.step(
                    action)
                valid_move = reward != -1
                consecutive_invalid_moves = 0 if valid_move else consecutive_invalid_moves + 1

            all_moves.append((vehicle_letter, move_str, valid_move))

            step_data.append({
                "step_num": num_steps,
                "state": state.tolist(),
                "action": int(action),
                "vehicle": vehicle_letter,
                "move": move_str,
                "valid": valid_move,
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated)
            })

            state = next_state
            last_info = info
            num_steps += 1

        red_car_escaped = last_info.get("red_car_escaped", False)
        print(
            f"Agent {run_idx + 1} {'✅ escaped' if red_car_escaped else '❌ did not escape'} on board {board_idx}")

        board_hash = initial_board.get_hash()
        self.save_trajectory_json(
            board_idx, run_idx, step_data, board_hash, num_steps, red_car_escaped)

        if self.save_video:
            self.save_video_mp4(board_idx, run_idx, all_moves, initial_board)

        return board_hash

    def save_trajectory_json(self, board_idx, run_idx, step_data, board_hash, total_steps, red_car_escaped):
        data = {
            "board_index": board_idx,
            "run_index": run_idx,
            "steps": step_data,
            "red_car_escaped": red_car_escaped,
            "completed": red_car_escaped,
            "board_hash": board_hash,
            "total_steps_attempted": total_steps
        }
        path = self.output_dir / \
            f"trajectory_{board_idx}_agent{run_idx + 1}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"📄 Trajectory saved: {path}")

    def save_video_mp4(self, board_idx, run_idx, all_moves, initial_board):
        frames = [generate_board_image(
            initial_board, scale=self.scale, draw_letters=False)]
        board = deepcopy(initial_board)

        for letter, direction, valid in all_moves:
            if not letter:
                continue
            vehicle = board.get_vehicle_by_letter(letter)
            if not vehicle:
                continue
            board.move_vehicle(vehicle, direction)
            frames.append(generate_board_image(
                board, scale=self.scale, draw_letters=False))

        video_path = self.output_dir / \
            f"trajectory_{board_idx}_agent{run_idx + 1}_video.mp4"
        imageio.mimsave(video_path, [f.convert("RGB") for f in frames], fps=4)
        print(f"🎥 Video saved: {video_path}")


if __name__ == "__main__":
    generator = TrajectoryGenerator(
        model_path=Path(
            "models_zip/PPO_MLP_full_run_1748637856.zip"),
        output_dir=Path("database/trajectories_mlp_policy"),
        num_boards=200,
        save_video=True
    )
    generator.run()
