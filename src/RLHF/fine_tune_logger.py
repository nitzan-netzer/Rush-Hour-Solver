import csv
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path


class FineTuneLogger(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_data = []
        self.episode_count = 0

        # Create CSV file with headers
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "total_timesteps", "reward", "num_steps",
                "red_car_escaped", "model_score", "done_reason"
            ])

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        self.episode_data.append(reward)

        if done:
            total_reward = sum(self.episode_data)
            red_car_escaped = int(info.get("red_car_escaped", False))
            model_score = info.get("model_score", None)
            num_steps = info.get("num_steps", len(self.episode_data))

            # ✅ Unwrap to access base env and its max_steps
            base_env = self.training_env.envs[0]
            while hasattr(base_env, "env"):
                base_env = base_env.env

            if red_car_escaped:
                done_reason = "win"
            elif num_steps >= base_env.max_steps:
                done_reason = "timeout"
            else:
                done_reason = "truncated"

            # Log to CSV
            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_count,
                    self.num_timesteps,
                    total_reward,
                    num_steps,
                    red_car_escaped,
                    model_score if model_score is not None else "NA",
                    done_reason
                ])

            if self.verbose:
                print(
                    f"[Episode {self.episode_count}] Reward: {total_reward:.2f} | "
                    f"Steps: {num_steps} | Escaped: {red_car_escaped} | "
                    f"Model Score: {model_score if model_score is not None else 'NA'}"
                )

            self.episode_count += 1
            self.episode_data = []

        return True
