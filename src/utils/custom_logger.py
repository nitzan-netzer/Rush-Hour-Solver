import csv
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path


class RushHourCSVLogger(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_rewards = []
        self.episode_count = 0

        # Initialize the CSV file with headers
        with open(self.log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["episode", "timesteps", "reward", "red_car_escaped"])

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self.episode_rewards.append(reward)

        if done:
            total_reward = sum(self.episode_rewards)
            info = self.locals.get("infos", [{}])[0]
            escaped = info.get("red_car_escaped", False)

            # Log to CSV
            try:
                with open(self.log_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [self.episode_count, self.num_timesteps, total_reward, int(escaped)])
            except Exception as e:
                print(f"❌ Failed to write log: {e}")

            print(
                f"[Episode {self.episode_count}] Reward: {total_reward:.2f} | Escaped: {escaped}")

            self.episode_count += 1
            self.episode_rewards = []

        return True
