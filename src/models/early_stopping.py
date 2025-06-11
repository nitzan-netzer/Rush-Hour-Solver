from stable_baselines3.common.callbacks import BaseCallback
from collections import deque


class EarlyStoppingRewardCallback(BaseCallback):
    def __init__(self, window_size=100, reward_threshold=900, verbose=1):
        super().__init__(verbose)
        self.window_size = window_size
        self.reward_threshold = reward_threshold
        self.reward_history = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]
        reward = info.get("total_reward")
        if done:
            self.reward_history.append(reward)

            if len(self.reward_history) == self.window_size:
                avg_reward = sum(self.reward_history) / self.window_size
                if self.verbose:
                    print(
                        f"💰 Average reward (last {self.window_size} episodes): {avg_reward:.2f}")
                if avg_reward >= self.reward_threshold:
                    print("🛑 Early stopping: reward threshold reached!")
                    return False  # Stop training

        return True