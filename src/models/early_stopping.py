from stable_baselines3.common.callbacks import BaseCallback
from collections import deque


class EarlyStoppingSuccessRateCallback(BaseCallback):
    def __init__(self, window_size=100, success_threshold=0.9, verbose=1):
        super().__init__(verbose)
        self.reward_threshold = 900 # fix reward threshold
        self.window_size = 50
        #self.success_threshold = success_threshold
        self.reward_history = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        if done:
            reward = info.get('total_reward')
            self.reward_history.append(reward)
            if len(self.reward_history) == self.window_size:
                reward_rate = sum(self.reward_history) / self.window_size
                if reward_rate >= self.reward_threshold:
                    print("🛑 Early stopping: reward rate threshold reached!")
                    return False  # Stop training
        return True
