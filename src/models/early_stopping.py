from stable_baselines3.common.callbacks import BaseCallback
from collections import deque


class EarlyStoppingSuccessRateCallback(BaseCallback):
    def __init__(self, window_size=100, success_threshold=0.9, verbose=1):
        super().__init__(verbose)
        self.window_size = window_size
        self.success_threshold = success_threshold
        self.success_history = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        done = self.locals["dones"][0]
        info = self.locals.get("infos", [{}])[0]

        if done:
            escaped = info.get("red_car_escaped", False)
            self.success_history.append(1 if escaped else 0)

            if len(self.success_history) == self.window_size:
                success_rate = sum(self.success_history) / self.window_size
                print(
                    f"✅ Success rate (last {self.window_size} episodes): {success_rate:.2f}")
                if success_rate >= self.success_threshold:
                    print("🛑 Early stopping triggered.")
                    return False  # Stop training

        return True
