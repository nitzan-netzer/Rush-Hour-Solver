from stable_baselines3.common.callbacks import BaseCallback
from collections import deque


class EarlyStoppingSuccessRateCallback(BaseCallback):
    def __init__(self, window_size=100, success_threshold=0.9, patience=5, verbose=1):
        super().__init__(verbose)
        self.window_size = window_size
        self.success_threshold = success_threshold
        self.patience = patience
        self.patience_counter = 0
        self.success_history = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        if done:
            escaped = info.get("red_car_escaped", False)
            self.success_history.append(1 if escaped else 0)

            # Check if we have enough data to calculate success rate
            if len(self.success_history) == self.window_size:
                success_rate = sum(self.success_history) / self.window_size

                if self.verbose:
                    print(
                        f"✅ Success rate (last {self.window_size} episodes): {success_rate:.2f}")

                # Check if success rate meets threshold
                if success_rate >= self.success_threshold:
                    self.patience_counter += 1
                    print(
                        f"⏳ Patience: {self.patience_counter}/{self.patience}")

                    # Stop only if patience limit is reached
                    if self.patience_counter >= self.patience:
                        print(
                            "🛑 Early stopping: success rate threshold reached consistently!")
                        return False  # Stop training
                else:
                    # Reset patience if success rate drops below threshold
                    self.patience_counter = 0

        return True
