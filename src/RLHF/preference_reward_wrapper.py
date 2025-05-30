import numpy as np
from gymnasium import Wrapper
import tensorflow as tf


class PreferenceRewardWrapper(Wrapper):
    def __init__(self, env, reward_model_path):
        super().__init__(env)
        self.reward_model = tf.keras.models.load_model(reward_model_path)
        self.trajectory = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.trajectory = [obs.copy()]
        self.num_steps = 0  # ✅ initialize step counter
        return obs, info

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        self.trajectory.append(obs.copy())
        self.num_steps += 1  # ✅ count steps

        if done or truncated:
            traj_array = np.array(self.trajectory, dtype=np.float32)
            steps_tensor = np.array(
                [[len(self.trajectory) / 2000]], dtype=np.float32)
            last_state_tensor = traj_array[-1].reshape(1, -1)

            model_input = np.concatenate(
                [last_state_tensor, steps_tensor], axis=1)
            model_score = self.reward_model(model_input).numpy().item()

            if info.get("red_car_escaped", False):
                reward = np.clip(model_score * 2.5 + 5, -20, 100)
                reward -= self.num_steps * 0.2  # ✅ apply penalty based on actual steps
            else:
                reward = -5
        else:
            reward = self._shaped_reward(obs)

        return obs, reward, done, truncated, info

    def _shaped_reward(self, obs):
        board = np.array(obs).reshape(6, 6)
        red_row = board[2]
        clear_path = sum(1 for i in range(5, 6) if red_row[i] == 0)
        return -0.1 + 0.1 * clear_path
