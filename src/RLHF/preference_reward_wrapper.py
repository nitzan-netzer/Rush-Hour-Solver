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
        self.num_steps = 0
        return obs, info

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        self.trajectory.append(obs.copy())
        self.num_steps += 1

        if done or truncated:
            traj_array = np.array(self.trajectory, dtype=np.float32)
            steps_tensor = np.array(
                [[len(self.trajectory) / 2000]], dtype=np.float32)
            last_state_tensor = traj_array[-1].reshape(1, -1)

            # Concatenate last state and normalized step count
            model_input = np.concatenate(
                [last_state_tensor, steps_tensor], axis=1)
            model_score = self.reward_model(model_input).numpy().item()

            if info.get("red_car_escaped", False):
                # Base reward from preference model scaled and offset
                reward = model_score * 3.0 + 10
                # Strong penalty for longer solutions
                reward -= self.num_steps * 0.3
                reward = np.clip(reward, -50, 100)
            else:
                reward = -10  # penalty for not escaping
        else:
            reward = self._shaped_reward(obs)

        return obs, reward, done, truncated, info

    def _shaped_reward(self, obs):
        """
        Dense shaping: reward progress toward goal based on red car path clearance.
        Encourages the agent to unblock the red car's row early.
        """
        board = np.array(obs).reshape(6, 6)
        red_row = board[2]
        blocked = sum(1 for i in range(5, -1, -1) if red_row[i] != 0)
        # The more the path is clear, the higher the reward (max: +0.5)
        return -1.0 + 0.1 * (6 - blocked)
