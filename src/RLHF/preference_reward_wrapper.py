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
        return obs, info

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        self.trajectory.append(obs.copy())

        if done or truncated:
            traj_array = np.array(self.trajectory, dtype=np.float32)
            steps_tensor = np.array(
                [[len(self.trajectory) / 2000]], dtype=np.float32)
            last_state_tensor = traj_array[-1].reshape(1, -1)

            # Prepare input as [final_state + normalized_length]
            model_input = np.concatenate(
                [last_state_tensor, steps_tensor], axis=1)
            model_score = self.reward_model(model_input).numpy().item()

            if info.get("red_car_escaped", False):
                reward = model_score + 5
            else:
                reward = -10.0  # punishment for not solving
        else:
            reward = 0.0

        return obs, reward, done, truncated, info
