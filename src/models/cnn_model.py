import setup_path  # NOQA

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.utils import set_random_seed
from cnn_policy import RushHourCNN
from environments.rush_hour_image_env import RushHourImageEnv
from environments.board import Board
from gymnasium.wrappers import NormalizeObservation
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.env_util import make_vec_env

import torch as th

# 1. Load boards
boards = Board.load_multiple_boards("database/1000_cards_3_cars_1_trucks.json")

# 2. Define Environment


def make_env(seed=0):
    env = RushHourImageEnv(boards, num_vehicles=6)
    env.reset()
    env.action_space.seed(seed)
    return env


# Vectorized env for stability
vec_env = make_vec_env(make_env, n_envs=4)

# 3. Define Custom Policy
policy_kwargs = dict(
    features_extractor_class=RushHourCNN,
    features_extractor_kwargs=dict(features_dim=256)
)

# 4. Initialize PPO Model
model = PPO(
    policy=ActorCriticCnnPolicy,
    env=vec_env,
    learning_rate=2.5e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="auto"
)

# 5. Train
model.learn(total_timesteps=100_000)
model.save("ppo_rush_hour_cnn")

# 6. Test
obs = vec_env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    obs, rewards, dones, infos = vec_env.step(action)
    vec_env.render("human")
