import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class SmallBoardCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        # 16 channels (vehicle letters), 6x6 board
        super().__init__(observation_space, features_dim=128)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=observation_space.shape[0], out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # compute the output size after flattening
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
