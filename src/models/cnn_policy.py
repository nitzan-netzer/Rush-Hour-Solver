import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RushHourCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]  # 3 for RGB

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8,
                      stride=4, padding=0),  # -> [32, 74, 74]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2,
                      padding=0),  # -> [64, 36, 36]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,
                      padding=0),  # -> [64, 34, 34]
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute CNN output shape dynamically
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[
                                        None]).permute(0, 3, 1, 2).float()
            cnn_output_dim = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Convert (B, H, W, C) -> (B, C, H, W)
        x = observations.permute(0, 3, 1, 2)
        x = self.cnn(x)
        return self.linear(x)
