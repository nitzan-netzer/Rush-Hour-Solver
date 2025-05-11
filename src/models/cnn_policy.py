import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RushHourCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]  # Should be 3 (RGB)
        # print(f"[DEBUG] Input channels: {n_input_channels}")
        # print(f"[DEBUG] Observation space shape: {observation_space.shape}")

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5,
                      stride=2, padding=1),  # → [32, H, W]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1,
                      padding=1),                # → [64, H, W]
            nn.ReLU(),
            # → H/2
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=2,
                      padding=1),                # → [64, H/2, W/2]
            nn.ReLU(),
            # → [64, 2, 2]
            nn.AdaptiveAvgPool2d((2, 2)),
            # → 64 × 2 × 2 = 256
            nn.Flatten()
        )

        # Compute output shape
        with th.no_grad():
            sample_input = th.as_tensor(
                observation_space.sample()[None])  # [1, H, W, C]
            sample_input = sample_input.permute(
                0, 3, 1, 2).float()         # [1, C, H, W]
            cnn_output_dim = self.cnn(sample_input).shape[1]
            # print(f"[DEBUG] CNN output dim: {cnn_output_dim}")

        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = observations.permute(0, 3, 1, 2).float()  # NHWC → NCHW
        # print(f"[DEBUG] Forward input shape: {x.shape}")
        x = self.cnn(x)
        # print(f"[DEBUG] CNN output shape: {x.shape}")
        x = self.linear(x)
        # print(f"[DEBUG] Linear output shape: {x.shape}")
        return x
