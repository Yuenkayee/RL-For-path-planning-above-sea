"""PyTorch dual-resolution dueling Q-network."""

from __future__ import annotations

from torch import Tensor, nn

from algorithm.common.featureExtractor import DualResolutionFeatureExtractor


class DuelingQNetwork(nn.Module):
    def __init__(self, history_frames: int, action_count: int) -> None:
        super().__init__()
        self.history_frames = history_frames
        self.action_count = action_count
        self.extractor = DualResolutionFeatureExtractor(history_frames)
        self.shared = nn.Sequential(nn.Linear(self.extractor.output_dim, 256), nn.ReLU())
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, action_count)

    def forward(self, observation: dict[str, Tensor]) -> Tensor:
        features = self.shared(self.extractor(observation))
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


__all__ = ["DuelingQNetwork"]
