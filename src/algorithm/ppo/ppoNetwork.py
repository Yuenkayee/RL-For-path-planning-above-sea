"""PyTorch dual-resolution actor-critic network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from algorithm.common.featureExtractor import DualResolutionFeatureExtractor


class PPONetwork(nn.Module):
    def __init__(self, history_frames: int, action_count: int) -> None:
        super().__init__()
        self.history_frames = history_frames
        self.action_count = action_count
        self.extractor = DualResolutionFeatureExtractor(history_frames)
        self.actor = nn.Sequential(
            nn.Linear(self.extractor.output_dim, 256), nn.ReLU(), nn.Linear(256, action_count)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.extractor.output_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        features = self.extractor(observation)
        logits = self.actor(features)
        masked_logits = logits.masked_fill(
            ~observation["action_mask"], torch.finfo(logits.dtype).min
        )
        return masked_logits, self.critic(features).squeeze(-1)


__all__ = ["PPONetwork"]
