"""PyTorch actor and twin critics for discrete SAC."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from algorithm.common.featureExtractor import DualResolutionFeatureExtractor


class SACNetwork(nn.Module):
    def __init__(self, history_frames: int, action_count: int) -> None:
        super().__init__()
        self.history_frames = history_frames
        self.action_count = action_count
        self.actor_extractor = DualResolutionFeatureExtractor(history_frames)
        self.critic_extractor = DualResolutionFeatureExtractor(history_frames)
        self.actor_head = nn.Sequential(
            nn.Linear(self.actor_extractor.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_count),
        )
        self.q1_head = nn.Sequential(
            nn.Linear(self.critic_extractor.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_count),
        )
        self.q2_head = nn.Sequential(
            nn.Linear(self.critic_extractor.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_count),
        )

    def actor_logits(self, observation: dict[str, Tensor]) -> Tensor:
        logits = self.actor_head(self.actor_extractor(observation))
        return logits.masked_fill(~observation["action_mask"], torch.finfo(logits.dtype).min)

    def critics(self, observation: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        features = self.critic_extractor(observation)
        return self.q1_head(features), self.q2_head(features)

    def actor_parameters(self):
        return list(self.actor_extractor.parameters()) + list(self.actor_head.parameters())

    def critic_parameters(self):
        return (
            list(self.critic_extractor.parameters())
            + list(self.q1_head.parameters())
            + list(self.q2_head.parameters())
        )


__all__ = ["SACNetwork"]
