"""Masked categorical wait-or-heading PPO policy."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from .ppoNetwork import PPONetwork


class MaskedCategoricalPolicy:
    def __init__(self, network: PPONetwork) -> None:
        self.network = network

    def select(
        self,
        observation: dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        logits, values = self.network(observation)
        distribution = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return (
            int(actions[0].item()),
            float(distribution.log_prob(actions)[0].item()),
            float(values[0].item()),
        )


__all__ = ["MaskedCategoricalPolicy"]
