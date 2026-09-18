"""Masked categorical wait-or-heading PPO policy."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from algorithm.common.mathUtils import argmax_masked, sample_categorical

from .ppoNetwork import PPONetwork


class MaskedCategoricalPolicy:
    def __init__(self, network: PPONetwork, *, seed: int | None = None) -> None:
        self.network = network
        self.rng = random.Random(seed)

    def select(
        self,
        features: Sequence[float],
        action_mask: Sequence[bool],
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        probabilities = self.network.probabilities(features, action_mask)
        action = (
            argmax_masked(probabilities, action_mask)
            if deterministic
            else sample_categorical(probabilities, self.rng)
        )
        return action, math.log(max(probabilities[action], 1e-12)), self.network.value(features)


__all__ = ["MaskedCategoricalPolicy"]
