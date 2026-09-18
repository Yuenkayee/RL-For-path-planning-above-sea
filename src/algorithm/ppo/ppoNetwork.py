"""Linear actor-critic network with a future CNN-compatible interface."""

from __future__ import annotations

from collections.abc import Sequence
import random

from algorithm.common.mathUtils import dot, masked_softmax


class PPONetwork:
    def __init__(self, feature_count: int, action_count: int, *, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self.feature_count = feature_count
        self.action_count = action_count
        self.actor_weights = [
            [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
            for _ in range(action_count)
        ]
        self.value_weights = [0.0] * feature_count

    def logits(self, features: Sequence[float]) -> tuple[float, ...]:
        return tuple(dot(weights, features) for weights in self.actor_weights)

    def probabilities(
        self, features: Sequence[float], action_mask: Sequence[bool]
    ) -> tuple[float, ...]:
        return masked_softmax(self.logits(features), action_mask)

    def value(self, features: Sequence[float]) -> float:
        return dot(self.value_weights, features)

    def state_dict(self) -> dict:
        return {
            "feature_count": self.feature_count,
            "action_count": self.action_count,
            "actor_weights": self.actor_weights,
            "value_weights": self.value_weights,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["feature_count"] != self.feature_count or state["action_count"] != self.action_count:
            raise ValueError("checkpoint network dimensions do not match")
        self.actor_weights = [list(map(float, row)) for row in state["actor_weights"]]
        self.value_weights = list(map(float, state["value_weights"]))


__all__ = ["PPONetwork"]
