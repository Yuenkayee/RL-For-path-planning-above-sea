"""Linear actor and twin critics for dependency-free discrete SAC."""

from __future__ import annotations

from collections.abc import Sequence
import random

from algorithm.common.mathUtils import dot, masked_softmax


class SACNetwork:
    def __init__(self, feature_count: int, action_count: int, *, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self.feature_count = feature_count
        self.action_count = action_count
        self.actor_weights = [
            [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
            for _ in range(action_count)
        ]
        self.q1_weights = [[0.0] * feature_count for _ in range(action_count)]
        self.q2_weights = [[0.0] * feature_count for _ in range(action_count)]

    def probabilities(self, features: Sequence[float], mask: Sequence[bool]) -> tuple[float, ...]:
        logits = [dot(row, features) for row in self.actor_weights]
        return masked_softmax(logits, mask)

    def q_values(self, features: Sequence[float], critic: int) -> tuple[float, ...]:
        weights = self.q1_weights if critic == 1 else self.q2_weights
        return tuple(dot(row, features) for row in weights)

    def state_dict(self) -> dict:
        return {
            "feature_count": self.feature_count,
            "action_count": self.action_count,
            "actor_weights": self.actor_weights,
            "q1_weights": self.q1_weights,
            "q2_weights": self.q2_weights,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["feature_count"] != self.feature_count or state["action_count"] != self.action_count:
            raise ValueError("checkpoint network dimensions do not match")
        for name in ("actor_weights", "q1_weights", "q2_weights"):
            setattr(self, name, [list(map(float, row)) for row in state[name]])


__all__ = ["SACNetwork"]
