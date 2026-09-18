"""Linear dueling Q-network for one wait and discrete heading actions."""

from __future__ import annotations

from collections.abc import Sequence
import random

from algorithm.common.mathUtils import dot


class DuelingQNetwork:
    def __init__(self, feature_count: int, action_count: int, *, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self.feature_count = feature_count
        self.action_count = action_count
        self.value_weights = [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
        self.advantage_weights = [
            [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
            for _ in range(action_count)
        ]

    def q_values(self, features: Sequence[float]) -> tuple[float, ...]:
        value = dot(self.value_weights, features)
        advantages = [dot(weights, features) for weights in self.advantage_weights]
        mean = sum(advantages) / len(advantages)
        return tuple(value + advantage - mean for advantage in advantages)

    def update(self, features: Sequence[float], action: int, target: float, learning_rate: float) -> float:
        prediction = self.q_values(features)[action]
        error = target - prediction
        for index, feature in enumerate(features):
            self.value_weights[index] += learning_rate * error * feature
        for action_index, weights in enumerate(self.advantage_weights):
            coefficient = (1.0 if action_index == action else 0.0) - 1.0 / self.action_count
            for index, feature in enumerate(features):
                weights[index] += learning_rate * error * coefficient * feature
        return error

    def copy_from(self, other: "DuelingQNetwork") -> None:
        self.value_weights = list(other.value_weights)
        self.advantage_weights = [list(row) for row in other.advantage_weights]

    def state_dict(self) -> dict:
        return {
            "feature_count": self.feature_count,
            "action_count": self.action_count,
            "value_weights": self.value_weights,
            "advantage_weights": self.advantage_weights,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["feature_count"] != self.feature_count or state["action_count"] != self.action_count:
            raise ValueError("checkpoint network dimensions do not match")
        self.value_weights = list(map(float, state["value_weights"]))
        self.advantage_weights = [list(map(float, row)) for row in state["advantage_weights"]]


__all__ = ["DuelingQNetwork"]
