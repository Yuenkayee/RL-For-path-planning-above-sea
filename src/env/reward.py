"""Reward terms shared by PPO, DQN, SAC and evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping


@dataclass(frozen=True)
class RewardWeights:
    success: float = 200.0
    storm_collision: float = -200.0
    out_of_bounds: float = -150.0
    timeout: float = -100.0
    step: float = -0.02
    wait: float = -0.01
    speed_switch: float = -0.005
    progress: float = 1.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "RewardWeights":
        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown reward fields: {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in values.items()})


def calculate_reward(
    weights: RewardWeights,
    *,
    previous_distance_nm: float,
    current_distance_nm: float,
    waited: bool,
    speed_switched: bool,
    outcome: str | None,
) -> float:
    reward = weights.step + weights.progress * (
        previous_distance_nm - current_distance_nm
    )
    if waited:
        reward += weights.wait
    if speed_switched:
        reward += weights.speed_switch
    terminal_rewards = {
        "success": weights.success,
        "storm_collision": weights.storm_collision,
        "out_of_bounds": weights.out_of_bounds,
        "timeout": weights.timeout,
    }
    if outcome in terminal_rewards:
        reward += terminal_rewards[outcome]
    return reward


__all__ = ["RewardWeights", "calculate_reward"]
