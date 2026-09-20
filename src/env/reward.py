"""Reward terms shared by PPO, DQN, SAC and evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields


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
    planner_potential: float = 1.0
    proximity_5_nm: float = 5.0
    proximity_2_nm: float = 10.0
    proximity_1_nm: float = 20.0
    proximity_0_5_nm: float = 40.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> RewardWeights:
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
    proximity_bonus: float = 0.0,
    potential_shaping: float = 0.0,
) -> float:
    reward = (
        weights.step
        + weights.progress * (previous_distance_nm - current_distance_nm)
        + weights.planner_potential * potential_shaping
        + proximity_bonus
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


_PROXIMITY_REWARDS = (
    (5.0, "proximity_5_nm"),
    (2.0, "proximity_2_nm"),
    (1.0, "proximity_1_nm"),
    (0.5, "proximity_0_5_nm"),
)


def newly_reached_proximity_bonus(
    weights: RewardWeights,
    current_distance_nm: float,
    previously_reached_nm: set[float] | frozenset[float],
) -> tuple[float, tuple[float, ...]]:
    """Return one-time bonuses for distance bands first reached on this step."""
    reached = tuple(
        threshold
        for threshold, _ in _PROXIMITY_REWARDS
        if current_distance_nm <= threshold and threshold not in previously_reached_nm
    )
    bonus = sum(
        getattr(weights, field_name)
        for threshold, field_name in _PROXIMITY_REWARDS
        if threshold in reached
    )
    return bonus, reached


__all__ = ["RewardWeights", "calculate_reward", "newly_reached_proximity_bonus"]
