"""Dependency-free dual-resolution weather feature extractor.

The interface intentionally returns a compact tuple so a future CNN-backed
implementation can replace it without changing agents or training loops.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _region_means(values: Sequence[int], shape: tuple[int, int]) -> tuple[float, ...]:
    height, width = shape
    if len(values) != height * width:
        raise ValueError("weather layer length does not match its declared shape")
    halves = (height // 2, width // 2)
    regions = [0.0, 0.0, 0.0, 0.0]
    counts = [0, 0, 0, 0]
    for row in range(height):
        for column in range(width):
            index = (1 if row >= halves[0] else 0) * 2 + (1 if column >= halves[1] else 0)
            regions[index] += values[row * width + column]
            counts[index] += 1
    overall = sum(values) / max(1, len(values))
    return (overall, *(value / max(1, count) for value, count in zip(regions, counts)))


class DualResolutionFeatureExtractor:
    """Summarize map history plus normalized kinematics for linear agents."""

    def __call__(self, observation: dict[str, Any]) -> tuple[float, ...]:
        if "features" in observation:
            return tuple(float(value) for value in observation["features"])
        global_history = observation["global_weather"]
        local_history = observation["local_weather"]
        global_latest = global_history[-1]
        local_latest = local_history[-1]
        global_features = _region_means(global_latest, observation["global_shape"])
        local_features = _region_means(local_latest, observation["local_shape"])
        guidance_features = _region_means(
            observation["guidance_global"], observation["global_shape"]
        )
        global_change = (
            sum(global_latest) - sum(global_history[0])
        ) / max(1, len(global_latest))
        local_change = (
            sum(local_latest) - sum(local_history[0])
        ) / max(1, len(local_latest))
        mask = observation["action_mask"]
        valid_fraction = sum(bool(value) for value in mask) / len(mask)
        return (
            1.0,
            *tuple(float(value) for value in observation["kinematics"]),
            *global_features,
            *local_features,
            *guidance_features,
            *tuple(float(value) for value in observation["guidance_vector"]),
            global_change,
            local_change,
            valid_fraction,
        )

    def compact(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Discard raster payload after extracting features for replay storage."""
        return {
            "features": self(observation),
            "action_mask": tuple(bool(value) for value in observation["action_mask"]),
        }


__all__ = ["DualResolutionFeatureExtractor"]
