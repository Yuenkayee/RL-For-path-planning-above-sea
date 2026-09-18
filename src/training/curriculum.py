"""Curriculum stages from simple interception to dynamic thunderstorms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    initial_storm_count: int
    maximum_storm_count: int
    storm_area_scale: float
    moving_weather: bool


DEFAULT_CURRICULUM = (
    CurriculumStage("interception_only", 0, 0, 1.0, False),
    CurriculumStage("static_weather", 2, 2, 0.8, False),
    CurriculumStage("moving_weather", 3, 5, 1.0, True),
    CurriculumStage("dense_dynamic_weather", 5, 8, 2.0, True),
)


def stage_for_progress(progress: float) -> CurriculumStage:
    bounded = min(1.0, max(0.0, progress))
    index = min(len(DEFAULT_CURRICULUM) - 1, int(bounded * len(DEFAULT_CURRICULUM)))
    return DEFAULT_CURRICULUM[index]


__all__ = ["CurriculumStage", "DEFAULT_CURRICULUM", "stage_for_progress"]
