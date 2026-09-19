"""Curriculum stages from simple interception to dynamic thunderstorms."""

from __future__ import annotations

from dataclasses import dataclass, replace

from model.weatherSystem import SimulationParameters


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
    CurriculumStage("dense_dynamic_weather", 5, 8, 1.5, True),
)


def stage_for_progress(progress: float) -> CurriculumStage:
    bounded = min(1.0, max(0.0, progress))
    index = min(len(DEFAULT_CURRICULUM) - 1, int(bounded * len(DEFAULT_CURRICULUM)))
    return DEFAULT_CURRICULUM[index]


def stage_for_episode(episode_id: int, total_episodes: int) -> CurriculumStage:
    """Return the curriculum stage for a zero-based global episode id."""
    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    if not 0 <= episode_id < total_episodes:
        raise ValueError("episode_id must be in [0, total_episodes)")
    return stage_for_progress(episode_id / total_episodes)


def parameters_for_stage(
    base: SimulationParameters, stage: CurriculumStage
) -> SimulationParameters:
    """Apply one curriculum stage without changing observation dimensions."""
    base_weather = base.weather
    maximum_storm_count = min(stage.maximum_storm_count, base_weather.maximum_storm_count)
    initial_storm_count = min(
        stage.initial_storm_count,
        base_weather.initial_storm_count,
        maximum_storm_count,
    )
    motion_speed = base_weather.storm_motion_speed_knots if stage.moving_weather else 0.0
    weather = replace(
        base_weather,
        initial_storm_count=initial_storm_count,
        maximum_storm_count=maximum_storm_count,
        storm_area_scale=stage.storm_area_scale,
        storm_area_scale_range=(stage.storm_area_scale, stage.storm_area_scale),
        storm_motion_speed_knots=motion_speed,
    )
    return replace(base, weather=weather)


__all__ = [
    "CurriculumStage",
    "DEFAULT_CURRICULUM",
    "parameters_for_stage",
    "stage_for_episode",
    "stage_for_progress",
]
