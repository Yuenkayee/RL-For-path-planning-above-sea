"""Curriculum stages from simple interception to dynamic thunderstorms."""

from __future__ import annotations

from dataclasses import dataclass, replace

from model.weatherSystem import SimulationParameters


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    initial_storm_count: int
    maximum_storm_count: int
    storm_area_scale_range: tuple[float, float]
    moving_weather: bool

    @property
    def storm_area_scale(self) -> float:
        """Compatibility value used by reports; returns the range maximum."""
        return self.storm_area_scale_range[1]


DEFAULT_CURRICULUM = (
    CurriculumStage("interception_only", 0, 0, (1.0, 1.0), False),
    CurriculumStage("static_weather", 2, 2, (0.8, 0.8), False),
    CurriculumStage("moving_weather", 4, 6, (1.0, 1.25), True),
    CurriculumStage("full_dynamic_weather", 8, 12, (1.0, 1.5), True),
)

_STAGE_END_PROGRESS = (0.15, 0.35, 0.60, 1.0)


def stage_for_progress(progress: float) -> CurriculumStage:
    bounded = min(1.0, max(0.0, progress))
    for stage, end_progress in zip(
        DEFAULT_CURRICULUM, _STAGE_END_PROGRESS, strict=True
    ):
        if bounded < end_progress:
            return stage
    return DEFAULT_CURRICULUM[-1]


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
        storm_area_scale=stage.storm_area_scale_range[0],
        storm_area_scale_range=stage.storm_area_scale_range,
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
