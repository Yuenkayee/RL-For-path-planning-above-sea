"""Run the rolling time-expanded A-star classical baseline."""

from __future__ import annotations

from env.returnEnv import ReturnEnv
from planner.timeExpandedAStar import TimeExpandedAStarPlanner

from .onlinePlanner import ClassicalPlannerController, EpisodeResult, run_policy_episode


def run_time_astar(
    *, seed: int = 0, max_steps: int | None = None, minimum_route_conflicts: int = 0
) -> EpisodeResult:
    env = ReturnEnv()
    planner = TimeExpandedAStarPlanner()
    controller = ClassicalPlannerController(env, planner)
    return run_policy_episode(
        env,
        controller,
        seed=seed,
        max_steps=max_steps,
        reset_options={"minimum_route_conflicts": minimum_route_conflicts},
    )


__all__ = ["run_time_astar"]
