"""Run the rolling-horizon SIPP classical baseline."""

from __future__ import annotations

from env.returnEnv import ReturnEnv
from planner.sippPlanner import SIPPPlanner

from .onlinePlanner import ClassicalPlannerController, EpisodeResult, run_policy_episode


def run_sipp(*, seed: int = 0, max_steps: int | None = None) -> EpisodeResult:
    env = ReturnEnv()
    planner = SIPPPlanner()
    controller = ClassicalPlannerController(env, planner)
    return run_policy_episode(env, controller, seed=seed, max_steps=max_steps)


__all__ = ["run_sipp"]
