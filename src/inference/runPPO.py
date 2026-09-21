"""Load a PPO checkpoint and run deterministic masked inference."""

from __future__ import annotations

from pathlib import Path

from algorithm.ppo import PPOAgent
from env.returnEnv import ReturnEnv
from training.checkpoint import load_checkpoint

from .onlinePlanner import EpisodeResult, run_policy_episode


def run_ppo(
    checkpoint_path: str | Path,
    *,
    seed: int = 0,
    max_steps: int | None = None,
    minimum_route_conflicts: int = 0,
) -> EpisodeResult:
    env = ReturnEnv()
    reset_options = {"minimum_route_conflicts": minimum_route_conflicts}
    observation, _ = env.reset(seed=seed, options=reset_options)
    agent = PPOAgent(
        env.action_count,
        observation,
        seed=seed,
        residual_heading_offsets_deg=env.config.residual_heading_offsets_deg,
    )
    load_checkpoint(agent, checkpoint_path)
    return run_policy_episode(
        env,
        agent,
        seed=seed,
        max_steps=max_steps,
        reset_options=reset_options,
    )


__all__ = ["run_ppo"]
