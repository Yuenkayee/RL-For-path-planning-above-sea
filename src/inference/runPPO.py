"""Load a PPO checkpoint and run deterministic masked inference."""

from __future__ import annotations

from pathlib import Path

from algorithm.ppo import PPOAgent
from env.returnEnv import ReturnEnv
from training.checkpoint import load_checkpoint

from .onlinePlanner import EpisodeResult, run_policy_episode


def run_ppo(
    checkpoint_path: str | Path, *, seed: int = 0, max_steps: int | None = None
) -> EpisodeResult:
    env = ReturnEnv()
    observation, _ = env.reset(seed=seed)
    agent = PPOAgent(env.action_count, observation, seed=seed)
    load_checkpoint(agent, checkpoint_path)
    return run_policy_episode(env, agent, seed=seed, max_steps=max_steps)


__all__ = ["run_ppo"]
