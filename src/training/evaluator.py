"""Seed-matched evaluation shared by learned and classical planners."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from env.returnEnv import ReturnEnv


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    successes: int
    collisions: int
    timeouts: int
    mean_reward: float
    mean_steps: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


def evaluate_policy(
    env_factory: Callable[[], ReturnEnv],
    policy: Any,
    *,
    seeds: tuple[int, ...],
    max_steps: int | None = None,
) -> EvaluationResult:
    rewards: list[float] = []
    steps_per_episode: list[int] = []
    outcomes: list[str | None] = []
    for seed in seeds:
        env = env_factory()
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        step = 0
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            if terminated or truncated or (max_steps is not None and step >= max_steps):
                outcomes.append(info.get("outcome") or "timeout")
                break
        rewards.append(total_reward)
        steps_per_episode.append(step)
    return EvaluationResult(
        episodes=len(seeds),
        successes=sum(outcome == "success" for outcome in outcomes),
        collisions=sum(outcome == "storm_collision" for outcome in outcomes),
        timeouts=sum(outcome == "timeout" for outcome in outcomes),
        mean_reward=sum(rewards) / len(rewards) if rewards else math.nan,
        mean_steps=sum(steps_per_episode) / len(steps_per_episode) if steps_per_episode else math.nan,
    )


__all__ = ["EvaluationResult", "evaluate_policy"]
