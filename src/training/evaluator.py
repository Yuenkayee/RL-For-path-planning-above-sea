"""Seed-matched evaluation shared by learned and classical planners."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_REPOSITORY_ROOT / "build" / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from env.returnEnv import ReturnEnv  # noqa: E402


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
        mean_steps=sum(steps_per_episode) / len(steps_per_episode)
        if steps_per_episode
        else math.nan,
    )


def save_evaluation_plot(results: dict[str, EvaluationResult], path: str | Path) -> Path:
    """Save success-rate and mean-step comparison as a PNG figure."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    names = list(results)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, [results[name].success_rate for name in names])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Success rate")
    axes[1].bar(names, [results[name].mean_steps for name in names])
    axes[1].set_title("Mean control steps")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


__all__ = ["EvaluationResult", "evaluate_policy", "save_evaluation_plot"]
