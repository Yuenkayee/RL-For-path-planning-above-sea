"""Train PPO briefly and execute its deterministic policy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from env import ReturnEnv  # noqa: E402
from inference.onlinePlanner import run_policy_episode  # noqa: E402
from training.trainPPO import train_ppo  # noqa: E402


def main() -> None:
    env = ReturnEnv()
    agent, metrics = train_ppo(
        env,
        episodes=1,
        max_steps_per_episode=20,
        config_path=ROOT / "config/ppoConfig.json",
        seed=0,
    )
    result = run_policy_episode(env, agent, seed=1, max_steps=20)
    print(metrics)
    print(result.outcome, result.steps, round(result.total_reward, 3))


if __name__ == "__main__":
    main()
