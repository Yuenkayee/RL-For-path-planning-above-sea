"""Command-line entry point for DQN training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env import ReturnEnv  # noqa: E402
from training.trainDQN import train_dqn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(ROOT / "build/checkpoints/dqn.pt"))
    parser.add_argument("--log-dir", default=str(ROOT / "build/logs/dqn"))
    args = parser.parse_args()
    _, metrics = train_dqn(
        ReturnEnv(),
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        config_path=ROOT / "config/dqnConfig.json",
        checkpoint_path=args.checkpoint,
        log_dir=args.log_dir,
        seed=args.seed,
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
