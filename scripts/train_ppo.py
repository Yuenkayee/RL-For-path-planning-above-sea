"""Command-line entry point for PPO training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env import ReturnEnv  # noqa: E402
from training.trainPPO import train_ppo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(ROOT / "build/checkpoints/ppo.pt"))
    parser.add_argument("--log-dir", default=str(ROOT / "build/logs/ppo"))
    args = parser.parse_args()
    _, metrics = train_ppo(
        ReturnEnv(),
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        config_path=ROOT / "config/ppoConfig.json",
        checkpoint_path=args.checkpoint,
        log_dir=args.log_dir,
        seed=args.seed,
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
