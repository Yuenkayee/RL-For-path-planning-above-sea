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
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="number of environments sampled in parallel",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="neural-network device; auto prefers CUDA, then Apple MPS",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="print in-episode progress every N environment steps",
    )
    parser.add_argument("--quiet", action="store_true", help="disable terminal progress output")
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="disable the four-stage weather curriculum",
    )
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
        num_envs=args.num_envs,
        device=args.device,
        progress_interval_steps=args.progress_interval,
        show_progress=not args.quiet,
        use_curriculum=not args.no_curriculum,
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
