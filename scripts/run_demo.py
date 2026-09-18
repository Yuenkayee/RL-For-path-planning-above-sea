"""Small terminal demo for the two classical online planners."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.runSIPP import run_sipp  # noqa: E402
from inference.runTimeAStar import run_time_astar  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=("sipp", "astar"), default="sipp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    runner = run_sipp if args.planner == "sipp" else run_time_astar
    result = runner(seed=args.seed, max_steps=args.max_steps)
    print(
        f"planner={args.planner} outcome={result.outcome} "
        f"steps={result.steps} reward={result.total_reward:.3f}"
    )


if __name__ == "__main__":
    main()
