"""Run one experiment method on a common scenario seed."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.runDQN import run_dqn  # noqa: E402
from inference.runPPO import run_ppo  # noqa: E402
from inference.runSAC import run_sac  # noqa: E402
from inference.runSIPP import run_sipp  # noqa: E402
from inference.runTimeAStar import run_time_astar  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("ppo", "dqn", "sac", "sipp", "astar"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    if args.method in {"ppo", "dqn", "sac"} and not args.checkpoint:
        parser.error("--checkpoint is required for learned methods")
    runners = {
        "ppo": lambda: run_ppo(args.checkpoint, seed=args.seed, max_steps=args.max_steps),
        "dqn": lambda: run_dqn(args.checkpoint, seed=args.seed, max_steps=args.max_steps),
        "sac": lambda: run_sac(args.checkpoint, seed=args.seed, max_steps=args.max_steps),
        "sipp": lambda: run_sipp(seed=args.seed, max_steps=args.max_steps),
        "astar": lambda: run_time_astar(seed=args.seed, max_steps=args.max_steps),
    }
    result = runners[args.method]()
    payload = asdict(result)
    payload["path_nm"] = {"points": len(result.path_nm)}
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
