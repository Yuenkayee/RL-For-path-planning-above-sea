"""Visualize one fixed-seed path-planning episode as GIF and PNG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algorithm.dqn import DQNAgent  # noqa: E402
from algorithm.ppo import PPOAgent  # noqa: E402
from algorithm.sac import SACAgent  # noqa: E402
from env import ReturnEnv  # noqa: E402
from inference.episodeVisualization import (  # noqa: E402
    run_episode_trace,
    save_episode_visualization,
)
from inference.onlinePlanner import ClassicalPlannerController  # noqa: E402
from planner.sippPlanner import SIPPPlanner  # noqa: E402
from planner.timeExpandedAStar import TimeExpandedAStarPlanner  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_policy(
    method: str,
    env: ReturnEnv,
    observation: dict,
    *,
    checkpoint: Path | None,
    seed: int,
):
    if method == "sipp":
        return ClassicalPlannerController(env, SIPPPlanner())
    if method == "astar":
        return ClassicalPlannerController(env, TimeExpandedAStarPlanner())

    default_name = "ppo_residual.pt" if method == "ppo" else f"{method}.pt"
    checkpoint = checkpoint or ROOT / "build" / "checkpoints" / default_name
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. Train {method.upper()} first or pass --checkpoint."
        )
    agent_types = {"ppo": PPOAgent, "dqn": DQNAgent, "sac": SACAgent}
    agent_kwargs = (
        {"residual_heading_offsets_deg": env.config.residual_heading_offsets_deg}
        if method == "ppo"
        else {}
    )
    agent = agent_types[method](env.action_count, observation, seed=seed, **agent_kwargs)
    load_checkpoint(agent, checkpoint)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("ppo", "dqn", "sac", "sipp", "astar"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=_positive_integer, default=1200)
    parser.add_argument("--minimum-route-conflicts", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--frame-stride",
        type=_positive_integer,
        default=10,
        help="capture one frame every N control steps; 10 equals one simulated minute",
    )
    parser.add_argument("--fps", type=_positive_integer, default=8)
    parser.add_argument("--output", type=Path, help="animated GIF output path")
    parser.add_argument("--snapshot-output", type=Path, help="final-state PNG output path")
    args = parser.parse_args()

    output = args.output or ROOT / "build" / "evaluation" / (
        f"{args.method}_seed{args.seed}.gif"
    )
    snapshot_output = args.snapshot_output or output.with_suffix(".png")

    env = ReturnEnv()
    reset_options = {"minimum_route_conflicts": args.minimum_route_conflicts}
    observation, _ = env.reset(seed=args.seed, options=reset_options)
    try:
        policy = _build_policy(
            args.method,
            env,
            observation,
            checkpoint=args.checkpoint,
            seed=args.seed,
        )
    except FileNotFoundError as error:
        parser.error(str(error))

    trace = run_episode_trace(
        env,
        policy,
        method=args.method,
        seed=args.seed,
        max_steps=args.max_steps,
        frame_stride=args.frame_stride,
        reset_options=reset_options,
    )
    gif_path = save_episode_visualization(trace, output, fps=args.fps)
    png_path = save_episode_visualization(trace, snapshot_output, fps=args.fps)
    print(
        json.dumps(
            {
                "method": trace.method,
                "seed": trace.seed,
                "outcome": trace.outcome,
                "total_reward": trace.total_reward,
                "steps": trace.steps,
                "animation": str(gif_path),
                "snapshot": str(png_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
