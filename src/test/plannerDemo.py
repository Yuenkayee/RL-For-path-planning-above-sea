"""Compare SIPP and time-expanded A-star on the current weather frame."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from env import ReturnEnv  # noqa: E402
from planner import SIPPPlanner, TimeExpandedAStarPlanner  # noqa: E402


def main() -> None:
    env = ReturnEnv()
    env.reset(seed=0)
    snapshot = env.weather_map.snapshot()
    frames = tuple(snapshot for _ in range(31))
    angle = math.radians(env.frigate.heading_deg)
    velocity = (
        env.frigate.speed_knots * math.sin(angle),
        env.frigate.speed_knots * math.cos(angle),
    )
    arguments = (
        frames,
        (env.helicopter.x_nm, env.helicopter.y_nm),
        (env.frigate.x_nm, env.frigate.y_nm),
        velocity,
    )
    for planner in (SIPPPlanner(), TimeExpandedAStarPlanner()):
        result = planner.plan(*arguments)
        print(
            type(planner).__name__,
            result.reached_goal,
            result.expanded_states,
            len(result.waypoints),
        )


if __name__ == "__main__":
    main()
