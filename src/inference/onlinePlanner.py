"""Common online execution for learned and classical methods."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from env.returnEnv import ReturnEnv


@dataclass(frozen=True)
class EpisodeResult:
    outcome: str
    total_reward: float
    steps: int
    path_nm: tuple[tuple[float, float], ...]


def run_policy_episode(
    env: ReturnEnv,
    policy: Any,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    path = [(env.helicopter.x_nm, env.helicopter.y_nm)]
    steps = 0
    while True:
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        path.append(info["helicopter_position_nm"])
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            return EpisodeResult(
                info.get("outcome") or "step_limit",
                total_reward,
                steps,
                tuple(path),
            )


class ClassicalPlannerController:
    """Adapt a global planner to the environment's wait-plus-heading actions."""

    def __init__(self, env: ReturnEnv, planner: Any, *, forecast_steps: int = 30) -> None:
        self.env = env
        self.planner = planner
        self.forecast_steps = forecast_steps

    def predict(self, observation: dict, *, deterministic: bool = True) -> tuple[int, dict]:
        del deterministic
        snapshot = self.env.weather_map.snapshot()
        frames = tuple(snapshot for _ in range(self.forecast_steps + 1))
        angle = math.radians(self.env.frigate.heading_deg)
        velocity = (
            self.env.frigate.speed_knots * math.sin(angle),
            self.env.frigate.speed_knots * math.cos(angle),
        )
        result = self.planner.plan(
            frames,
            (self.env.helicopter.x_nm, self.env.helicopter.y_nm),
            (self.env.frigate.x_nm, self.env.frigate.y_nm),
            velocity,
        )
        if len(result.waypoints) < 2:
            return 0, {"plan": result}
        first = result.waypoints[0]
        target = result.waypoints[1]
        dx = target.x_nm - self.env.helicopter.x_nm
        dy = target.y_nm - self.env.helicopter.y_nm
        planner_step_seconds = float(getattr(self.planner, "step_seconds", 60.0))
        travel_steps = max(
            1,
            math.ceil(
                math.hypot(target.x_nm - first.x_nm, target.y_nm - first.y_nm)
                / self.env.config.flight_speed_knots
                * 3600.0
                / planner_step_seconds
            ),
        )
        if target.time_step - first.time_step > travel_steps:
            return 0, {"plan": result}
        if math.hypot(dx, dy) < 1e-9:
            return 0, {"plan": result}
        heading = math.degrees(math.atan2(dx, dy)) % 360.0
        increment = 360.0 / self.env.config.heading_count
        action = int(round(heading / increment)) % self.env.config.heading_count + 1
        if not observation["action_mask"][action]:
            valid_flight = [
                index
                for index, enabled in enumerate(observation["action_mask"])
                if enabled and index != 0
            ]
            action = min(
                valid_flight,
                key=lambda candidate: abs(
                    (((candidate - 1) * increment - heading + 180.0) % 360.0) - 180.0
                ),
                default=0,
            )
        return action, {"plan": result}


__all__ = ["ClassicalPlannerController", "EpisodeResult", "run_policy_episode"]
