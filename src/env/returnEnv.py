"""Gymnasium helicopter return environment with NumPy observations."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from model.frigate import Frigate
from model.helicopter import Helicopter
from model.weatherSystem import DEFAULT_CONFIG_PATH, SimulationParameters, WeatherSystem
from planner.guidanceMap import rasterize_guidance
from planner.interceptPredictor import predict_intercept
from planner.timeExpandedAStar import TimeExpandedAStarPlanner

from .actionMask import (
    build_residual_action_mask,
    heading_for_residual_action,
    segment_is_clear,
)
from .config import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_PLANNER_CONFIG_PATH,
    EnvironmentConfig,
    PlannerConfig,
)
from .observation import build_observation
from .reward import calculate_reward, newly_reached_proximity_bonus
from .scenario import (
    RouteConflictSummary,
    count_nominal_route_conflicts,
    sample_frigate_initial_position,
)
from .termination import inside_map, successful_rendezvous


class ReturnEnv(gym.Env):
    """Coordinate weather, vehicles, rewards and terminal conditions.

    The public ``reset`` and ``step`` signatures match Gymnasium closely, but
    Action zero waits; actions 1..N fly at cruise speed with a configured
    heading residual relative to the rolling global-planner guidance.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: EnvironmentConfig | None = None,
        *,
        env_config_path: str | Path = DEFAULT_ENV_CONFIG_PATH,
        planner_config: PlannerConfig | None = None,
        planner_config_path: str | Path = DEFAULT_PLANNER_CONFIG_PATH,
        weather_parameters: SimulationParameters | None = None,
        weather_config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.config = env_config or EnvironmentConfig.from_json(env_config_path)
        self.planner_config = planner_config or PlannerConfig.from_json(planner_config_path)
        if not math.isclose(
            self.planner_config.flight_speed_knots,
            self.config.flight_speed_knots,
            abs_tol=1e-9,
        ) or not math.isclose(
            self.planner_config.wait_speed_knots,
            self.config.wait_speed_knots,
            abs_tol=1e-9,
        ):
            raise ValueError("planner and environment speed modes must match")
        self._base_weather_parameters = weather_parameters or SimulationParameters.from_json(
            weather_config_path
        )
        self.action_count = self.config.action_count
        self.action_space = spaces.Discrete(self.action_count)
        global_height = round(
            self._base_weather_parameters.map_size_nm[1]
            / self._base_weather_parameters.global_resolution_nm
        )
        global_width = round(
            self._base_weather_parameters.map_size_nm[0]
            / self._base_weather_parameters.global_resolution_nm
        )
        local_height = round(
            self._base_weather_parameters.local_size_nm[1]
            / self._base_weather_parameters.local_resolution_nm
        )
        local_width = round(
            self._base_weather_parameters.local_size_nm[0]
            / self._base_weather_parameters.local_resolution_nm
        )
        self.observation_space = spaces.Dict(
            {
                "global_weather": spaces.Box(
                    0,
                    1,
                    shape=(self.config.weather_history_frames, global_height, global_width),
                    dtype=np.uint8,
                ),
                "local_weather": spaces.Box(
                    0,
                    1,
                    shape=(self.config.weather_history_frames, local_height, local_width),
                    dtype=np.uint8,
                ),
                "kinematics": spaces.Box(-2.0, 2.0, shape=(11,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(self.action_count),
                "guidance_global": spaces.Box(
                    0.0, 1.0, shape=(global_height, global_width), dtype=np.float32
                ),
                "guidance_vector": spaces.Box(-1.5, 1.5, shape=(6,), dtype=np.float32),
            }
        )
        self.weather_system: WeatherSystem
        self.helicopter: Helicopter
        self.frigate: Frigate
        self.elapsed_seconds = 0.0
        self._weather_elapsed_seconds = 0.0
        self._guidance_elapsed_seconds = 0.0
        self._previous_moving = False
        self._history: deque = deque(maxlen=self.config.weather_history_frames)
        self._guidance_planner = TimeExpandedAStarPlanner(
            horizon_steps=self.planner_config.horizon_steps,
            step_seconds=self.planner_config.planning_step_seconds,
            helicopter_speed_knots=self.config.flight_speed_knots,
            allow_wait=self.planner_config.allow_wait,
            goal_tolerance_nm=self.config.success_distance_nm,
            maximum_expanded_states=self.planner_config.maximum_expanded_states,
        )
        self._guidance_plan = None
        self._guidance_grid: tuple[tuple[float, ...], ...] | None = None
        self._forecast_frames = ()
        self._forecast_time_minutes: float | None = None
        self._planner_potential = 0.0
        self._reached_proximity_thresholds_nm: set[float] = set()
        self._episode_seed = -1
        self._route_conflicts = RouteConflictSummary(0, 0, 0)
        self._minimum_storm_clearance_nm = math.inf
        self._wait_steps = 0
        self._heading_change_steps = 0
        self._avoidance_decision_steps = 0
        self._blocked_flight_actions = 0
        self._done = False
        self.reset()

    @property
    def weather_map(self):
        return self.weather_system.weather_map

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_options = options or {}
        minimum_route_conflicts = int(reset_options.get("minimum_route_conflicts", 0))
        if minimum_route_conflicts < 0:
            raise ValueError("minimum_route_conflicts must be non-negative")
        super().reset(seed=seed)
        parameters = self._base_weather_parameters
        episode_seed = seed if seed is not None else parameters.random_seed
        episode_rng = random.Random(episode_seed)
        frigate_heading_deg = episode_rng.choice(self.config.frigate_heading_choices_deg)
        storm_area_scale = episode_rng.uniform(*parameters.weather.storm_area_scale_range)
        frigate_initial_nm = parameters.frigate_initial_nm
        if self.config.randomize_frigate_initial_position:
            frigate_initial_nm, _ = sample_frigate_initial_position(
                episode_rng,
                map_size_nm=parameters.map_size_nm,
                helicopter_nm=parameters.helicopter_initial_nm,
                heading_deg=frigate_heading_deg,
                frigate_speed_knots=self.config.frigate_speed_knots,
                helicopter_speed_knots=self.config.flight_speed_knots,
                route_minutes=self.config.frigate_minimum_route_minutes,
                boundary_margin_nm=self.config.frigate_boundary_margin_nm,
                minimum_distance_nm=self.config.frigate_minimum_initial_distance_nm,
            )
        parameters = replace(
            parameters,
            frigate_initial_nm=frigate_initial_nm,
            weather=replace(parameters.weather, storm_area_scale=storm_area_scale),
        )
        if seed is not None:
            parameters = replace(parameters, random_seed=seed)
        self.weather_system = WeatherSystem(parameters)
        self.helicopter = Helicopter(
            *parameters.helicopter_initial_nm,
            cruise_speed_knots=self.config.flight_speed_knots,
            wait_speed_knots=self.config.wait_speed_knots,
        )
        self.frigate = Frigate(
            *parameters.frigate_initial_nm,
            speed_knots=self.config.frigate_speed_knots,
            heading_deg=frigate_heading_deg,
        )
        intercept = predict_intercept(
            (self.helicopter.x_nm, self.helicopter.y_nm),
            (self.frigate.x_nm, self.frigate.y_nm),
            self._frigate_velocity(),
            self.config.flight_speed_knots,
        )
        if minimum_route_conflicts:
            if intercept is None:
                raise RuntimeError("cannot create route conflicts without a feasible intercept")
            self.weather_system.configure_route_conflicts(
                route_start_nm=(self.helicopter.x_nm, self.helicopter.y_nm),
                route_end_nm=intercept.point_nm,
                route_duration_minutes=intercept.time_hours * 60.0,
                conflict_count=minimum_route_conflicts,
            )
        self.elapsed_seconds = 0.0
        self._weather_elapsed_seconds = 0.0
        self._guidance_elapsed_seconds = 0.0
        self._previous_moving = False
        self._reached_proximity_thresholds_nm.clear()
        self._episode_seed = int(parameters.random_seed) if parameters.random_seed is not None else -1
        self._minimum_storm_clearance_nm = self.weather_system.minimum_storm_clearance_nm(
            parameters.helicopter_initial_nm
        )
        self._wait_steps = 0
        self._heading_change_steps = 0
        self._avoidance_decision_steps = 0
        self._blocked_flight_actions = 0
        self._done = False
        self._history.clear()
        self._forecast_frames = ()
        self._forecast_time_minutes = None
        snapshot = self.weather_map.snapshot()
        for _ in range(self.config.weather_history_frames):
            self._history.append(snapshot)
        self._refresh_guidance()
        self._route_conflicts = (
            count_nominal_route_conflicts(
                self._forecast_frames,
                start_nm=(self.helicopter.x_nm, self.helicopter.y_nm),
                intercept=intercept,
                forecast_step_minutes=self._guidance_planner.step_seconds / 60.0,
            )
            if intercept is not None
            else RouteConflictSummary(0, 0, 0)
        )
        if self._route_conflicts.regions < minimum_route_conflicts:
            raise RuntimeError(
                "configured dynamic scenario did not produce the required independent "
                f"route conflicts: required={minimum_route_conflicts}, "
                f"actual={self._route_conflicts.regions}"
            )
        self._planner_potential = self._guidance_potential()
        return self._observation(), {
            "seed": parameters.random_seed,
            "episode_config": {
                "frigate_heading_deg": frigate_heading_deg,
                "frigate_initial_nm": frigate_initial_nm,
                "storm_area_scale": storm_area_scale,
                "route_conflict_regions": self._route_conflicts.regions,
                "route_conflict_fraction": self._route_conflicts.occupied_fraction,
                "required_route_conflicts": minimum_route_conflicts,
            },
        }

    def _refresh_guidance(self) -> None:
        snapshot = self.weather_map.snapshot()
        if self._forecast_time_minutes != self.weather_system.time_minutes:
            self._forecast_frames = self.weather_system.forecast_snapshots(
                self._guidance_planner.horizon_steps,
                step_minutes=self._guidance_planner.step_seconds / 60.0,
            )
            self._forecast_time_minutes = self.weather_system.time_minutes
        heading_rad = math.radians(self.frigate.heading_deg)
        velocity = (
            self.frigate.speed_knots * math.sin(heading_rad),
            self.frigate.speed_knots * math.cos(heading_rad),
        )
        self._guidance_plan = self._guidance_planner.plan(
            self._forecast_frames,
            (self.helicopter.x_nm, self.helicopter.y_nm),
            (self.frigate.x_nm, self.frigate.y_nm),
            velocity,
        )
        self._guidance_grid = rasterize_guidance(
            self._guidance_plan.waypoints,
            shape=(len(snapshot.global_grid), len(snapshot.global_grid[0])),
            resolution_nm=snapshot.global_resolution_nm,
        )
        self._guidance_elapsed_seconds = 0.0

    def _frigate_velocity(self) -> tuple[float, float]:
        heading_rad = math.radians(self.frigate.heading_deg)
        return (
            self.frigate.speed_knots * math.sin(heading_rad),
            self.frigate.speed_knots * math.cos(heading_rad),
        )

    def _fallback_intercept_point(self) -> tuple[float, float]:
        helicopter = (self.helicopter.x_nm, self.helicopter.y_nm)
        frigate = (self.frigate.x_nm, self.frigate.y_nm)
        intercept = predict_intercept(
            helicopter,
            frigate,
            self._frigate_velocity(),
            self.config.flight_speed_knots,
        )
        return intercept.point_nm if intercept is not None else frigate

    def _lookahead_point(self) -> tuple[float, float]:
        start = (self.helicopter.x_nm, self.helicopter.y_nm)
        if self._guidance_plan is None or not self._guidance_plan.waypoints:
            return self._fallback_intercept_point()
        waypoints = self._guidance_plan.waypoints
        distance = 0.0
        previous = start
        for waypoint in waypoints:
            point = (waypoint.x_nm, waypoint.y_nm)
            segment = math.dist(previous, point)
            if distance + segment >= self.planner_config.lookahead_distance_nm:
                remaining = self.planner_config.lookahead_distance_nm - distance
                fraction = 0.0 if segment == 0.0 else remaining / segment
                return (
                    previous[0] + (point[0] - previous[0]) * fraction,
                    previous[1] + (point[1] - previous[1]) * fraction,
                )
            distance += segment
            previous = point
        final = waypoints[-1]
        return final.x_nm, final.y_nm

    def _reference_heading_deg(self) -> float:
        target = self._lookahead_point()
        dx = target[0] - self.helicopter.x_nm
        dy = target[1] - self.helicopter.y_nm
        if math.hypot(dx, dy) < 1e-9:
            return self.helicopter.heading_deg
        return math.degrees(math.atan2(dx, dy)) % 360.0

    def _guidance_potential(self) -> float:
        """Negative estimated remaining route length for potential shaping."""
        position = (self.helicopter.x_nm, self.helicopter.y_nm)
        if self._guidance_plan is None or not self._guidance_plan.waypoints:
            return -math.dist(position, self._fallback_intercept_point())
        points = [(waypoint.x_nm, waypoint.y_nm) for waypoint in self._guidance_plan.waypoints]
        suffix = [0.0] * len(points)
        for index in range(len(points) - 2, -1, -1):
            suffix[index] = suffix[index + 1] + math.dist(points[index], points[index + 1])
        remaining = min(
            math.dist(position, point) + suffix[index]
            for index, point in enumerate(points)
        )
        return -remaining

    def get_action_mask(self) -> tuple[bool, ...]:
        return build_residual_action_mask(
            self.weather_map,
            (self.helicopter.x_nm, self.helicopter.y_nm),
            reference_heading_deg=self._reference_heading_deg(),
            residual_offsets_deg=self.config.residual_heading_offsets_deg,
            flight_speed_knots=self.config.flight_speed_knots,
            wait_speed_knots=self.config.wait_speed_knots,
            duration_seconds=self.config.control_step_seconds,
            current_heading_deg=self.helicopter.heading_deg,
        )

    def _observation(self) -> dict[str, Any]:
        frigate_velocity = self._frigate_velocity()
        guidance_vector: tuple[float, ...] = ()
        if self._guidance_plan is not None and self._guidance_plan.waypoints:
            next_x, next_y = self._lookahead_point()
            final_waypoint = self._guidance_plan.waypoints[-1]
            width, height = self.weather_map.area_size_nm
            guidance_vector = (
                (next_x - self.helicopter.x_nm) / width,
                (next_y - self.helicopter.y_nm) / height,
                (final_waypoint.x_nm - self.helicopter.x_nm) / width,
                (final_waypoint.y_nm - self.helicopter.y_nm) / height,
                final_waypoint.time_step / self._guidance_planner.horizon_steps,
                1.0 if self._guidance_plan.reached_goal else 0.0,
            )
        else:
            target_x, target_y = self._fallback_intercept_point()
            width, height = self.weather_map.area_size_nm
            dx = (target_x - self.helicopter.x_nm) / width
            dy = (target_y - self.helicopter.y_nm) / height
            travel_minutes = (
                math.dist(
                    (self.helicopter.x_nm, self.helicopter.y_nm),
                    (target_x, target_y),
                )
                / self.config.flight_speed_knots
                * 60.0
            )
            guidance_vector = (
                dx,
                dy,
                dx,
                dy,
                min(1.0, travel_minutes / self.planner_config.planning_horizon_minutes),
                0.0,
            )
        return build_observation(
            tuple(self._history),
            helicopter_nm=(self.helicopter.x_nm, self.helicopter.y_nm),
            frigate_nm=(self.frigate.x_nm, self.frigate.y_nm),
            helicopter_heading_deg=self.helicopter.heading_deg,
            helicopter_speed_knots=self.helicopter.speed_knots,
            helicopter_cruise_speed_knots=self.config.flight_speed_knots,
            frigate_velocity_nm_per_hour=frigate_velocity,
            elapsed_seconds=self.elapsed_seconds,
            maximum_seconds=self.config.maximum_episode_minutes * 60.0,
            action_mask=self.get_action_mask(),
            guidance_global=self._guidance_grid,
            guidance_vector=guidance_vector,
        )

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("episode is done; call reset before step")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action = int(action)
        if not 0 <= action < self.action_count:
            raise ValueError(f"action must be in [0, {self.action_count - 1}]")

        dt = self.config.control_step_seconds
        start = (self.helicopter.x_nm, self.helicopter.y_nm)
        frigate_start = (self.frigate.x_nm, self.frigate.y_nm)
        previous_distance = math.dist(start, frigate_start)
        previous_potential = self._planner_potential
        moving = action != 0
        speed_switched = moving != self._previous_moving
        previous_heading = self.helicopter.heading_deg
        action_mask_before = self.get_action_mask()
        blocked_flight_actions = sum(not allowed for allowed in action_mask_before[1:])
        self._blocked_flight_actions += blocked_flight_actions
        if blocked_flight_actions:
            self._avoidance_decision_steps += 1
        if not moving:
            self._wait_steps += 1
        reference_heading = self._reference_heading_deg()
        heading = heading_for_residual_action(
            action,
            reference_heading,
            self.config.residual_heading_offsets_deg,
        )
        self.helicopter.select_motion(moving=moving, heading_deg=heading)
        if moving and self._previous_moving:
            heading_change = abs((self.helicopter.heading_deg - previous_heading + 180.0) % 360.0 - 180.0)
            if heading_change > 1e-6:
                self._heading_change_steps += 1
        self.helicopter.step(dt)
        self.frigate.step(dt)
        self.elapsed_seconds += dt
        self._weather_elapsed_seconds += dt
        self._guidance_elapsed_seconds += dt

        weather_updated = False
        while self._weather_elapsed_seconds + 1e-9 >= self.config.weather_step_seconds:
            try:
                self.weather_system.step()
            except StopIteration:
                break
            self._weather_elapsed_seconds -= self.config.weather_step_seconds
            weather_updated = True

        end = (self.helicopter.x_nm, self.helicopter.y_nm)
        frigate_end = (self.frigate.x_nm, self.frigate.y_nm)
        current_distance = math.dist(end, frigate_end)
        self._minimum_storm_clearance_nm = min(
            self._minimum_storm_clearance_nm,
            self.weather_system.minimum_storm_clearance_nm(end),
        )
        outcome: str | None = None
        terminated = False
        truncated = False

        if not inside_map(end, self.weather_map.area_size_nm):
            outcome = "out_of_bounds"
            terminated = True
        elif not segment_is_clear(self.weather_map, start, end):
            outcome = "storm_collision"
            terminated = True
        else:
            self.weather_system.move_local_window(end)
            if inside_map(frigate_end, self.weather_map.area_size_nm) and successful_rendezvous(
                self.weather_map,
                end,
                frigate_end,
                maximum_distance_nm=self.config.success_distance_nm,
            ):
                outcome = "success"
                terminated = True

        if (
            not terminated
            and (
                weather_updated
                or self._guidance_elapsed_seconds + 1e-9
                >= self.planner_config.replanning_interval_seconds
            )
        ):
            self._refresh_guidance()

        maximum_seconds = self.config.maximum_episode_minutes * 60.0
        if not terminated and self.elapsed_seconds + 1e-9 >= maximum_seconds:
            outcome = "timeout"
            truncated = True
        if (
            not terminated
            and not truncated
            and not inside_map(frigate_end, self.weather_map.area_size_nm)
        ):
            outcome = "timeout"
            truncated = True

        proximity_bonus = 0.0
        newly_reached_thresholds: tuple[float, ...] = ()
        if outcome not in {"storm_collision", "out_of_bounds"}:
            proximity_bonus, newly_reached_thresholds = newly_reached_proximity_bonus(
                self.config.reward,
                current_distance,
                self._reached_proximity_thresholds_nm,
            )
            self._reached_proximity_thresholds_nm.update(newly_reached_thresholds)
        current_potential = self._guidance_potential()
        potential_shaping = (
            self.config.potential_discount_factor * current_potential - previous_potential
        )
        reward = calculate_reward(
            self.config.reward,
            previous_distance_nm=previous_distance,
            current_distance_nm=current_distance,
            waited=not moving,
            speed_switched=speed_switched,
            outcome=outcome,
            proximity_bonus=proximity_bonus,
            potential_shaping=potential_shaping,
        )
        self._planner_potential = current_potential
        self._previous_moving = moving
        self._done = terminated or truncated
        self._history.append(self.weather_map.snapshot())
        observation = self._observation()
        info = {
            "outcome": outcome,
            "elapsed_seconds": self.elapsed_seconds,
            "distance_to_frigate_nm": current_distance,
            "proximity_bonus": proximity_bonus,
            "newly_reached_proximity_thresholds_nm": newly_reached_thresholds,
            "weather_updated": weather_updated,
            "helicopter_position_nm": end,
            "frigate_position_nm": frigate_end,
            "action_mask": observation["action_mask"],
            "reference_heading_deg": reference_heading,
            "executed_heading_deg": heading,
            "planner_potential": current_potential,
            "potential_shaping": potential_shaping,
            "episode_seed": self._episode_seed,
            "route_conflict_regions": self._route_conflicts.regions,
            "route_conflict_fraction": self._route_conflicts.occupied_fraction,
            "minimum_storm_clearance_nm": self._minimum_storm_clearance_nm,
            "wait_steps": self._wait_steps,
            "heading_change_steps": self._heading_change_steps,
            "avoidance_decision_steps": self._avoidance_decision_steps,
            "blocked_flight_actions": self._blocked_flight_actions,
        }
        return observation, reward, terminated, truncated, info


__all__ = ["ReturnEnv"]
