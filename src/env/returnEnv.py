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
from planner.timeExpandedAStar import TimeExpandedAStarPlanner

from .actionMask import build_action_mask, heading_for_action, segment_is_clear
from .config import DEFAULT_ENV_CONFIG_PATH, EnvironmentConfig
from .observation import build_observation
from .reward import calculate_reward, newly_reached_proximity_bonus
from .termination import inside_map, successful_rendezvous


class ReturnEnv(gym.Env):
    """Coordinate weather, vehicles, rewards and terminal conditions.

    The public ``reset`` and ``step`` signatures match Gymnasium closely, but
    the project does not require Gymnasium to be installed. Action zero waits;
    actions 1..N fly at cruise speed on evenly spaced absolute headings.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: EnvironmentConfig | None = None,
        *,
        env_config_path: str | Path = DEFAULT_ENV_CONFIG_PATH,
        weather_parameters: SimulationParameters | None = None,
        weather_config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.config = env_config or EnvironmentConfig.from_json(env_config_path)
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
        self._previous_moving = False
        self._history: deque = deque(maxlen=self.config.weather_history_frames)
        self._guidance_planner = TimeExpandedAStarPlanner(
            horizon_steps=30,
            step_seconds=60.0,
            helicopter_speed_knots=self.config.flight_speed_knots,
        )
        self._guidance_plan = None
        self._guidance_grid: tuple[tuple[float, ...], ...] | None = None
        self._reached_proximity_thresholds_nm: set[float] = set()
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
        del options
        super().reset(seed=seed)
        parameters = self._base_weather_parameters
        episode_seed = seed if seed is not None else parameters.random_seed
        episode_rng = random.Random(episode_seed)
        frigate_heading_deg = episode_rng.choice(self.config.frigate_heading_choices_deg)
        storm_area_scale = episode_rng.uniform(*parameters.weather.storm_area_scale_range)
        parameters = replace(
            parameters,
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
        self.elapsed_seconds = 0.0
        self._weather_elapsed_seconds = 0.0
        self._previous_moving = False
        self._reached_proximity_thresholds_nm.clear()
        self._done = False
        self._history.clear()
        snapshot = self.weather_map.snapshot()
        for _ in range(self.config.weather_history_frames):
            self._history.append(snapshot)
        self._refresh_guidance()
        return self._observation(), {
            "seed": parameters.random_seed,
            "episode_config": {
                "frigate_heading_deg": frigate_heading_deg,
                "storm_area_scale": storm_area_scale,
            },
        }

    def _refresh_guidance(self) -> None:
        snapshot = self.weather_map.snapshot()
        frames = tuple(snapshot for _ in range(self._guidance_planner.horizon_steps + 1))
        heading_rad = math.radians(self.frigate.heading_deg)
        velocity = (
            self.frigate.speed_knots * math.sin(heading_rad),
            self.frigate.speed_knots * math.cos(heading_rad),
        )
        self._guidance_plan = self._guidance_planner.plan(
            frames,
            (self.helicopter.x_nm, self.helicopter.y_nm),
            (self.frigate.x_nm, self.frigate.y_nm),
            velocity,
        )
        self._guidance_grid = rasterize_guidance(
            self._guidance_plan.waypoints,
            shape=(len(snapshot.global_grid), len(snapshot.global_grid[0])),
            resolution_nm=snapshot.global_resolution_nm,
        )

    def get_action_mask(self) -> tuple[bool, ...]:
        return build_action_mask(
            self.weather_map,
            (self.helicopter.x_nm, self.helicopter.y_nm),
            heading_count=self.config.heading_count,
            flight_speed_knots=self.config.flight_speed_knots,
            wait_speed_knots=self.config.wait_speed_knots,
            duration_seconds=self.config.control_step_seconds,
            current_heading_deg=self.helicopter.heading_deg,
        )

    def _observation(self) -> dict[str, Any]:
        heading_rad = math.radians(self.frigate.heading_deg)
        frigate_velocity = (
            self.frigate.speed_knots * math.sin(heading_rad),
            self.frigate.speed_knots * math.cos(heading_rad),
        )
        guidance_vector: tuple[float, ...] = ()
        if self._guidance_plan is not None and self._guidance_plan.waypoints:
            next_index = min(1, len(self._guidance_plan.waypoints) - 1)
            next_waypoint = self._guidance_plan.waypoints[next_index]
            final_waypoint = self._guidance_plan.waypoints[-1]
            width, height = self.weather_map.area_size_nm
            guidance_vector = (
                (next_waypoint.x_nm - self.helicopter.x_nm) / width,
                (next_waypoint.y_nm - self.helicopter.y_nm) / height,
                final_waypoint.x_nm / width,
                final_waypoint.y_nm / height,
                final_waypoint.time_step / self._guidance_planner.horizon_steps,
                1.0 if self._guidance_plan.reached_goal else 0.0,
            )
        else:
            guidance_vector = (0.0,) * 6
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
        moving = action != 0
        speed_switched = moving != self._previous_moving
        heading = heading_for_action(action, self.config.heading_count)
        self.helicopter.select_motion(moving=moving, heading_deg=heading)
        self.helicopter.step(dt)
        self.frigate.step(dt)
        self.elapsed_seconds += dt
        self._weather_elapsed_seconds += dt

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

        if weather_updated and not terminated:
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
        reward = calculate_reward(
            self.config.reward,
            previous_distance_nm=previous_distance,
            current_distance_nm=current_distance,
            waited=not moving,
            speed_switched=speed_switched,
            outcome=outcome,
            proximity_bonus=proximity_bonus,
        )
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
        }
        return observation, reward, terminated, truncated, info


__all__ = ["ReturnEnv"]
