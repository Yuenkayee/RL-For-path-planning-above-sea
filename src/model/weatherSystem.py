"""Stochastic thunderstorm simulator backed by :mod:`weatherMap`.

This module implements a deliberately lightweight *simulation* rather than a
numerical weather-prediction model.  It follows three public meteorological
observations:

* WMO sea-state code 6 is "very rough", with 4--6 m waves.  Sea state does
  not uniquely determine wind speed, so wave height and storm motion are kept
  as separate parameters.
* NOAA describes ordinary thunderstorm cells as having a roughly 30--60
  minute life cycle with developing, mature and dissipating stages.
* Radar nowcasting commonly decomposes short-term evolution into advection
  plus stochastic growth and decay of precipitation areas.

References:
    https://community.wmo.int/site/knowledge-hub/programmes-and-initiatives/
        marine-services/frequently-asked-questions
    https://www.nssl.noaa.gov/education/svrwx101/thunderstorms/types/
    https://journals.ametsoc.org/view/journals/atsc/69/11/jas-d-12-029.1.xml

Each simulated cell is therefore an advected, irregular ellipse whose size
follows a three-stage life-cycle envelope.  Cells can dissipate and new cells
can be initiated.  The resulting union is rasterized into the shared binary
``WeatherMap`` at both global and local resolutions after every time step.
No physical claim beyond this scenario-level abstraction is intended.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random
from typing import Iterator

try:  # Package import: ``from model.weatherSystem import WeatherSystem``.
    from .weatherMap import FREE, THUNDERSTORM, WeatherMap, WeatherMapSnapshot
except ImportError:  # Direct execution from the ``src/model`` directory.
    from weatherMap import FREE, THUNDERSTORM, WeatherMap, WeatherMapSnapshot


@dataclass(frozen=True)
class WeatherParameters:
    """Weather and sea-state controls used by the stochastic simulator.

    Directions are headings *towards* which a cell moves, measured clockwise
    from north: 0 degrees is north and 90 degrees is east.
    """

    sea_state_code: int = 6
    significant_wave_height_range_m: tuple[float, float] = (4.0, 6.0)
    significant_wave_height_m: float = 5.0

    storm_motion_speed_knots: float = 16.0
    storm_motion_direction_deg: float = 45.0
    motion_speed_jitter_fraction: float = 0.08
    motion_direction_jitter_deg: float = 3.0

    initial_storm_count: int = 3
    maximum_storm_count: int = 8
    cell_radius_nm_range: tuple[float, float] = (2.0, 5.0)
    cell_aspect_ratio_range: tuple[float, float] = (0.50, 0.90)
    cell_lifetime_minutes_range: tuple[float, float] = (30.0, 60.0)
    new_cell_interval_minutes_range: tuple[float, float] = (8.0, 15.0)
    shape_irregularity: float = 0.18

    def __post_init__(self) -> None:
        if self.sea_state_code < 0 or self.sea_state_code > 9:
            raise ValueError("sea_state_code must be between 0 and 9")
        self._validate_range(
            self.significant_wave_height_range_m,
            "significant_wave_height_range_m",
            minimum=0.0,
        )
        wave_min, wave_max = self.significant_wave_height_range_m
        if not wave_min <= self.significant_wave_height_m <= wave_max:
            raise ValueError(
                "significant_wave_height_m must lie inside its configured range"
            )
        if not math.isfinite(self.storm_motion_speed_knots) or self.storm_motion_speed_knots < 0:
            raise ValueError("storm_motion_speed_knots must be finite and non-negative")
        if not math.isfinite(self.storm_motion_direction_deg):
            raise ValueError("storm_motion_direction_deg must be finite")
        if not 0 <= self.motion_speed_jitter_fraction <= 1:
            raise ValueError("motion_speed_jitter_fraction must be between 0 and 1")
        if self.motion_direction_jitter_deg < 0:
            raise ValueError("motion_direction_jitter_deg must be non-negative")
        if self.initial_storm_count < 0:
            raise ValueError("initial_storm_count must be non-negative")
        if self.maximum_storm_count < self.initial_storm_count:
            raise ValueError("maximum_storm_count cannot be smaller than initial_storm_count")
        self._validate_range(self.cell_radius_nm_range, "cell_radius_nm_range", 0.0)
        self._validate_range(
            self.cell_aspect_ratio_range, "cell_aspect_ratio_range", 0.0
        )
        if self.cell_radius_nm_range[0] <= 0:
            raise ValueError("cell radii must be strictly positive")
        if self.cell_aspect_ratio_range[0] <= 0:
            raise ValueError("cell aspect ratios must be strictly positive")
        if self.cell_aspect_ratio_range[1] > 1.0:
            raise ValueError("cell aspect ratio cannot exceed 1")
        self._validate_range(
            self.cell_lifetime_minutes_range,
            "cell_lifetime_minutes_range",
            0.0,
        )
        self._validate_range(
            self.new_cell_interval_minutes_range,
            "new_cell_interval_minutes_range",
            0.0,
        )
        if self.cell_lifetime_minutes_range[0] <= 0:
            raise ValueError("cell lifetimes must be strictly positive")
        if self.new_cell_interval_minutes_range[0] <= 0:
            raise ValueError("new-cell intervals must be strictly positive")
        if not 0 <= self.shape_irregularity < 0.5:
            raise ValueError("shape_irregularity must be in [0, 0.5)")

    @staticmethod
    def _validate_range(
        value: tuple[float, float], name: str, minimum: float
    ) -> None:
        if len(value) != 2 or not all(math.isfinite(item) for item in value):
            raise ValueError(f"{name} must contain two finite values")
        if value[0] < minimum or value[0] > value[1]:
            raise ValueError(f"{name} must be an ordered range above {minimum}")
        if value[0] == value[1] == minimum:
            raise ValueError(f"{name} cannot be entirely zero")


@dataclass(frozen=True)
class SimulationParameters:
    """Complete configuration for a weather simulation run."""

    map_size_nm: tuple[float, float] = (50.0, 50.0)
    global_resolution_nm: float = 1.0
    local_resolution_nm: float = 0.1
    local_size_nm: tuple[float, float] = (10.0, 10.0)
    local_origin_nm: tuple[float, float] | None = None

    weather: WeatherParameters = field(default_factory=WeatherParameters)
    time_step_minutes: float = 1.0
    total_time_minutes: float = 60.0
    helicopter_initial_nm: tuple[float, float] = (5.0, 5.0)
    frigate_initial_nm: tuple[float, float] = (45.0, 45.0)
    random_seed: int | None = 20260917

    def __post_init__(self) -> None:
        self._validate_positive_pair(self.map_size_nm, "map_size_nm")
        self._validate_positive_pair(self.local_size_nm, "local_size_nm")
        if self.global_resolution_nm <= 0 or self.local_resolution_nm <= 0:
            raise ValueError("map resolutions must be positive")
        if self.local_resolution_nm >= self.global_resolution_nm:
            raise ValueError("local resolution must be finer than global resolution")
        if self.time_step_minutes <= 0 or not math.isfinite(self.time_step_minutes):
            raise ValueError("time_step_minutes must be positive and finite")
        if self.total_time_minutes <= 0 or not math.isfinite(self.total_time_minutes):
            raise ValueError("total_time_minutes must be positive and finite")
        step_count = self.total_time_minutes / self.time_step_minutes
        if not math.isclose(step_count, round(step_count), abs_tol=1e-9):
            raise ValueError("total time must be an integer multiple of the time step")
        for name, point in (
            ("helicopter_initial_nm", self.helicopter_initial_nm),
            ("frigate_initial_nm", self.frigate_initial_nm),
        ):
            self._validate_point(point, name)
        if self.local_origin_nm is not None:
            self._validate_point(self.local_origin_nm, "local_origin_nm", upper_edge=True)
            if (
                self.local_origin_nm[0] + self.local_size_nm[0] > self.map_size_nm[0]
                or self.local_origin_nm[1] + self.local_size_nm[1] > self.map_size_nm[1]
            ):
                raise ValueError("local window must fit inside the map")

    @staticmethod
    def _validate_positive_pair(value: tuple[float, float], name: str) -> None:
        if len(value) != 2 or any(not math.isfinite(item) or item <= 0 for item in value):
            raise ValueError(f"{name} must contain two positive finite values")

    def _validate_point(
        self,
        point: tuple[float, float],
        name: str,
        *,
        upper_edge: bool = False,
    ) -> None:
        if len(point) != 2 or not all(math.isfinite(item) for item in point):
            raise ValueError(f"{name} must contain two finite coordinates")
        comparison = (lambda coordinate, bound: coordinate <= bound) if upper_edge else (
            lambda coordinate, bound: coordinate < bound
        )
        if (
            point[0] < 0
            or point[1] < 0
            or not comparison(point[0], self.map_size_nm[0])
            or not comparison(point[1], self.map_size_nm[1])
        ):
            raise ValueError(f"{name} must lie inside the map")

    @property
    def step_count(self) -> int:
        return round(self.total_time_minutes / self.time_step_minutes)


@dataclass
class StormCell:
    """Continuous-space state of one simulated convective cell."""

    identifier: int
    center_x_nm: float
    center_y_nm: float
    major_radius_nm: float
    minor_radius_nm: float
    orientation_rad: float
    lifetime_minutes: float
    age_minutes: float
    speed_knots: float
    heading_deg: float
    phase_one: float
    phase_two: float

    @property
    def life_fraction(self) -> float:
        return min(1.0, max(0.0, self.age_minutes / self.lifetime_minutes))

    @property
    def lifecycle_stage(self) -> str:
        fraction = self.life_fraction
        if fraction < 0.30:
            return "developing"
        if fraction < 0.70:
            return "mature"
        return "dissipating"

    @property
    def scale(self) -> float:
        """Three-stage size envelope for growth, maturity and decay."""
        fraction = self.life_fraction
        if fraction < 0.30:
            return 0.20 + 0.80 * fraction / 0.30
        if fraction < 0.70:
            progress = (fraction - 0.30) / 0.40
            return 1.0 + 0.08 * math.sin(math.pi * progress)
        return max(0.0, (1.0 - fraction) / 0.30)


@dataclass(frozen=True)
class WeatherFrame:
    """One immutable output frame from :meth:`WeatherSystem.run`."""

    time_minutes: float
    storm_count: int
    weather_map: WeatherMapSnapshot


class WeatherSystem:
    """Generate and advance a binary thunderstorm occupancy simulation."""

    def __init__(self, parameters: SimulationParameters | None = None) -> None:
        self.parameters = parameters or SimulationParameters()
        self._rng = random.Random(self.parameters.random_seed)
        self._next_identifier = 1
        self._cells: list[StormCell] = []
        self.time_minutes = 0.0
        self._next_birth_minutes = math.inf

        local_origin = self.parameters.local_origin_nm or self._default_local_origin()
        self.weather_map = WeatherMap(
            area_size_nm=self.parameters.map_size_nm,
            global_resolution_nm=self.parameters.global_resolution_nm,
            local_resolution_nm=self.parameters.local_resolution_nm,
            local_size_nm=self.parameters.local_size_nm,
            local_origin_nm=local_origin,
        )
        self.reset()

    @property
    def cells(self) -> tuple[StormCell, ...]:
        """Return detached copies of the active cell states."""
        return tuple(replace(cell) for cell in self._cells)

    def _default_local_origin(self) -> tuple[float, float]:
        map_width, map_height = self.parameters.map_size_nm
        local_width, local_height = self.parameters.local_size_nm
        helicopter_x, helicopter_y = self.parameters.helicopter_initial_nm
        return (
            min(max(0.0, helicopter_x - local_width / 2.0), map_width - local_width),
            min(max(0.0, helicopter_y - local_height / 2.0), map_height - local_height),
        )

    def reset(self) -> WeatherFrame:
        """Reset time, regenerate initial cells and redraw the same shared map."""
        self._rng.seed(self.parameters.random_seed)
        self._next_identifier = 1
        self._cells.clear()
        self.time_minutes = 0.0

        protected = (
            self.parameters.helicopter_initial_nm,
            self.parameters.frigate_initial_nm,
        )
        for _ in range(self.parameters.weather.initial_storm_count):
            self._cells.append(self._new_cell(protected_points=protected))

        self._schedule_next_birth()
        self._render_to_map(protect_initial_positions=True)
        return self.current_frame()

    def _schedule_next_birth(self) -> None:
        low, high = self.parameters.weather.new_cell_interval_minutes_range
        self._next_birth_minutes = self.time_minutes + self._rng.uniform(low, high)

    def _new_cell(
        self,
        *,
        protected_points: tuple[tuple[float, float], ...] = (),
    ) -> StormCell:
        weather = self.parameters.weather
        map_width, map_height = self.parameters.map_size_nm
        minimum_radius, maximum_radius = weather.cell_radius_nm_range
        clearance = self.parameters.global_resolution_nm * math.sqrt(2.0)

        for _ in range(500):
            radius = self._rng.uniform(minimum_radius, maximum_radius)
            aspect = self._rng.uniform(*weather.cell_aspect_ratio_range)
            center_x = self._rng.uniform(0.0, map_width)
            center_y = self._rng.uniform(0.0, map_height)
            if any(
                math.hypot(center_x - point[0], center_y - point[1])
                <= radius * (1.0 + weather.shape_irregularity) + clearance
                for point in protected_points
            ):
                continue

            lifetime = self._rng.uniform(*weather.cell_lifetime_minutes_range)
            # Initial cells start at varied early/mature ages. Newly born cells
            # start at age zero and pass no protected points.
            initial_age = self._rng.uniform(0.0, lifetime * 0.55) if protected_points else 0.0
            cell = StormCell(
                identifier=self._next_identifier,
                center_x_nm=center_x,
                center_y_nm=center_y,
                major_radius_nm=radius,
                minor_radius_nm=radius * aspect,
                orientation_rad=self._rng.uniform(0.0, math.tau),
                lifetime_minutes=lifetime,
                age_minutes=initial_age,
                speed_knots=weather.storm_motion_speed_knots
                * self._rng.uniform(0.85, 1.15),
                heading_deg=(
                    weather.storm_motion_direction_deg
                    + self._rng.uniform(-12.0, 12.0)
                )
                % 360.0,
                phase_one=self._rng.uniform(0.0, math.tau),
                phase_two=self._rng.uniform(0.0, math.tau),
            )
            self._next_identifier += 1
            return cell

        raise RuntimeError(
            "could not place an initial thunderstorm away from protected positions; "
            "reduce storm count/radius or enlarge the map"
        )

    def step(self) -> WeatherFrame:
        """Advance exactly one configured time step and redraw the shared map."""
        if self.time_minutes >= self.parameters.total_time_minutes - 1e-9:
            raise StopIteration("the configured simulation time has been reached")

        dt = self.parameters.time_step_minutes
        weather = self.parameters.weather
        distance_noise_scale = math.sqrt(dt)

        for cell in self._cells:
            speed_noise = self._rng.gauss(
                0.0,
                weather.motion_speed_jitter_fraction * distance_noise_scale,
            )
            direction_noise = self._rng.gauss(
                0.0,
                weather.motion_direction_jitter_deg * distance_noise_scale,
            )
            effective_speed = max(0.0, cell.speed_knots * (1.0 + speed_noise))
            cell.heading_deg = (cell.heading_deg + direction_noise) % 360.0
            heading_rad = math.radians(cell.heading_deg)
            distance_nm = effective_speed * dt / 60.0
            cell.center_x_nm += distance_nm * math.sin(heading_rad)
            cell.center_y_nm += distance_nm * math.cos(heading_rad)
            cell.age_minutes += dt
            cell.orientation_rad += self._rng.gauss(0.0, 0.015 * distance_noise_scale)

        self.time_minutes += dt
        self._cells = [cell for cell in self._cells if self._cell_is_active(cell)]

        if (
            self.time_minutes + 1e-9 >= self._next_birth_minutes
            and len(self._cells) < weather.maximum_storm_count
        ):
            self._cells.append(self._new_cell())
            self._schedule_next_birth()

        self._render_to_map()
        return self.current_frame()

    def _cell_is_active(self, cell: StormCell) -> bool:
        if cell.age_minutes >= cell.lifetime_minutes:
            return False
        margin = cell.major_radius_nm * (1.0 + self.parameters.weather.shape_irregularity)
        width, height = self.parameters.map_size_nm
        return (
            cell.center_x_nm + margin >= 0
            and cell.center_y_nm + margin >= 0
            and cell.center_x_nm - margin <= width
            and cell.center_y_nm - margin <= height
        )

    def _contains_storm(self, x_nm: float, y_nm: float) -> bool:
        irregularity = self.parameters.weather.shape_irregularity
        for cell in self._cells:
            scale = cell.scale
            if scale <= 0:
                continue
            dx = x_nm - cell.center_x_nm
            dy = y_nm - cell.center_y_nm
            cosine = math.cos(cell.orientation_rad)
            sine = math.sin(cell.orientation_rad)
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy
            major = cell.major_radius_nm * scale
            minor = cell.minor_radius_nm * scale
            normalized_radius = math.hypot(local_x / major, local_y / minor)
            angle = math.atan2(local_y / minor, local_x / major)
            boundary = 1.0 + irregularity * (
                0.65 * math.sin(3.0 * angle + cell.phase_one)
                + 0.35 * math.sin(5.0 * angle + cell.phase_two)
            )
            if normalized_radius <= boundary:
                return True
        return False

    def _rasterize(
        self,
        *,
        origin_nm: tuple[float, float],
        resolution_nm: float,
        shape: tuple[int, int],
    ) -> list[list[int]]:
        origin_x, origin_y = origin_nm
        height, width = shape
        rows: list[list[int]] = []
        for row in range(height):
            y = origin_y + (row + 0.5) * resolution_nm
            rows.append(
                [
                    THUNDERSTORM
                    if self._contains_storm(
                        origin_x + (column + 0.5) * resolution_nm,
                        y,
                    )
                    else FREE
                    for column in range(width)
                ]
            )
        return rows

    def _render_to_map(self, *, protect_initial_positions: bool = False) -> None:
        global_values = self._rasterize(
            origin_nm=(0.0, 0.0),
            resolution_nm=self.weather_map.global_resolution_nm,
            shape=self.weather_map.global_grid.shape,
        )
        local_values = self._rasterize(
            origin_nm=self.weather_map.local_origin_nm,
            resolution_nm=self.weather_map.local_resolution_nm,
            shape=self.weather_map.local_grid.shape,
        )

        if protect_initial_positions:
            for point in (
                self.parameters.helicopter_initial_nm,
                self.parameters.frigate_initial_nm,
            ):
                row, column = self.weather_map.global_cell_at(*point)
                global_values[row][column] = FREE
                if self.weather_map.contains_local(*point):
                    row, column = self.weather_map.local_cell_at(*point)
                    local_values[row][column] = FREE

        # WeatherMap.update mutates both existing grids under one lock; systems
        # holding references to this map or either grid see the new frame.
        self.weather_map.update(
            global_values=global_values,
            local_values=local_values,
        )

    def current_frame(self) -> WeatherFrame:
        return WeatherFrame(
            time_minutes=self.time_minutes,
            storm_count=len(self._cells),
            weather_map=self.weather_map.snapshot(),
        )

    def run(self, *, include_initial: bool = True) -> Iterator[WeatherFrame]:
        """Yield the initial frame and every subsequent one through end time."""
        if include_initial:
            yield self.current_frame()
        while self.time_minutes < self.parameters.total_time_minutes - 1e-9:
            yield self.step()


__all__ = [
    "SimulationParameters",
    "StormCell",
    "WeatherFrame",
    "WeatherParameters",
    "WeatherSystem",
]
