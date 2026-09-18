"""Kinematic state for the two-speed helicopter model."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Helicopter:
    """A point-mass helicopter with a wait and a cruise speed mode.

    Heading is measured clockwise from north. Coordinates and distances use
    nautical miles, speed uses knots, and integration time uses seconds.
    """

    x_nm: float
    y_nm: float
    heading_deg: float = 0.0
    speed_knots: float = 0.0
    cruise_speed_knots: float = 50.0
    wait_speed_knots: float = 0.0

    def select_motion(self, *, moving: bool, heading_deg: float | None = None) -> None:
        if heading_deg is not None:
            if not math.isfinite(heading_deg):
                raise ValueError("heading_deg must be finite")
            self.heading_deg = heading_deg % 360.0
        self.speed_knots = self.cruise_speed_knots if moving else self.wait_speed_knots

    def step(self, duration_seconds: float) -> tuple[float, float]:
        if duration_seconds < 0 or not math.isfinite(duration_seconds):
            raise ValueError("duration_seconds must be finite and non-negative")
        distance_nm = self.speed_knots * duration_seconds / 3600.0
        heading_rad = math.radians(self.heading_deg)
        self.x_nm += distance_nm * math.sin(heading_rad)
        self.y_nm += distance_nm * math.cos(heading_rad)
        return self.x_nm, self.y_nm
