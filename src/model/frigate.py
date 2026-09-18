"""Constant-course frigate kinematics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Frigate:
    """A point-mass frigate moving at a known constant speed and heading."""

    x_nm: float
    y_nm: float
    speed_knots: float = 30.0
    heading_deg: float = 0.0

    def position_after(self, duration_seconds: float) -> tuple[float, float]:
        if duration_seconds < 0 or not math.isfinite(duration_seconds):
            raise ValueError("duration_seconds must be finite and non-negative")
        distance_nm = self.speed_knots * duration_seconds / 3600.0
        heading_rad = math.radians(self.heading_deg)
        return (
            self.x_nm + distance_nm * math.sin(heading_rad),
            self.y_nm + distance_nm * math.cos(heading_rad),
        )

    def step(self, duration_seconds: float) -> tuple[float, float]:
        self.x_nm, self.y_nm = self.position_after(duration_seconds)
        return self.x_nm, self.y_nm
