from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.frigate import Frigate
from model.helicopter import Helicopter


class VehicleModelTests(unittest.TestCase):
    def test_helicopter_wait_and_cruise(self) -> None:
        helicopter = Helicopter(1.0, 1.0)
        helicopter.select_motion(moving=False)
        self.assertEqual(helicopter.step(60.0), (1.0, 1.0))
        helicopter.select_motion(moving=True, heading_deg=90.0)
        helicopter.step(60.0)
        self.assertAlmostEqual(helicopter.x_nm, 1.0 + 50.0 / 60.0)
        self.assertAlmostEqual(helicopter.y_nm, 1.0)

    def test_frigate_known_future_position(self) -> None:
        frigate = Frigate(0.0, 0.0, speed_knots=30.0, heading_deg=0.0)
        self.assertEqual(frigate.position_after(120.0), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
