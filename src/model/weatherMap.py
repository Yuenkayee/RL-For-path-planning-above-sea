"""Shared two-resolution weather occupancy map.

The map contains two binary raster layers:

* ``global_grid`` covers the complete operating area at low resolution.
* ``local_grid`` covers a movable sub-area at high resolution.

Both layers use row-major indexing ``grid[row, column]``.  A value of ``0``
means that the cell is free of thunderstorms and ``1`` means that a
thunderstorm occupies the cell.  Rows increase with the world ``y``
coordinate and columns increase with the world ``x`` coordinate.

The same :class:`WeatherMap` instance is intended to be shared by the weather
system and the planner.  Grid objects keep their identity even when the local
window is resized, so code holding ``weather_map.local_grid`` continues to
refer to the active local layer.  Use :meth:`WeatherMap.snapshot` when a
planner needs an immutable, internally consistent view while the weather
system continues updating the live map.

Only storage, coordinate conversion and update mechanics live here.  Weather
generation, motion and prediction belong in the weather-system module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from threading import RLock
from typing import Callable, Iterable, Iterator, Sequence


FREE = 0
THUNDERSTORM = 1
_VALID_CELL_VALUES = (FREE, THUNDERSTORM)
_FLOAT_TOLERANCE = 1e-9


def _validate_binary(value: int) -> int:
    """Return a normalized binary value or raise a descriptive error."""
    if value not in _VALID_CELL_VALUES:
        raise ValueError(
            f"weather cells must be {FREE} (free) or {THUNDERSTORM} "
            f"(thunderstorm), got {value!r}"
        )
    return int(value)


def _cell_count(length_nm: float, resolution_nm: float, name: str) -> int:
    if not math.isfinite(length_nm) or length_nm <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    if not math.isfinite(resolution_nm) or resolution_nm <= 0:
        raise ValueError("grid resolution must be a positive finite value")

    count = length_nm / resolution_nm
    rounded = round(count)
    if not math.isclose(count, rounded, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(
            f"{name} ({length_nm}) must be an integer multiple of the "
            f"resolution ({resolution_nm})"
        )
    return int(rounded)


class _GridRow:
    """A mutable row view returned by ``BinaryGrid[row]``."""

    def __init__(self, grid: "BinaryGrid", row: int) -> None:
        self._grid = grid
        self._row = row

    def __len__(self) -> int:
        return self._grid.width

    def __getitem__(self, column: int | slice) -> int | list[int]:
        if isinstance(column, slice):
            return [self[index] for index in range(*column.indices(len(self)))]
        return self._grid[self._row, column]

    def __setitem__(self, column: int | slice, value: int | Iterable[int]) -> None:
        if isinstance(column, slice):
            indices = list(range(*column.indices(len(self))))
            values = list(value)  # type: ignore[arg-type]
            if len(indices) != len(values):
                raise ValueError("slice assignment cannot change a grid row's size")
            with self._grid.transaction():
                for index, item in zip(indices, values):
                    self._grid[self._row, index] = item
            return
        self._grid[self._row, column] = value  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[int]:
        for column in range(len(self)):
            yield self[column]

    def __repr__(self) -> str:
        return repr(list(self))


class BinaryGrid:
    """A compact, mutable 2-D grid that accepts only the values 0 and 1.

    The object is deliberately array-like: both ``grid[row, column]`` and
    ``grid[row][column]`` are supported.  Mutations are protected by the map's
    shared re-entrant lock and notify the owning :class:`WeatherMap`.
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        fill: int = FREE,
        lock: RLock | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._lock = lock or RLock()
        self._on_change = on_change
        self._height = 0
        self._width = 0
        self._cells = bytearray()
        self._set_shape(height, width, fill)

    @staticmethod
    def _validate_dimension(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _set_shape(self, height: int, width: int, fill: int) -> None:
        height = self._validate_dimension(height, "height")
        width = self._validate_dimension(width, "width")
        fill = _validate_binary(fill)
        self._height = height
        self._width = width
        self._cells = bytearray([fill]) * (height * width)

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def shape(self) -> tuple[int, int]:
        return self._height, self._width

    def __len__(self) -> int:
        return self._height

    @staticmethod
    def _normalize_index(index: int, size: int, name: str) -> int:
        if not isinstance(index, int):
            raise TypeError(f"{name} index must be an integer")
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(f"{name} index out of range")
        return index

    def _offset(self, row: int, column: int) -> int:
        row = self._normalize_index(row, self._height, "row")
        column = self._normalize_index(column, self._width, "column")
        return row * self._width + column

    def __getitem__(self, key: int | tuple[int, int]) -> _GridRow | int:
        with self._lock:
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise IndexError("grid index must contain row and column")
                return self._cells[self._offset(key[0], key[1])]
            row = self._normalize_index(key, self._height, "row")
            return _GridRow(self, row)

    def __setitem__(
        self, key: int | tuple[int, int], value: int | Sequence[int]
    ) -> None:
        with self._lock:
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise IndexError("grid index must contain row and column")
                normalized = _validate_binary(value)  # type: ignore[arg-type]
                offset = self._offset(key[0], key[1])
                if self._cells[offset] != normalized:
                    self._cells[offset] = normalized
                    self._changed()
                return

            row = self._normalize_index(key, self._height, "row")
            values = [_validate_binary(item) for item in value]  # type: ignore[union-attr]
            if len(values) != self._width:
                raise ValueError(f"row must contain exactly {self._width} cells")
            start = row * self._width
            replacement = bytearray(values)
            if self._cells[start : start + self._width] != replacement:
                self._cells[start : start + self._width] = replacement
                self._changed()

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    @contextmanager
    def transaction(self) -> Iterator["BinaryGrid"]:
        """Hold the grid lock while applying a group of cell mutations."""
        with self._lock:
            yield self

    @staticmethod
    def normalize_rows(
        rows: Iterable[Iterable[int]], expected_shape: tuple[int, int] | None = None
    ) -> tuple[tuple[int, ...], ...]:
        normalized = tuple(tuple(_validate_binary(cell) for cell in row) for row in rows)
        if not normalized or not normalized[0]:
            raise ValueError("grid values must not be empty")
        width = len(normalized[0])
        if any(len(row) != width for row in normalized):
            raise ValueError("all grid rows must have the same length")
        if expected_shape is not None and (len(normalized), width) != expected_shape:
            raise ValueError(
                f"grid shape must be {expected_shape}, got {(len(normalized), width)}"
            )
        return normalized

    def replace(self, rows: Iterable[Iterable[int]]) -> None:
        """Replace every cell without replacing the ``BinaryGrid`` object."""
        normalized = self.normalize_rows(rows, self.shape)
        flat = bytearray(cell for row in normalized for cell in row)
        with self._lock:
            if self._cells != flat:
                self._cells[:] = flat
                self._changed()

    def resize(self, height: int, width: int, *, fill: int = FREE) -> None:
        """Resize in place so references to this grid object remain valid."""
        height = self._validate_dimension(height, "height")
        width = self._validate_dimension(width, "width")
        fill = _validate_binary(fill)
        with self._lock:
            if self.shape == (height, width):
                self.fill(fill)
                return
            self._set_shape(height, width, fill)
            self._changed()

    def fill(self, value: int) -> None:
        value = _validate_binary(value)
        replacement = bytearray([value]) * (self._height * self._width)
        with self._lock:
            if self._cells != replacement:
                self._cells[:] = replacement
                self._changed()

    def to_rows(self) -> tuple[tuple[int, ...], ...]:
        """Return an immutable copy suitable for planning or serialization."""
        with self._lock:
            return tuple(
                tuple(self._cells[start : start + self._width])
                for start in range(0, len(self._cells), self._width)
            )

    def __iter__(self) -> Iterator[_GridRow]:
        for row in range(self._height):
            yield _GridRow(self, row)

    def __repr__(self) -> str:
        return f"BinaryGrid(shape={self.shape})"


@dataclass(frozen=True)
class WeatherMapSnapshot:
    """Immutable, same-version view of both weather layers."""

    version: int
    area_size_nm: tuple[float, float]
    global_resolution_nm: float
    local_resolution_nm: float
    local_origin_nm: tuple[float, float]
    local_size_nm: tuple[float, float]
    global_grid: tuple[tuple[int, ...], ...]
    local_grid: tuple[tuple[int, ...], ...]


class WeatherMap:
    """Mutable low-resolution global map plus high-resolution local window."""

    def __init__(
        self,
        *,
        area_size_nm: tuple[float, float] = (50.0, 50.0),
        global_resolution_nm: float = 1.0,
        local_resolution_nm: float = 0.1,
        local_size_nm: tuple[float, float] = (10.0, 10.0),
        local_origin_nm: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._lock = RLock()
        self._version = 0
        self._transaction_depth = 0
        self._transaction_dirty = False

        self.area_size_nm = self._validate_pair(
            area_size_nm, "area_size_nm", positive=True
        )
        self.global_resolution_nm = self._validate_resolution(global_resolution_nm)
        self.local_resolution_nm = self._validate_resolution(local_resolution_nm)
        if self.local_resolution_nm >= self.global_resolution_nm:
            raise ValueError(
                "local_resolution_nm must be smaller than global_resolution_nm"
            )

        global_width = _cell_count(
            self.area_size_nm[0], self.global_resolution_nm, "area width"
        )
        global_height = _cell_count(
            self.area_size_nm[1], self.global_resolution_nm, "area height"
        )

        self.local_origin_nm, self.local_size_nm = self._validate_local_window(
            local_origin_nm, local_size_nm
        )
        local_width, local_height = self._local_grid_dimensions(self.local_size_nm)

        self.global_grid = BinaryGrid(
            global_height,
            global_width,
            lock=self._lock,
            on_change=self._mark_changed,
        )
        self.local_grid = BinaryGrid(
            local_height,
            local_width,
            lock=self._lock,
            on_change=self._mark_changed,
        )

    @staticmethod
    def _validate_resolution(value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("resolution must be a positive finite value")
        return float(value)

    @staticmethod
    def _validate_pair(
        value: tuple[float, float], name: str, *, positive: bool = False
    ) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        pair = float(value[0]), float(value[1])
        if not all(math.isfinite(item) for item in pair):
            raise ValueError(f"{name} values must be finite")
        if positive and any(item <= 0 for item in pair):
            raise ValueError(f"{name} values must be positive")
        return pair

    def _validate_local_window(
        self,
        origin_nm: tuple[float, float],
        size_nm: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        origin = self._validate_pair(origin_nm, "local_origin_nm")
        size = self._validate_pair(size_nm, "local_size_nm", positive=True)
        if origin[0] < 0 or origin[1] < 0:
            raise ValueError("local window origin must lie inside the global area")
        if (
            origin[0] + size[0] > self.area_size_nm[0] + _FLOAT_TOLERANCE
            or origin[1] + size[1] > self.area_size_nm[1] + _FLOAT_TOLERANCE
        ):
            raise ValueError("local window must be fully contained in the global area")
        self._local_grid_dimensions(size)
        return origin, size

    def _local_grid_dimensions(self, size_nm: tuple[float, float]) -> tuple[int, int]:
        width = _cell_count(size_nm[0], self.local_resolution_nm, "local width")
        height = _cell_count(size_nm[1], self.local_resolution_nm, "local height")
        return width, height

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def _mark_changed(self) -> None:
        if self._transaction_depth:
            self._transaction_dirty = True
        else:
            self._version += 1

    @contextmanager
    def transaction(self) -> Iterator["WeatherMap"]:
        """Group several mutations into one atomic, single-version update."""
        with self._lock:
            self._transaction_depth += 1
            try:
                yield self
            finally:
                self._transaction_depth -= 1
                if self._transaction_depth == 0 and self._transaction_dirty:
                    self._version += 1
                    self._transaction_dirty = False

    def replace_global(self, values: Iterable[Iterable[int]]) -> None:
        """Replace all low-resolution cells in place."""
        self.global_grid.replace(values)

    def replace_local(self, values: Iterable[Iterable[int]]) -> None:
        """Replace all high-resolution cells in the current window in place."""
        self.local_grid.replace(values)

    def update(
        self,
        *,
        global_values: Iterable[Iterable[int]] | None = None,
        local_values: Iterable[Iterable[int]] | None = None,
        local_origin_nm: tuple[float, float] | None = None,
        local_size_nm: tuple[float, float] | None = None,
    ) -> None:
        """Atomically update one or both layers and optionally move the window.

        When the local window changes, ``local_values`` must describe its new
        high-resolution contents.  If omitted, the local layer is initialized
        by expanding the currently stored global cells into the new window.
        """
        prepared_global = None
        if global_values is not None:
            prepared_global = BinaryGrid.normalize_rows(
                global_values, self.global_grid.shape
            )

        window_changes = local_origin_nm is not None or local_size_nm is not None
        new_origin = local_origin_nm or self.local_origin_nm
        new_size = local_size_nm or self.local_size_nm
        new_origin, new_size = self._validate_local_window(new_origin, new_size)
        new_width, new_height = self._local_grid_dimensions(new_size)

        prepared_local = None
        if local_values is not None:
            prepared_local = BinaryGrid.normalize_rows(
                local_values, (new_height, new_width)
            )

        with self.transaction():
            if prepared_global is not None:
                self.global_grid.replace(prepared_global)

            if window_changes:
                old_origin = self.local_origin_nm
                old_size = self.local_size_nm
                self.local_origin_nm = new_origin
                self.local_size_nm = new_size
                if (old_origin, old_size) != (new_origin, new_size):
                    self._mark_changed()
                if self.local_grid.shape != (new_height, new_width):
                    self.local_grid.resize(new_height, new_width)

            if prepared_local is not None:
                self.local_grid.replace(prepared_local)
            elif window_changes:
                self.local_grid.replace(self._expanded_global_window())

    def move_local_window(
        self,
        origin_nm: tuple[float, float],
        *,
        values: Iterable[Iterable[int]] | None = None,
        size_nm: tuple[float, float] | None = None,
    ) -> None:
        """Move or resize the high-resolution window without replacing it.

        If detailed values are unavailable, each new local cell inherits the
        value of the low-resolution global cell that contains its center.
        """
        self.update(
            local_values=values,
            local_origin_nm=origin_nm,
            local_size_nm=size_nm,
        )

    def _expanded_global_window(self) -> tuple[tuple[int, ...], ...]:
        rows: list[tuple[int, ...]] = []
        origin_x, origin_y = self.local_origin_nm
        for row in range(self.local_grid.height):
            y = origin_y + (row + 0.5) * self.local_resolution_nm
            output_row = []
            for column in range(self.local_grid.width):
                x = origin_x + (column + 0.5) * self.local_resolution_nm
                global_row, global_column = self.global_cell_at(x, y)
                output_row.append(self.global_grid[global_row, global_column])
            rows.append(tuple(output_row))
        return tuple(rows)

    def global_cell_at(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        """Convert world coordinates to ``(row, column)`` in the global grid."""
        self._validate_world_point(x_nm, y_nm)
        return (
            min(int(y_nm / self.global_resolution_nm), self.global_grid.height - 1),
            min(int(x_nm / self.global_resolution_nm), self.global_grid.width - 1),
        )

    def local_cell_at(self, x_nm: float, y_nm: float) -> tuple[int, int]:
        """Convert a point inside the local window to local grid indices."""
        if not self.contains_local(x_nm, y_nm):
            raise ValueError("point lies outside the high-resolution local window")
        origin_x, origin_y = self.local_origin_nm
        return (
            min(
                int((y_nm - origin_y) / self.local_resolution_nm),
                self.local_grid.height - 1,
            ),
            min(
                int((x_nm - origin_x) / self.local_resolution_nm),
                self.local_grid.width - 1,
            ),
        )

    def _validate_world_point(self, x_nm: float, y_nm: float) -> None:
        if not (math.isfinite(x_nm) and math.isfinite(y_nm)):
            raise ValueError("world coordinates must be finite")
        if not (
            0 <= x_nm < self.area_size_nm[0]
            and 0 <= y_nm < self.area_size_nm[1]
        ):
            raise ValueError("point lies outside the global weather area")

    def contains_local(self, x_nm: float, y_nm: float) -> bool:
        origin_x, origin_y = self.local_origin_nm
        width, height = self.local_size_nm
        return (
            origin_x <= x_nm < origin_x + width
            and origin_y <= y_nm < origin_y + height
        )

    def weather_at(self, x_nm: float, y_nm: float) -> int:
        """Read the finest available weather value at a world coordinate."""
        with self._lock:
            self._validate_world_point(x_nm, y_nm)
            if self.contains_local(x_nm, y_nm):
                row, column = self.local_cell_at(x_nm, y_nm)
                return self.local_grid[row, column]
            row, column = self.global_cell_at(x_nm, y_nm)
            return self.global_grid[row, column]

    def set_weather_at(self, x_nm: float, y_nm: float, value: int) -> None:
        """Write to the finest layer available at a world coordinate."""
        value = _validate_binary(value)
        with self._lock:
            self._validate_world_point(x_nm, y_nm)
            if self.contains_local(x_nm, y_nm):
                row, column = self.local_cell_at(x_nm, y_nm)
                self.local_grid[row, column] = value
            else:
                row, column = self.global_cell_at(x_nm, y_nm)
                self.global_grid[row, column] = value

    def clear(self) -> None:
        """Set every cell in both layers to ``FREE`` atomically."""
        with self.transaction():
            self.global_grid.fill(FREE)
            self.local_grid.fill(FREE)

    def snapshot(self) -> WeatherMapSnapshot:
        """Copy both layers under one lock for a consistent planner input."""
        with self._lock:
            return WeatherMapSnapshot(
                version=self._version,
                area_size_nm=self.area_size_nm,
                global_resolution_nm=self.global_resolution_nm,
                local_resolution_nm=self.local_resolution_nm,
                local_origin_nm=self.local_origin_nm,
                local_size_nm=self.local_size_nm,
                global_grid=self.global_grid.to_rows(),
                local_grid=self.local_grid.to_rows(),
            )


__all__ = [
    "FREE",
    "THUNDERSTORM",
    "BinaryGrid",
    "WeatherMap",
    "WeatherMapSnapshot",
]
