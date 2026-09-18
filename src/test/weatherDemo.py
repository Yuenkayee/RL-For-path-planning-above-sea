"""Generate an animated GIF for the default one-hour weather simulation.

The demo uses only the Python standard library.  Run it from the repository
root with::

    python3 src/test/weatherDemo.py

The resulting ``build/weatherSimu/weather_simulation.gif`` contains the
initial frame and sixty one-minute updates with the supplied default config.
The high-resolution local window is outlined in green and rendered over the
low-resolution global layer.  ``H`` and ``F`` mark the stationary helicopter
and frigate cells respectively.
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.weatherMap import THUNDERSTORM, WeatherMapSnapshot  # noqa: E402
from model.weatherSystem import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    SimulationParameters,
    WeatherFrame,
    WeatherSystem,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "build" / "weatherSimu" / "weather_simulation.gif"


_PALETTE = (
    (82, 155, 191),  # 0: clear ocean
    (190, 42, 48),  # 1: thunderstorm
    (203, 225, 232),  # 2: map grid
    (255, 196, 45),  # 3: helicopter
    (64, 72, 86),  # 4: frigate
    (10, 20, 30),  # 5: text and marker outlines
    (33, 210, 123),  # 6: high-resolution local-window border
    (248, 250, 251),  # 7: title background
)

_FONT = {
    " ": ("00000",) * 7,
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
}


class _Canvas:
    def __init__(self, width: int, height: int, fill: int = 7) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray([fill]) * (width * height)

    def set_pixel(self, x: int, y: int, color: int) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y * self.width + x] = color

    def rectangle(self, left: int, top: int, right: int, bottom: int, color: int) -> None:
        left = max(0, left)
        right = min(self.width, right)
        top = max(0, top)
        bottom = min(self.height, bottom)
        for y in range(top, bottom):
            start = y * self.width + left
            self.pixels[start : start + max(0, right - left)] = bytes([color]) * max(
                0, right - left
            )

    def line(self, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            twice_error = 2 * error
            if twice_error >= dy:
                error += dy
                x0 += sx
            if twice_error <= dx:
                error += dx
                y0 += sy

    def text(self, x: int, y: int, value: str, color: int = 5, scale: int = 1) -> None:
        cursor = x
        for character in value.upper():
            glyph = _FONT.get(character, _FONT[" "])
            for row, pattern in enumerate(glyph):
                for column, bit in enumerate(pattern):
                    if bit == "1":
                        self.rectangle(
                            cursor + column * scale,
                            y + row * scale,
                            cursor + (column + 1) * scale,
                            y + (row + 1) * scale,
                            color,
                        )
            cursor += 6 * scale


def _draw_layer(
    canvas: _Canvas,
    values: Sequence[Sequence[int]],
    *,
    origin_nm: tuple[float, float],
    resolution_nm: float,
    map_size_nm: tuple[float, float],
    plot_box: tuple[int, int, int, int],
) -> None:
    plot_left, plot_top, plot_width, plot_height = plot_box
    map_width, map_height = map_size_nm
    for row, cells in enumerate(values):
        y0_nm = origin_nm[1] + row * resolution_nm
        y1_nm = y0_nm + resolution_nm
        top = plot_top + round((map_height - y1_nm) / map_height * plot_height)
        bottom = plot_top + round((map_height - y0_nm) / map_height * plot_height)
        for column, state in enumerate(cells):
            x0_nm = origin_nm[0] + column * resolution_nm
            x1_nm = x0_nm + resolution_nm
            left = plot_left + round(x0_nm / map_width * plot_width)
            right = plot_left + round(x1_nm / map_width * plot_width)
            canvas.rectangle(left, top, right, bottom, 1 if state == THUNDERSTORM else 0)


def _world_to_pixel(
    point_nm: tuple[float, float],
    map_size_nm: tuple[float, float],
    plot_box: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, width, height = plot_box
    return (
        left + round(point_nm[0] / map_size_nm[0] * width),
        top + height - round(point_nm[1] / map_size_nm[1] * height),
    )


def _draw_marker(canvas: _Canvas, point: tuple[int, int], label: str, fill: int) -> None:
    x, y = point
    canvas.rectangle(x - 6, y - 6, x + 7, y + 7, 5)
    canvas.rectangle(x - 5, y - 5, x + 6, y + 6, fill)
    canvas.text(x - 2, y - 3, label, color=5)


def _global_cell_center(
    point_nm: tuple[float, float], snapshot: WeatherMapSnapshot
) -> tuple[float, float]:
    resolution = snapshot.global_resolution_nm
    column = min(int(point_nm[0] / resolution), len(snapshot.global_grid[0]) - 1)
    row = min(int(point_nm[1] / resolution), len(snapshot.global_grid) - 1)
    return (column + 0.5) * resolution, (row + 0.5) * resolution


def _render_frame(frame: WeatherFrame, parameters: SimulationParameters) -> tuple[int, int, bytes]:
    snapshot = frame.weather_map
    plot_size = 500
    header_height = 58
    margin = 10
    width = plot_size + margin * 2
    height = plot_size + header_height + margin
    plot_box = (margin, header_height, plot_size, plot_size)
    canvas = _Canvas(width, height)

    _draw_layer(
        canvas,
        snapshot.global_grid,
        origin_nm=(0.0, 0.0),
        resolution_nm=snapshot.global_resolution_nm,
        map_size_nm=snapshot.area_size_nm,
        plot_box=plot_box,
    )
    _draw_layer(
        canvas,
        snapshot.local_grid,
        origin_nm=snapshot.local_origin_nm,
        resolution_nm=snapshot.local_resolution_nm,
        map_size_nm=snapshot.area_size_nm,
        plot_box=plot_box,
    )

    # Five-nautical-mile reference grid.
    for coordinate in range(0, round(snapshot.area_size_nm[0]) + 1, 5):
        x, _ = _world_to_pixel((coordinate, 0.0), snapshot.area_size_nm, plot_box)
        canvas.line(x, plot_box[1], x, plot_box[1] + plot_box[3], 2)
    for coordinate in range(0, round(snapshot.area_size_nm[1]) + 1, 5):
        _, y = _world_to_pixel((0.0, coordinate), snapshot.area_size_nm, plot_box)
        canvas.line(plot_box[0], y, plot_box[0] + plot_box[2], y, 2)

    # Local high-resolution window border.
    local_left, local_bottom = _world_to_pixel(
        snapshot.local_origin_nm, snapshot.area_size_nm, plot_box
    )
    local_right, local_top = _world_to_pixel(
        (
            snapshot.local_origin_nm[0] + snapshot.local_size_nm[0],
            snapshot.local_origin_nm[1] + snapshot.local_size_nm[1],
        ),
        snapshot.area_size_nm,
        plot_box,
    )
    for offset in range(2):
        canvas.line(local_left + offset, local_top, local_right, local_top, 6)
        canvas.line(local_left + offset, local_bottom, local_right, local_bottom, 6)
        canvas.line(local_left, local_top + offset, local_left, local_bottom, 6)
        canvas.line(local_right, local_top + offset, local_right, local_bottom, 6)

    helicopter_pixel = _world_to_pixel(
        _global_cell_center(parameters.helicopter_initial_nm, snapshot),
        snapshot.area_size_nm,
        plot_box,
    )
    frigate_pixel = _world_to_pixel(
        _global_cell_center(parameters.frigate_initial_nm, snapshot),
        snapshot.area_size_nm,
        plot_box,
    )
    _draw_marker(canvas, helicopter_pixel, "H", 3)
    _draw_marker(canvas, frigate_pixel, "F", 4)

    canvas.text(
        margin,
        8,
        f"T={frame.time_minutes:02.0f} MIN  STORMS={frame.storm_count}",
        scale=2,
    )
    canvas.text(margin, 34, "H HELICOPTER  F FRIGATE", scale=1)
    return width, height, bytes(canvas.pixels)


class _BitWriter:
    def __init__(self) -> None:
        self.output = bytearray()
        self.buffer = 0
        self.bit_count = 0

    def write(self, code: int, width: int) -> None:
        self.buffer |= code << self.bit_count
        self.bit_count += width
        while self.bit_count >= 8:
            self.output.append(self.buffer & 0xFF)
            self.buffer >>= 8
            self.bit_count -= 8

    def finish(self) -> bytes:
        if self.bit_count:
            self.output.append(self.buffer & 0xFF)
        return bytes(self.output)


def _lzw_compress(indices: bytes, minimum_code_size: int = 3) -> bytes:
    """Encode one indexed-color frame using the GIF LZW bit ordering."""
    clear_code = 1 << minimum_code_size
    end_code = clear_code + 1
    dictionary = {bytes([value]): value for value in range(clear_code)}
    next_code = end_code + 1
    code_width = minimum_code_size + 1
    writer = _BitWriter()
    writer.write(clear_code, code_width)

    prefix = bytes([indices[0]])
    for value in indices[1:]:
        candidate = prefix + bytes([value])
        if candidate in dictionary:
            prefix = candidate
            continue

        writer.write(dictionary[prefix], code_width)
        if next_code == (1 << code_width) and code_width < 12:
            code_width += 1
        if next_code < 4096:
            dictionary[candidate] = next_code
            next_code += 1
        else:
            writer.write(clear_code, code_width)
            dictionary = {bytes([item]): item for item in range(clear_code)}
            next_code = end_code + 1
            code_width = minimum_code_size + 1
        prefix = bytes([value])

    writer.write(dictionary[prefix], code_width)
    if next_code == (1 << code_width) and code_width < 12:
        code_width += 1
    writer.write(end_code, code_width)
    return writer.finish()


def _sub_blocks(data: bytes) -> bytes:
    blocks = bytearray()
    for offset in range(0, len(data), 255):
        block = data[offset : offset + 255]
        blocks.append(len(block))
        blocks.extend(block)
    blocks.append(0)
    return bytes(blocks)


def _write_animated_gif(
    path: Path,
    frames: Iterable[tuple[int, int, bytes]],
    *,
    frame_duration_ms: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    first_width, first_height, first_pixels = next(iterator)
    all_frames = [(first_width, first_height, first_pixels), *iterator]
    if any(width != first_width or height != first_height for width, height, _ in all_frames):
        raise ValueError("all GIF frames must have identical dimensions")

    output = bytearray(b"GIF89a")
    output.extend(struct.pack("<HH", first_width, first_height))
    output.extend(bytes([0xF2, 0x00, 0x00]))  # 8-entry global color table.
    for red, green, blue in _PALETTE:
        output.extend(bytes([red, green, blue]))
    output.extend(b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00")

    delay = max(1, round(frame_duration_ms / 10))
    for width, height, pixels in all_frames:
        output.extend(b"\x21\xf9\x04\x04")
        output.extend(struct.pack("<H", delay))
        output.extend(b"\x00\x00")
        output.extend(b"\x2c\x00\x00\x00\x00")
        output.extend(struct.pack("<HH", width, height))
        output.append(0x00)
        output.append(3)
        output.extend(_sub_blocks(_lzw_compress(pixels)))
    output.append(0x3B)
    path.write_bytes(output)
    return len(all_frames)


def create_weather_animation(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    parameters: SimulationParameters | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    frame_duration_ms: int = 160,
) -> Path:
    """Run a simulation and save all frames as a looping animated GIF."""
    configuration = parameters or SimulationParameters.from_json(config_path)
    system = WeatherSystem(configuration)
    output = Path(output_path).expanduser().resolve()
    rendered_frames = (
        _render_frame(frame, configuration) for frame in system.run(include_initial=True)
    )
    frame_count = _write_animated_gif(
        output,
        rendered_frames,
        frame_duration_ms=frame_duration_ms,
    )
    expected_count = configuration.step_count + 1
    if frame_count != expected_count:
        raise AssertionError(f"expected {expected_count} frames, wrote {frame_count}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"output GIF path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"weather JSON path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=160,
        help="display duration of each simulation frame in milliseconds",
    )
    arguments = parser.parse_args()
    output = create_weather_animation(
        arguments.output,
        config_path=arguments.config,
        frame_duration_ms=arguments.frame_ms,
    )
    print(f"Created {output} using {Path(arguments.config).resolve()}.")


if __name__ == "__main__":
    main()
