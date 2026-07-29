"""
tools/make_floor.py

Generates the floor texture used by the simulator: a dark, seamlessly tiling
grid in the style of Isaac Sim / Gazebo lab floors, rather than PyBullet's
default blue-and-white checkerboard.

The texture tiles, so every line is drawn with wrap-around (modulo) indexing --
otherwise the seam between tiles shows up as a visible double or missing line.

Run it directly to regenerate the asset:

    python tools/make_floor.py
"""
import os

import numpy as np
from PIL import Image

ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
FLOOR_PNG = os.path.join(ASSETS, "floor_grid.png")

SIZE = 1024
MINOR_EVERY = 128
MAJOR_EVERY = 512

BASE = (0.155, 0.169, 0.192)
MINOR = (0.235, 0.255, 0.290)
MAJOR = (0.340, 0.375, 0.425)
SPECK = 0.007


def _draw_lines(img, spacing, colour, width):
    """Draw grid lines every ``spacing`` px, wrapping around the edges."""
    half = width // 2
    for base in range(0, SIZE, spacing):
        for off in range(-half, half + 1):
            i = (base + off) % SIZE
            img[i, :, :] = colour
            img[:, i, :] = colour


def make_floor(path=FLOOR_PNG, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    img[:, :] = BASE

    img += rng.normal(0.0, SPECK, (SIZE, SIZE, 1)).astype(np.float32)

    _draw_lines(img, MINOR_EVERY, MINOR, 2)
    _draw_lines(img, MAJOR_EVERY, MAJOR, 4)

    img = np.clip(img, 0.0, 1.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path


if __name__ == "__main__":
    print("wrote", make_floor())
