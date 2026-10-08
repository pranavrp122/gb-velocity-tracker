"""Synthetic frames that mimic OVITO renders of a migrating bicrystal.

The tests and the demo need frames whose boundary positions are known exactly.
Each frame has a black background, a rectangular sample filled with a dotted
green atom field, a white middle boundary band that moves right at a fixed
speed, an optional white side band at the right edge of the sample, and random
white specks standing in for thermally disordered atoms.
"""

from __future__ import annotations

import numpy as np

GREEN = (70, 220, 70)
DARK_GREEN = (20, 80, 20)
WHITE = (235, 235, 235)
ATOM_PITCH = 6
BAND_SIGMA = 1.5
SIDE_INSET = 6.0  # far enough from the sample edge that the band is not clipped


def sample_box(width: int, height: int) -> tuple[int, int, int, int]:
    """Return the sample rectangle as (x0, x1, y0, y1), half-open."""
    return int(0.08 * width), int(0.92 * width), int(0.12 * height), int(0.88 * height)


def band_positions(frame: int, width: int, v_px_per_frame: float,
                   side_v_px_per_frame: float = 0.0) -> tuple[float, float]:
    """True (middle_x, side_x) band centres in pixels for a given frame index."""
    x1 = sample_box(width, 1)[1]
    middle = width / 2 + v_px_per_frame * frame
    side = x1 - SIDE_INSET + side_v_px_per_frame * frame
    return middle, side


def _atom_field(width: int, height: int) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    x0, x1, y0, y1 = sample_box(width, height)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dots = ((xx % ATOM_PITCH) < 4) & ((yy % ATOM_PITCH) < 4)
    img[y0:y1, x0:x1] = np.where(dots[..., None], GREEN, DARK_GREEN)
    return img


def _paint_band(img: np.ndarray, centre: float, rng: np.random.Generator) -> None:
    # Coverage falls off smoothly with distance from the centre, so the band's
    # column profile has a single, sub-pixel-accurate maximum.
    x0, x1, y0, y1 = sample_box(img.shape[1], img.shape[0])
    cols = np.arange(x0, x1)
    coverage = np.exp(-0.5 * ((cols - centre) / BAND_SIGMA) ** 2)
    hit = rng.random((y1 - y0, cols.size)) < coverage
    img[y0:y1, x0:x1][hit] = WHITE


def _paint_specks(img: np.ndarray, fraction: float, rng: np.random.Generator) -> None:
    x0, x1, y0, y1 = sample_box(img.shape[1], img.shape[0])
    hit = rng.random((y1 - y0, x1 - x0)) < fraction
    img[y0:y1, x0:x1][hit] = WHITE


def make_frames(n_frames: int, width: int = 640, height: int = 360,
                v_px_per_frame: float = 1.0, speck_fraction: float = 0.05,
                seed: int = 0, side_gb: bool = True,
                side_v_px_per_frame: float = 0.0) -> list[np.ndarray]:
    """Generate `n_frames` RGB frames with boundaries at `band_positions`.

    The middle band starts at the image centre and moves right by
    `v_px_per_frame` each frame; once it passes the sample edge it is no
    longer drawn. The side band sits at the right edge of the sample and moves
    by `side_v_px_per_frame` (zero keeps it fixed).
    """
    rng = np.random.default_rng(seed)
    base = _atom_field(width, height)
    frames = []
    for f in range(n_frames):
        img = base.copy()
        middle, side = band_positions(f, width, v_px_per_frame, side_v_px_per_frame)
        _paint_specks(img, speck_fraction, rng)
        _paint_band(img, middle, rng)
        if side_gb:
            _paint_band(img, side, rng)
        frames.append(img)
    return frames


# Synthetic LAMMPS dump trajectories --------------------------------------

FCC_A = 3.52  # nickel lattice parameter, angstrom
DUMP_BOX = (140.8, 21.12, 21.12)
DUMP_COLUMNS = ["id", "type", "x", "y", "z", "c_pe", "v_ptm_type", "c_csym", "c_voro[1]"]
GRAIN_ANGLE_DEG = 26.57  # tilt about z, so lattice planes do not line up with x bins


def dump_band_positions(frame: int, box_x: float, v_A_per_frame: float,
                        side_v_A_per_frame: float = 0.0) -> tuple[float, float]:
    """True unwrapped (middle_x, side_x) slab centres in angstrom for a frame."""
    return box_x / 2 + v_A_per_frame * frame, side_v_A_per_frame * frame


def _rotated_fcc(box: tuple[float, float, float]) -> np.ndarray:
    lx, ly, lz = box
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * FCC_A
    reach = int(np.ceil(np.hypot(lx, ly) / FCC_A)) + 1
    span = np.arange(-reach, reach + 1)
    i, j, k = np.meshgrid(span, span, np.arange(round(lz / FCC_A)), indexing="ij")
    cells = np.stack([i, j, k], axis=-1).reshape(-1, 3) * FCC_A
    points = (cells[:, None, :] + basis[None]).reshape(-1, 3)
    t = np.radians(GRAIN_ANGLE_DEG)
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    points = points @ rot.T
    keep = (points[:, 0] >= 0) & (points[:, 0] < lx) & (points[:, 1] >= 0) & (points[:, 1] < ly)
    return points[keep]


def _slab(x: np.ndarray, centre: float, width: float, length: float) -> np.ndarray:
    d = x - centre
    return np.abs(d - length * np.round(d / length)) < width / 2


def make_dump_frames(n_frames: int, box: tuple[float, float, float] = DUMP_BOX,
                     v_A_per_frame: float = 0.35, slab_width_A: float = 4.0,
                     speck_fraction: float = 0.10, seed: int = 0, side_gb: bool = True,
                     side_v_A_per_frame: float = 0.0, steps_per_frame: int = 1000,
                     jitter_A: float = 0.1) -> list[tuple[int, list, np.ndarray]]:
    """(timestep, box bounds, data) frames with columns DUMP_COLUMNS.

    Atoms sit on a tilted FCC lattice with thermal jitter. Those inside the
    middle slab (moving +x by `v_A_per_frame`), inside the side slab at the
    periodic edge, or picked as random specks get PTM type 0 and a high
    centrosymmetry; everything else is FCC (type 1).
    """
    rng = np.random.default_rng(seed)
    lattice = _rotated_fcc(box)
    n = len(lattice)
    bounds = [(0.0, float(length)) for length in box]
    frames = []
    for f in range(n_frames):
        pos = np.mod(lattice + rng.normal(0, jitter_A, lattice.shape), box)
        middle, side = dump_band_positions(f, box[0], v_A_per_frame, side_v_A_per_frame)
        gb = _slab(pos[:, 0], middle, slab_width_A, box[0]) | (rng.random(n) < speck_fraction)
        if side_gb:
            gb |= _slab(pos[:, 0], side, slab_width_A, box[0])
        csym = np.where(gb, rng.uniform(10, 20, n), rng.uniform(0, 1.5, n))
        pe = -4.45 + 0.3 * gb + rng.normal(0, 0.02, n)
        data = np.column_stack([np.arange(1, n + 1), np.ones(n), pos, pe,
                                np.where(gb, 0, 1), csym, rng.normal(10.9, 0.2, n)])
        frames.append((f * steps_per_frame, bounds, data))
    return frames
