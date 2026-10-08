"""Follow the middle and side grain boundaries through a frame series.

Dump mode (track_dump, the primary path) works on LAMMPS trajectories, in
angstrom along the periodic simulation axis. Image mode (track_boundaries)
works on rendered OVITO snapshots, in pixels. Both follow the same scheme.

The first frame is searched globally for the two strongest boundary peaks.
Every later frame is searched only within a small window around each
boundary's previous position, which keeps a boundary from jumping to another
band or to a cluster of thermal specks. A boundary is declared lost (NaN from
then on) when its peak no longer rises clearly above the speck background,
which is what happens once it migrates out of the sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Union

import numpy as np

from .detect import (boundary_mask, column_profile, find_boundaries, load_frame,
                     refine_peak, sample_columns, sample_rows)
from .dump import axis_positions, iter_frames
from .gbatoms import boundary_atoms
from .profile import contrast, position_profile
from .profile import find_boundaries as find_profile_peaks

Frame = Union[np.ndarray, str, Path]


def frame_profile(frame: Frame, white_thresh: int = 170) -> tuple[np.ndarray, tuple[int, int]]:
    """Column profile of one frame plus the column range the sample occupies."""
    img = frame if isinstance(frame, np.ndarray) else load_frame(frame)
    mask = boundary_mask(img, white_thresh)
    return column_profile(mask, sample_rows(img)), sample_columns(img)


def _contrast(profile: np.ndarray, x: float, cols: tuple[int, int]) -> float:
    """Peak height above the median column, i.e. above the speck background."""
    return float(profile[int(round(x))] - np.median(profile[cols[0]:cols[1]]))


def _initial(profile: np.ndarray, cols: tuple[int, int]) -> list[float]:
    """Label the two strongest peaks: middle is the one nearest the sample centre."""
    peaks = find_boundaries(profile, n=2)
    centre = (cols[0] + cols[1]) / 2
    peaks.sort(key=lambda x: abs(x - centre))
    return (peaks + [np.nan, np.nan])[:2]


def _follow(profile: np.ndarray, prev: float, window: float,
            cols: tuple[int, int], min_contrast: float) -> float:
    """Best peak within +/- window of prev, or NaN if it is gone."""
    lo = max(cols[0], int(np.floor(prev - window)))
    hi = min(cols[1], int(np.ceil(prev + window)) + 1)
    if hi - lo < 1:
        return np.nan
    i = lo + int(np.argmax(profile[lo:hi]))
    if _contrast(profile, i, cols) < min_contrast:
        return np.nan
    return refine_peak(profile, i)


def track_boundaries(frames: Iterable[Frame], search_window_px: float = 10.0,
                     white_thresh: int = 170,
                     min_contrast_ratio: float = 0.35) -> np.ndarray:
    """Return an (n_frames, 2) array of [middle_x, side_x] positions in pixels.

    Frames may be arrays or image paths; paths are loaded one at a time so long
    series never have to fit in memory. A boundary counts as lost when its peak
    contrast drops below `min_contrast_ratio` of its frame-0 contrast, and once
    lost it stays NaN.
    """
    rows: list[list[float]] = []
    floors: list[float] = []
    for frame in frames:
        profile, cols = frame_profile(frame, white_thresh)
        if not rows:
            first = _initial(profile, cols)
            floors = [min_contrast_ratio * _contrast(profile, x, cols)
                      if np.isfinite(x) else np.inf for x in first]
            rows.append(first)
            continue
        rows.append([_follow(profile, prev, search_window_px, cols, floor)
                     if np.isfinite(prev) else np.nan
                     for prev, floor in zip(rows[-1], floors)])
    return np.array(rows, dtype=float).reshape(-1, 2)


def displacement_px(positions: np.ndarray) -> np.ndarray:
    """Signed displacement of each boundary relative to frame 0 (px or angstrom)."""
    positions = np.asarray(positions, dtype=float)
    return positions - positions[0]


# Dump mode: positions in angstrom on a periodic axis ---------------------

DumpProfile = tuple[int, np.ndarray, np.ndarray, float, float]


def wrap(x: float, lo: float, length: float) -> float:
    """Map a coordinate into [lo, lo + length)."""
    return lo + float(np.mod(x - lo, length))


def min_image(d: float, length: float) -> float:
    """Shortest signed separation on a periodic axis of the given length."""
    return d - length * round(d / length)


def dump_profiles(path: str | Path, every: int = 1, axis: str = "x",
                  structure_col: str = "auto", bin_A: float = 1.0) -> Iterator[DumpProfile]:
    """Yield (timestep, profile, bin_centers, box_lo, box_length) per kept frame."""
    for timestep, box, columns, data in iter_frames(path, every):
        mask = boundary_atoms(data, columns, structure_col)
        x = axis_positions(data, columns, axis, box)
        lo, hi = box["xyz".index(axis)]
        profile, centers = position_profile(x[mask], lo, hi, bin_A)
        yield timestep, profile, centers, lo, hi - lo


def _bin_of(x: float, lo: float, bin_A: float, n_bins: int) -> int:
    return int(np.floor((x - lo) / bin_A)) % n_bins


def _follow_periodic(profile: np.ndarray, centers: np.ndarray, prev: float,
                     window_A: float, min_contrast: float) -> float:
    """Best peak within +/- window of prev on the circular profile, or NaN."""
    bin_A = float(centers[1] - centers[0])
    lo, n = float(centers[0]) - bin_A / 2, len(profile)
    reach = int(np.ceil(window_A / bin_A))
    idx = (_bin_of(prev, lo, bin_A, n) + np.arange(-reach, reach + 1)) % n
    k = int(idx[np.argmax(profile[idx])])
    if contrast(profile, k) < min_contrast:
        return np.nan
    return wrap(lo + (refine_peak(profile, k, circular=True) + 0.5) * bin_A, lo, n * bin_A)


def _initial_periodic(profile: np.ndarray, centers: np.ndarray, lo: float,
                      length: float) -> list[float]:
    """Two strongest peaks; middle is the one nearest the box centre."""
    peaks = find_profile_peaks(profile, centers, n=2)
    peaks.sort(key=lambda x: abs(min_image(x - (lo + length / 2), length)))
    return (peaks + [np.nan, np.nan])[:2]


def track_periodic(profiles: Iterable[DumpProfile], search_window_A: float = 5.0,
                   min_contrast_ratio: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps, unwrapped [middle, side] positions in angstrom).

    Each step adds the minimum-image displacement since the previous frame,
    so a boundary that crosses the periodic edge keeps a continuous track.
    """
    timesteps: list[int] = []
    rows: list[list[float]] = []
    wrapped: list[float] = []
    floors: list[float] = []
    for timestep, profile, centers, lo, length in profiles:
        timesteps.append(timestep)
        if not rows:
            wrapped = _initial_periodic(profile, centers, lo, length)
            bin_A = float(centers[1] - centers[0])
            floors = [min_contrast_ratio * contrast(profile, _bin_of(x, lo, bin_A, len(profile)))
                      if np.isfinite(x) else np.inf for x in wrapped]
            rows.append(list(wrapped))
            continue
        new = [_follow_periodic(profile, centers, w, search_window_A, fl)
               if np.isfinite(w) else np.nan for w, fl in zip(wrapped, floors)]
        rows.append([u + min_image(x - w, length) for u, x, w in zip(rows[-1], new, wrapped)])
        wrapped = new
    return np.array(timesteps), np.array(rows, dtype=float).reshape(-1, 2)


def track_dump(path: str | Path, every: int = 1, axis: str = "x", structure_col: str = "auto",
               search_window_A: float = 5.0, bin_A: float = 1.0,
               min_contrast_ratio: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Track both boundaries through a LAMMPS dump: (timesteps, positions in angstrom)."""
    profiles = dump_profiles(path, every, axis, structure_col, bin_A)
    return track_periodic(profiles, search_window_A, min_contrast_ratio)
