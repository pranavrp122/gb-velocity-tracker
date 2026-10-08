"""Locate vertical grain boundaries in a single OVITO render.

OVITO colours grain-boundary atoms white and crystalline atoms green on a
black background. A grain boundary in a bicrystal is a coherent vertical band,
so the fraction of white pixels in each image column spikes at the boundary.
Thermal noise also produces white atoms, but those are scattered: they raise
every column by roughly the same small amount instead of piling up in a few
columns. The boundary therefore stands out as a peak in the column profile
even when the frame is full of specks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .profile import find_peaks, refine_peak, smooth  # noqa: F401  (re-exported)

BLACK_LEVEL = 30


def load_frame(path: str | Path) -> np.ndarray:
    """Read an image file as an (H, W, 3) uint8 RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def boundary_mask(img: np.ndarray, white_thresh: int = 170) -> np.ndarray:
    """Return a bool mask of whitish pixels (every channel above the threshold).

    Using the minimum channel rejects bright green atoms, whose red and blue
    channels stay low, while keeping grey-shaded white atoms.
    """
    return img.min(axis=2) > white_thresh


def _non_black(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img.max(axis=2) > BLACK_LEVEL
    return img.astype(bool)


def sample_rows(img_or_mask: np.ndarray) -> tuple[int, int]:
    """Return the half-open row range [top, bottom) where the sample is drawn.

    Pure-black rows above and below the sample would dilute the column
    fractions, so they are excluded. A mask is treated as "sample wherever True".
    """
    rows = np.flatnonzero(_non_black(img_or_mask).any(axis=1))
    if rows.size == 0:
        return 0, img_or_mask.shape[0]
    return int(rows[0]), int(rows[-1]) + 1


def sample_columns(img_or_mask: np.ndarray) -> tuple[int, int]:
    """Return the half-open column range [left, right) where the sample is drawn."""
    cols = np.flatnonzero(_non_black(img_or_mask).any(axis=0))
    if cols.size == 0:
        return 0, img_or_mask.shape[1]
    return int(cols[0]), int(cols[-1]) + 1


def column_profile(mask: np.ndarray, rows: tuple[int, int] | None = None,
                   window: int | None = None) -> np.ndarray:
    """Fraction of white pixels per column inside `rows`, smoothed.

    The default smoothing window is about 1% of the image width: wide enough to
    merge the individual atoms of a boundary into one peak, narrow enough not
    to blur two nearby boundaries together.
    """
    top, bottom = rows if rows is not None else (0, mask.shape[0])
    frac = mask[top:bottom].mean(axis=0)
    if window is None:
        window = max(3, round(mask.shape[1] * 0.01))
    return smooth(frac, window)


def find_boundaries(profile: np.ndarray, n: int = 2,
                    min_separation: int | None = None) -> list[float]:
    """Return x positions of the `n` strongest column-profile peaks, left to right.

    An image is not periodic, so the search does not wrap around its edges.
    """
    return find_peaks(profile, n, min_separation, circular=False)
