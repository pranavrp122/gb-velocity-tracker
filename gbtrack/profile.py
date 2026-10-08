"""Boundary profiles along the migration axis and peak finding, for both modes.

A grain boundary in a bicrystal is a thin planar slab of non-crystalline
atoms perpendicular to the migration axis. Projected onto that axis, its atoms
pile up in a few bins, while thermally disordered atoms elsewhere are spread
evenly. A histogram of boundary-atom positions (dump mode) or the fraction of
white pixels per image column (image mode) therefore shows each boundary as a
sharp peak above a flat background.

In dump mode the box is periodic, so the profile is circular: the boundary at
the periodic edge of the box may have half its atoms near box_lo and half near
box_hi. Smoothing, suppression and sub-bin refinement all wrap around in that
case, so a straddling peak is found as one peak.
"""

from __future__ import annotations

import numpy as np


def smooth(values: np.ndarray, window: int, circular: bool = False) -> np.ndarray:
    """Centred moving average; edges are padded, or wrapped when `circular`."""
    # An even window would shift every peak by half a bin, so round up to odd.
    window = max(1, int(window)) | 1
    if window == 1:
        return values.astype(float)
    half = window // 2
    padded = np.pad(values.astype(float), half, mode="wrap" if circular else "edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def refine_peak(profile: np.ndarray, i: int, circular: bool = False) -> float:
    """Sub-bin peak position from a parabola through bins i-1, i, i+1."""
    n = len(profile)
    if not circular and (i <= 0 or i >= n - 1):
        return float(i)
    left, mid, right = profile[(i - 1) % n], profile[i], profile[(i + 1) % n]
    denom = left - 2 * mid + right
    if denom == 0:
        return float(i)
    return float(i + 0.5 * (left - right) / denom)


def find_peak_indices(profile: np.ndarray, n: int = 2, min_separation: int | None = None,
                      circular: bool = False) -> list[float]:
    """Fractional bin indices of the `n` strongest peaks, sorted ascending.

    Peaks are taken greedily: the highest remaining bin wins, then everything
    within `min_separation` bins of it is suppressed so one wide band cannot
    be reported twice.
    """
    size = len(profile)
    if min_separation is None:
        min_separation = max(5, size // 20)
    work = profile.astype(float).copy()
    peaks: list[float] = []
    for _ in range(n):
        i = int(np.argmax(work))
        if not np.isfinite(work[i]) or work[i] <= 0:
            break
        x = refine_peak(profile, i, circular)
        peaks.append(x % size if circular else x)
        span = np.arange(i - min_separation, i + min_separation + 1)
        work[span % size if circular else span[(span >= 0) & (span < size)]] = -np.inf
    return sorted(peaks)


def contrast(profile: np.ndarray, i: int) -> float:
    """Height of bin i above the median bin, i.e. above the noise background."""
    return float(profile[i] - np.median(profile))


def position_histogram(x: np.ndarray, lo: float, hi: float, bin_A: float = 1.0,
                       smooth_bins: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Normalised, circularly smoothed histogram of boundary-atom positions.

    Returns (bin_centers, hist). The histogram sums to one, so frames with
    different numbers of boundary atoms stay comparable. Positions outside
    [lo, hi) are wrapped into the box first.
    """
    n_bins = max(1, int(round((hi - lo) / bin_A)))
    edges = np.linspace(lo, hi, n_bins + 1)
    wrapped = lo + np.mod(np.asarray(x, float) - lo, hi - lo)
    counts, _ = np.histogram(wrapped, bins=edges)
    total = counts.sum()
    hist = counts / total if total else counts.astype(float)
    return 0.5 * (edges[:-1] + edges[1:]), smooth(hist, smooth_bins, circular=True)


def find_peaks(bin_centers: np.ndarray, hist: np.ndarray, n: int = 2,
               min_separation_A: float = 10.0) -> list[float]:
    """Peak positions in angstrom on a periodic axis, wrapped into the box, sorted."""
    bin_A = float(bin_centers[1] - bin_centers[0])
    lo = float(bin_centers[0]) - bin_A / 2
    length = bin_A * len(bin_centers)
    separation = max(1, int(round(min_separation_A / bin_A)))
    idx = find_peak_indices(hist, n, separation, circular=True)
    return sorted(lo + float(np.mod((i + 0.5) * bin_A, length)) for i in idx)
