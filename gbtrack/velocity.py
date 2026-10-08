"""Unit conversion, boundary velocity fits and the Arrhenius fit.

Boundary velocity is the slope of migration distance against time. Its
temperature dependence follows v = v0 * exp(-Ea / (k_B T)), so ln v is linear
in 1/T with slope -Ea / k_B.
"""

from __future__ import annotations

import numpy as np

KB_EV_PER_K = 8.617333e-5
M_PER_S_PER_A_PER_PS = 100.0  # 1e-10 m / 1e-12 s


def to_angstrom(px: np.ndarray | float, scale_A_per_px: float) -> np.ndarray:
    """Convert pixel distances to angstrom."""
    return np.asarray(px, dtype=float) * scale_A_per_px


def angstrom_per_ps_to_m_per_s(v: np.ndarray | float) -> np.ndarray:
    """Convert a velocity from angstrom/ps to m/s."""
    return np.asarray(v, dtype=float) * M_PER_S_PER_A_PER_PS


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2:
        raise ValueError("need at least two finite points to fit a line")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r2


def fit_velocity(time_ps: np.ndarray, distance_A: np.ndarray) -> tuple[float, float, float]:
    """Least-squares line through distance vs time: (slope A/ps, intercept A, r2).

    Non-finite points, such as frames after a boundary left the sample, are ignored.
    """
    return _linear_fit(np.asarray(time_ps, float), np.asarray(distance_A, float))


def arrhenius_fit(temps_K: np.ndarray, velocities: np.ndarray) -> tuple[float, float, float]:
    """Fit ln v against 1/T: (activation energy eV, prefactor v0, r2).

    The prefactor is in the same units as `velocities`.
    """
    inv_t = 1.0 / np.asarray(temps_K, float)
    slope, intercept, r2 = _linear_fit(inv_t, np.log(np.asarray(velocities, float)))
    return -slope * KB_EV_PER_K, float(np.exp(intercept)), r2
