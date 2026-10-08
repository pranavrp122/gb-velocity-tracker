"""Velocity and Arrhenius fits against the recorded lab results."""

import numpy as np
import pytest

from gbtrack.velocity import (KB_EV_PER_K, angstrom_per_ps_to_m_per_s, arrhenius_fit,
                              fit_velocity, to_angstrom)

TEMPS = np.array([1000.0, 1100.0, 1200.0, 1300.0])
NICKEL_V = np.array([0.000290, 0.000413, 0.000545, 0.000660])
NICOCR_V = np.array([0.00137, 0.00147, 0.00169, 0.00195])


def _velocity(table: np.ndarray, temp: float) -> float:
    rows = table[table["temperature_K"] == temp]
    return fit_velocity(rows["time_ps"], rows["average_A"])[0]


def test_nickel_1000K_average_reproduces_recorded_velocity(nickel) -> None:
    assert _velocity(nickel, 1000) == pytest.approx(0.000290, rel=0.03)


@pytest.mark.parametrize("temp,expected", list(zip(TEMPS, NICKEL_V)))
def test_every_nickel_temperature_matches(nickel, temp: float, expected: float) -> None:
    assert _velocity(nickel, temp) == pytest.approx(expected, rel=0.03)


@pytest.mark.parametrize("temp,expected", list(zip(TEMPS, NICOCR_V)))
def test_every_nicocr_temperature_matches(nicocr, temp: float, expected: float) -> None:
    assert _velocity(nicocr, temp) == pytest.approx(expected, rel=0.03)


def test_unit_conversions_are_exact() -> None:
    assert angstrom_per_ps_to_m_per_s(1.0) == 100.0
    assert angstrom_per_ps_to_m_per_s(0.000290) == pytest.approx(0.029, abs=1e-15)
    assert to_angstrom(25, 0.4) == pytest.approx(10.0)


def test_fit_velocity_ignores_nan_and_reports_r2() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    d = np.array([1.0, 3.0, 5.0, 7.0, np.nan])
    assert fit_velocity(t, d) == pytest.approx((2.0, 1.0, 1.0))


def test_nickel_activation_energy() -> None:
    ea, prefactor, r2 = arrhenius_fit(TEMPS, NICKEL_V)
    assert 0.28 < ea < 0.33
    assert r2 > 0.95
    assert prefactor * np.exp(-ea / (KB_EV_PER_K * 1000)) == pytest.approx(0.00029, rel=0.05)


def test_nicocr_activation_energy() -> None:
    ea, _, _ = arrhenius_fit(TEMPS, NICOCR_V)
    assert 0.10 < ea < 0.16


def test_arrhenius_recovers_a_known_activation_energy() -> None:
    v = 3.0 * np.exp(-0.5 / (KB_EV_PER_K * TEMPS))
    assert arrhenius_fit(TEMPS, v) == pytest.approx((0.5, 3.0, 1.0))
