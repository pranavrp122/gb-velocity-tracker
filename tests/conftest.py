"""Shared paths and data loaders for the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OVITO = ROOT / "examples" / "ovito"


def load_distances(name: str) -> np.ndarray:
    """Structured array from one of the data/*_distances.csv tables."""
    return np.genfromtxt(DATA / name, delimiter=",", names=True)


@pytest.fixture(scope="session")
def nickel() -> np.ndarray:
    return load_distances("nickel_distances.csv")


@pytest.fixture(scope="session")
def nicocr() -> np.ndarray:
    return load_distances("nicocr_u008_distances.csv")
