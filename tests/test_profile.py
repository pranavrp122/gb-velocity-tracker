"""Position histograms and periodic peak finding."""

import numpy as np
import pytest

from gbtrack.profile import find_peak_indices, find_peaks, position_histogram
from gbtrack.track import min_image

LO, HI = -20.0, 100.0


def slab_atoms(centre: float, n: int, rng: np.random.Generator, width: float = 3.0) -> np.ndarray:
    return centre + rng.uniform(-width / 2, width / 2, n)


def noisy_positions(centres: list[float], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = [slab_atoms(c, 400, rng) for c in centres] + [rng.uniform(LO, HI, 300)]
    return np.concatenate(parts)


def test_histogram_is_normalised_and_binned() -> None:
    centers, hist = position_histogram(noisy_positions([30.0]), LO, HI, bin_A=0.5)
    assert len(hist) == 240
    assert hist.sum() == pytest.approx(1.0)
    assert centers[0] == pytest.approx(LO + 0.25)


def test_both_peaks_within_half_angstrom() -> None:
    centers, hist = position_histogram(noisy_positions([12.3, 71.8]), LO, HI)
    assert np.allclose(find_peaks(centers, hist, n=2), [12.3, 71.8], atol=0.5)


@pytest.mark.parametrize("edge_centre", [LO, LO + 0.6, HI - 0.6])
def test_peak_straddling_the_periodic_edge(edge_centre: float) -> None:
    centers, hist = position_histogram(noisy_positions([40.0, edge_centre]), LO, HI)
    found = find_peaks(centers, hist, n=2)
    assert all(LO <= x < HI for x in found)
    edge = [x for x in found if abs(x - 40.0) > 5]
    assert len(edge) == 1
    assert abs(min_image(edge[0] - edge_centre, HI - LO)) < 0.5


def test_circular_search_merges_the_two_ends_but_open_one_does_not() -> None:
    profile = np.zeros(50)
    profile[0], profile[49], profile[25] = 1.0, 0.9, 0.5
    assert find_peak_indices(profile, n=2, min_separation=5) == [0.0, 49.0]
    peaks = find_peak_indices(profile, n=2, min_separation=5, circular=True)
    assert 25.0 in peaks
    edge = [p for p in peaks if p != 25.0][0]
    assert abs(min_image(edge, 50)) < 0.5  # one peak between bins 49 and 0
