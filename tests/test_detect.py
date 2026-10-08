"""Single-frame detection on synthetic renders."""

import numpy as np
import pytest

from gbtrack.detect import (boundary_mask, column_profile, find_boundaries,
                            refine_peak, sample_rows)
from gbtrack.synth import band_positions, make_frames


@pytest.mark.parametrize("speck_fraction", [0.0, 0.15])
def test_both_boundaries_found_within_one_pixel(speck_fraction: float) -> None:
    img = make_frames(1, speck_fraction=speck_fraction, seed=3)[0]
    mask = boundary_mask(img)
    found = find_boundaries(column_profile(mask, sample_rows(img)), n=2)
    truth = sorted(band_positions(0, img.shape[1], 1.0))
    assert len(found) == 2
    assert np.allclose(found, truth, atol=1.0)


def test_sample_rows_excludes_black_border() -> None:
    img = make_frames(1, height=200, speck_fraction=0.0)[0]
    top, bottom = sample_rows(img)
    assert img[:top].max() == 0 and img[bottom:].max() == 0
    assert img[top].max() > 0 and img[bottom - 1].max() > 0


def test_green_atoms_are_not_boundary_pixels() -> None:
    img = make_frames(1, speck_fraction=0.0, side_gb=False)[0]
    mask = boundary_mask(img)
    centre = img.shape[1] // 2
    assert not mask[:, : centre - 10].any()


def test_parabolic_refinement_recovers_offset_peak() -> None:
    x = np.arange(20, dtype=float)
    profile = -((x - 9.3) ** 2)
    assert refine_peak(profile, int(np.argmax(profile))) == pytest.approx(9.3)


def test_min_separation_stops_double_counting_a_wide_band() -> None:
    profile = np.zeros(100)
    profile[40:46] = 1.0
    profile[80] = 0.5
    peaks = find_boundaries(profile, n=2, min_separation=10)
    assert len(peaks) == 2 and peaks[1] == pytest.approx(80)
