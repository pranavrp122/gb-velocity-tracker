"""Tracking a boundary through a synthetic frame series."""

import numpy as np
import pytest
from PIL import Image

from gbtrack.synth import band_positions, make_frames
from gbtrack.track import displacement_px, track_boundaries
from gbtrack.velocity import fit_velocity

V_TRUE = 1.7


@pytest.mark.parametrize("speck_fraction", [0.0, 0.15])
def test_recovers_velocity_within_three_percent(speck_fraction: float) -> None:
    frames = make_frames(40, v_px_per_frame=V_TRUE, speck_fraction=speck_fraction, seed=11)
    positions = track_boundaries(frames, search_window_px=8)
    slope, _, r2 = fit_velocity(np.arange(40), displacement_px(positions)[:, 0])
    assert slope == pytest.approx(V_TRUE, rel=0.03)
    assert r2 > 0.99


def test_positions_match_truth_frame_by_frame() -> None:
    frames = make_frames(40, v_px_per_frame=V_TRUE, speck_fraction=0.1, seed=5)
    positions = track_boundaries(frames)
    truth = np.array([band_positions(i, 640, V_TRUE) for i in range(40)])
    assert np.abs(positions - truth).max() < 1.0


def test_boundary_leaving_the_sample_gives_nan_tail() -> None:
    frames = make_frames(60, v_px_per_frame=6.0, speck_fraction=0.1, seed=2, side_gb=False)
    middle = track_boundaries(frames, search_window_px=12)[:, 0]
    lost = np.flatnonzero(np.isnan(middle))
    assert lost.size > 0
    assert np.all(np.isnan(middle[lost[0]:]))
    assert np.all(np.isfinite(middle[: lost[0]]))
    sample_right_edge = int(0.92 * 640)
    assert middle[lost[0] - 1] <= sample_right_edge


def test_accepts_image_paths(tmp_path) -> None:
    frames = make_frames(5, v_px_per_frame=2.0, seed=1)
    paths = []
    for i, img in enumerate(frames):
        paths.append(tmp_path / f"f{i}.png")
        Image.fromarray(img).save(paths[-1])
    assert np.allclose(track_boundaries(paths), track_boundaries(frames))


def test_displacement_is_relative_to_first_frame() -> None:
    positions = np.array([[10.0, 50.0], [12.0, 49.0], [15.0, np.nan]])
    disp = displacement_px(positions)
    assert disp[0].tolist() == [0.0, 0.0]
    assert disp[2, 0] == 5.0 and np.isnan(disp[2, 1])
