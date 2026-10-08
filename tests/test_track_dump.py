"""Tracking boundaries through synthetic LAMMPS dumps."""

import numpy as np
import pytest

from gbtrack.dump import write_dump
from gbtrack.synth import DUMP_COLUMNS, dump_band_positions, make_dump_frames
from gbtrack.track import displacement_px, min_image, track_dump, wrap
from gbtrack.velocity import fit_velocity

V = 0.35
# Long enough that two slabs closing in at 0.35 A/frame stay apart for 40 frames.
LONG_BOX = (120.0, 30.0, 30.0)


@pytest.fixture(scope="module")
def middle_only_motion(tmp_path_factory):
    """40 frames, 10% specks; middle moves +x, side boundary stays on the edge."""
    path = tmp_path_factory.mktemp("dump") / "middle.dmp.gz"
    write_dump(path, make_dump_frames(40, v_A_per_frame=V, speck_fraction=0.10, seed=3),
               DUMP_COLUMNS)
    return path


@pytest.fixture(scope="module")
def closing_dump(tmp_path_factory):
    """Middle moves +x while the side boundary starts on the edge and moves -x."""
    path = tmp_path_factory.mktemp("dump") / "closing.dmp.gz"
    write_dump(path, make_dump_frames(40, box=LONG_BOX, v_A_per_frame=V, speck_fraction=0.10,
                                      seed=4, side_v_A_per_frame=-V), DUMP_COLUMNS)
    return path


def test_recovers_velocity_within_three_percent(middle_only_motion) -> None:
    timesteps, positions = track_dump(middle_only_motion)
    assert timesteps.tolist() == [1000 * i for i in range(40)]
    slope, _, r2 = fit_velocity(np.arange(40), displacement_px(positions)[:, 0])
    assert slope == pytest.approx(V, rel=0.03)
    assert r2 > 0.99
    assert np.abs(displacement_px(positions)[:, 1]).max() < 1.0


def test_boundary_crossing_the_periodic_edge_is_continuous_and_linear(closing_dump) -> None:
    _, positions = track_dump(closing_dump)
    side = displacement_px(positions)[:, 1]
    truth = np.array([dump_band_positions(i, LONG_BOX[0], V, -V)[1] for i in range(40)])
    assert np.all(np.isfinite(side))
    assert np.abs(np.diff(side)).max() < 2.0
    assert np.abs(side - truth).max() < 1.0
    slope, _, r2 = fit_velocity(np.arange(40), side)
    assert slope == pytest.approx(-V, rel=0.03)
    assert r2 > 0.99


def test_middle_boundary_wraps_through_the_box_edge(tmp_path) -> None:
    box = (60.0, 21.12, 21.12)
    path = tmp_path / "fast.dmp"
    write_dump(path, make_dump_frames(30, box=box, v_A_per_frame=1.5, speck_fraction=0.05,
                                      seed=5, side_gb=False), DUMP_COLUMNS)
    _, positions = track_dump(path)
    middle = displacement_px(positions)[:, 0]
    assert middle[-1] == pytest.approx(1.5 * 29, abs=1.0)
    wrapped_final = wrap(box[0] / 2 + 1.5 * 29, 0.0, box[0])
    assert abs(min_image(positions[-1, 0] - wrapped_final, box[0])) < 1.0


def test_lost_boundary_becomes_nan_without_breaking_the_other(tmp_path) -> None:
    # With no side slab, the second "boundary" frame 0 picks is noise and is soon lost.
    path = tmp_path / "single.dmp"
    write_dump(path, make_dump_frames(20, v_A_per_frame=V, seed=8, side_gb=False), DUMP_COLUMNS)
    _, positions = track_dump(path)
    assert np.all(np.isfinite(positions[:, 0]))
    assert np.isnan(positions[-1, 1])
    assert min_image(float("nan"), 60.0) != min_image(float("nan"), 60.0)
