"""Tracking boundaries through synthetic LAMMPS dumps."""

import json

import numpy as np
import pytest

from gbtrack.cli import main
from gbtrack.dump import write_dump
from gbtrack.synth import DUMP_BOX, DUMP_COLUMNS, dump_band_positions, make_dump_frames
from gbtrack.track import displacement_px, min_image, track_dump, wrap
from gbtrack.velocity import fit_velocity

V = 0.35


@pytest.fixture(scope="module")
def closing_dump(tmp_path_factory):
    """40 frames, 10% specks; middle moves +x, side starts on the edge and moves -x."""
    path = tmp_path_factory.mktemp("dump") / "closing.dmp.gz"
    write_dump(path, make_dump_frames(40, v_A_per_frame=V, speck_fraction=0.10, seed=3,
                                      side_v_A_per_frame=-V), DUMP_COLUMNS)
    return path


def test_recovers_velocity_within_three_percent(closing_dump) -> None:
    timesteps, positions = track_dump(closing_dump)
    assert timesteps.tolist() == [1000 * i for i in range(40)]
    slope, _, r2 = fit_velocity(np.arange(40), displacement_px(positions)[:, 0])
    assert slope == pytest.approx(V, rel=0.03)
    assert r2 > 0.99


def test_boundary_crossing_the_periodic_edge_stays_continuous(closing_dump) -> None:
    _, positions = track_dump(closing_dump)
    side = displacement_px(positions)[:, 1]
    truth = np.array([dump_band_positions(i, DUMP_BOX[0], V, -V)[1] for i in range(40)])
    assert np.all(np.isfinite(side))
    assert np.abs(np.diff(side)).max() < 2.0
    assert np.abs(side - truth).max() < 1.0
    assert side[-1] == pytest.approx(-V * 39, abs=1.0)


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


def test_run_command_on_a_dump(closing_dump, tmp_path) -> None:
    main(["run", str(closing_dump), "--every", "2", "--dt-ps", "0.002", "--out", str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_frames"] == 20
    assert summary["velocity_A_per_ps"] == pytest.approx(V / (1000 * 0.002), rel=0.03)
    header = (tmp_path / "distances.csv").read_text().splitlines()[0]
    assert header == "frame,timestep,time_ps,middle_A,side_A,average_A"
    table = np.genfromtxt(tmp_path / "distances.csv", delimiter=",", names=True)
    assert table["frame"][:3].tolist() == [0, 2, 4]
    assert table["time_ps"][1] == pytest.approx(2000 * 0.002)
