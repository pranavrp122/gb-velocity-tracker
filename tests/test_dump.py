"""Reading and writing LAMMPS dump trajectories."""

import types

import numpy as np
import pytest

from gbtrack.dump import axis_positions, iter_frames, write_dump

COLUMNS = ["id", "type", "x", "y", "z", "c_pe", "v_ptm_type", "c_csym", "c_voro[1]", "v_nn11"]
BOX = [(0.0, 50.0), (-5.0, 5.0), (0.0, 12.5)]


def synthetic_frames(n: int = 3, atoms: int = 7) -> list:
    rng = np.random.default_rng(0)
    frames = []
    for f in range(n):
        data = np.column_stack([np.arange(1, atoms + 1), np.ones(atoms),
                                rng.uniform(0, 50, (atoms, 3)), rng.normal(-4.4, 0.1, atoms),
                                rng.integers(0, 2, atoms), rng.uniform(0, 20, atoms),
                                rng.uniform(9, 12, atoms), rng.integers(0, 12, atoms)])
        frames.append((f * 1000, BOX, data))
    return frames


@pytest.mark.parametrize("name", ["traj.dmp", "traj.dmp.gz"])
def test_round_trip(tmp_path, name: str) -> None:
    frames = synthetic_frames()
    write_dump(tmp_path / name, frames, COLUMNS)
    read = list(iter_frames(tmp_path / name))
    assert [r[0] for r in read] == [0, 1000, 2000]
    for (timestep, box, data), (t, b, cols, d) in zip(frames, read):
        assert t == timestep and b == BOX and cols == COLUMNS
        assert np.allclose(d, data, rtol=1e-5)


def test_gzip_file_is_really_compressed(tmp_path) -> None:
    write_dump(tmp_path / "t.dmp.gz", synthetic_frames(), COLUMNS)
    assert (tmp_path / "t.dmp.gz").read_bytes()[:2] == b"\x1f\x8b"


def test_streams_frame_by_frame(tmp_path) -> None:
    # A corrupt tail must not stop the first frames from being read lazily.
    path = tmp_path / "t.dmp"
    write_dump(path, synthetic_frames(3), COLUMNS)
    with path.open("a") as f:
        f.write("ITEM: TIMESTEP\n3000\nITEM: NUMBER OF ATOMS\n7\nITEM: BOX BOUNDS pp pp pp\n")
    frames = iter_frames(path)
    assert isinstance(frames, types.GeneratorType)
    assert [next(frames)[0] for _ in range(3)] == [0, 1000, 2000]
    with pytest.raises(ValueError, match="truncated"):
        next(frames)


def test_every_nth_frame(tmp_path) -> None:
    write_dump(tmp_path / "t.dmp", synthetic_frames(5), COLUMNS)
    assert [f[0] for f in iter_frames(tmp_path / "t.dmp", every=2)] == [0, 2000, 4000]


def test_truncated_frame_raises(tmp_path) -> None:
    path = tmp_path / "t.dmp"
    write_dump(path, synthetic_frames(1), COLUMNS)
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-2]))
    with pytest.raises(ValueError, match="truncated"):
        list(iter_frames(path))


def test_axis_positions_handles_scaled_and_unwrapped_columns() -> None:
    box = [(10.0, 20.0), (0.0, 1.0), (0.0, 1.0)]
    data = np.array([[0.25, 23.0], [0.95, 9.0]])
    assert np.allclose(axis_positions(data, ["xs", "y"], "x", box), [12.5, 19.5])
    assert np.allclose(axis_positions(data[:, ::-1], ["xu", "ys"], "x", box), [13.0, 19.0])
    with pytest.raises(ValueError, match="no 'z'"):
        axis_positions(data, ["xs", "y"], "z", box)
