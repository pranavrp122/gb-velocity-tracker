"""Stream LAMMPS dump trajectories, the same files OVITO loads.

A dump holds one block per frame:

    ITEM: TIMESTEP
    <timestep>
    ITEM: NUMBER OF ATOMS
    <n>
    ITEM: BOX BOUNDS pp pp pp
    <xlo> <xhi>
    <ylo> <yhi>
    <zlo> <zhi>
    ITEM: ATOMS id type x y z ...
    <n lines, one per atom>

Frames are parsed one at a time, so a trajectory far larger than memory can
be processed. Files ending in .gz are decompressed on the fly. Column names
come from the ITEM: ATOMS header verbatim, so expanded computes such as
c_voro[1] and any extra per-atom columns are kept as they are.
"""

from __future__ import annotations

import gzip
from itertools import islice
from pathlib import Path
from typing import IO, Iterable, Iterator

import numpy as np

Box = list[tuple[float, float]]
Frame = tuple[int, Box, list[str], np.ndarray]


def _open(path: str | Path, mode: str = "rt") -> IO[str]:
    path = Path(path)
    if path.suffix == ".gz":
        # Level 4 writes several times faster than the default 9 for ~10% more bytes.
        return gzip.open(path, mode, compresslevel=4)
    return path.open(mode)


def _expect(line: str, item: str) -> str:
    if not line.startswith(item):
        raise ValueError(f"malformed dump: expected '{item}', found '{line.strip()}'")
    return line


def _read_header(f: IO[str]) -> tuple[int, int, Box, list[str]]:
    timestep = int(next(f).split()[0])
    _expect(next(f), "ITEM: NUMBER OF ATOMS")
    n_atoms = int(next(f).split()[0])
    _expect(next(f), "ITEM: BOX BOUNDS")
    # Triclinic boxes carry a third tilt value per line; only lo and hi are used.
    box = [tuple(float(v) for v in next(f).split()[:2]) for _ in range(3)]
    columns = _expect(next(f), "ITEM: ATOMS").split()[2:]
    return timestep, n_atoms, box, columns


def iter_frames(path: str | Path, every: int = 1) -> Iterator[Frame]:
    """Yield (timestep, box, columns, data) for every `every`-th frame.

    `box` is [(xlo, xhi), (ylo, yhi), (zlo, zhi)] and `data` is an
    (n_atoms, n_columns) float array. Skipped frames are read past without
    being converted to numbers.
    """
    with _open(path) as f:
        index = 0
        for line in f:
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            try:
                timestep, n_atoms, box, columns = _read_header(f)
            except StopIteration:
                raise ValueError("truncated dump: the last frame header is incomplete") from None
            rows = list(islice(f, n_atoms))
            if len(rows) < n_atoms:
                raise ValueError(f"truncated dump: frame at timestep {timestep} is incomplete")
            if index % every == 0:
                data = np.array(" ".join(rows).split(), dtype=float).reshape(n_atoms, -1)
                yield timestep, box, columns, data
            index += 1


def axis_positions(data: np.ndarray, columns: list[str], axis: str, box: Box) -> np.ndarray:
    """Positions along `axis` wrapped into the box, from x, xu or scaled xs columns."""
    lo, hi = box["xyz".index(axis)]
    if axis in columns:
        pos = data[:, columns.index(axis)]
    elif f"{axis}u" in columns:
        pos = data[:, columns.index(f"{axis}u")]
    elif f"{axis}s" in columns:
        pos = lo + data[:, columns.index(f"{axis}s")] * (hi - lo)
    else:
        raise ValueError(f"dump has no '{axis}', '{axis}u' or '{axis}s' column; found {columns}")
    return lo + np.mod(pos - lo, hi - lo)


def write_dump(path: str | Path, frames: Iterable[tuple[int, Box, np.ndarray]],
               columns: list[str]) -> None:
    """Write (timestep, box, data) frames as a LAMMPS dump; .gz compresses."""
    with _open(path, "wt") as f:
        for timestep, box, data in frames:
            f.write(f"ITEM: TIMESTEP\n{timestep}\nITEM: NUMBER OF ATOMS\n{len(data)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.writelines(f"{lo:.6f} {hi:.6f}\n" for lo, hi in box)
            f.write("ITEM: ATOMS " + " ".join(columns) + "\n")
            np.savetxt(f, data, fmt="%.6g")
