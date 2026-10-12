"""Classify dump atoms as grain-boundary atoms.

OVITO renders an atom white when its local structure is not FCC, so the
boundary atoms are exactly the non-FCC atoms. The best signal is a structure
type column from Polyhedral Template Matching or common neighbour analysis
(0 other, 1 FCC, 2 HCP, 3 BCC, 4 ICO). Without one, centrosymmetry is used:
it is near zero for an intact FCC neighbourhood and large where the lattice
is broken.

A bicrystal under periodic boundary conditions always has two boundaries:
the middle one and a second one at the periodic edge of the box, where the
two grains meet across the boundary. Both are returned by this mask; nothing
else is excluded, so thermally disordered atoms inside the grains are
included too and show up as a flat background in the position profile.
"""

from __future__ import annotations

import numpy as np

STRUCTURE_COLUMNS = ("v_ptm_type", "ptm_type", "StructureType", "c_ptm[1]")
CSYM_COLUMNS = ("c_csym", "csym")
# Centrosymmetry threshold in angstrom^2: FCC thermal vibration stays below it, boundaries and faults sit above it.
DEFAULT_CSYM_THRESH = 8.0


def structure_column(columns: list[str], structure_col: str = "auto") -> str:
    """Pick the column used to classify atoms, or raise a helpful error."""
    if structure_col != "auto":
        if structure_col not in columns:
            raise ValueError(f"column '{structure_col}' is not in this dump; found {columns}")
        return structure_col
    for name in STRUCTURE_COLUMNS + CSYM_COLUMNS:
        if name in columns:
            return name
    raise ValueError(
        "cannot tell boundary atoms apart: the dump has none of "
        f"{', '.join(STRUCTURE_COLUMNS + CSYM_COLUMNS)}. Add a structure type "
        "(e.g. 'compute ptm all ptm/atom default 0.1 all' and dump c_ptm[1]) or "
        "centrosymmetry ('compute csym all centro/atom fcc' and dump c_csym) to "
        "your dump custom command."
    )


def boundary_atoms(data: np.ndarray, columns: list[str], structure_col: str = "auto",
                   fcc_value: int = 1,
                   csym_thresh: float = DEFAULT_CSYM_THRESH) -> np.ndarray:
    """Bool mask of boundary atoms: non-FCC structure, or high centrosymmetry.

    With structure_col="auto" the first column named v_ptm_type, ptm_type,
    StructureType or c_ptm[1] is used and an atom is a boundary atom when its
    type differs from `fcc_value`. If none of those exist, c_csym or csym is
    used and an atom is a boundary atom when its centrosymmetry exceeds
    `csym_thresh`. If neither exists a ValueError names the columns looked for.
    """
    name = structure_column(columns, structure_col)
    values = data[:, columns.index(name)]
    if name in CSYM_COLUMNS:
        return values > csym_thresh
    return np.rint(values) != fcc_value
