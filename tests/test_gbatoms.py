"""Choosing boundary atoms from structure type or centrosymmetry."""

import numpy as np
import pytest

from gbtrack.gbatoms import DEFAULT_CSYM_THRESH, boundary_atoms, structure_column

PTM = np.array([1, 0, 1, 2, 3, 4, 1], dtype=float)
CSYM = np.array([0.2, 14.0, 0.9, 9.0, 3.0, 25.0, 7.9])


def test_ptm_column_marks_every_non_fcc_atom() -> None:
    data = np.column_stack([PTM, CSYM])
    mask = boundary_atoms(data, ["v_ptm_type", "c_csym"])
    assert mask.tolist() == [False, True, False, True, True, True, False]


@pytest.mark.parametrize("name", ["v_ptm_type", "ptm_type", "StructureType", "c_ptm[1]"])
def test_structure_column_aliases_take_priority_over_csym(name: str) -> None:
    assert structure_column(["id", "c_csym", name]) == name


def test_csym_fallback_uses_documented_threshold() -> None:
    mask = boundary_atoms(CSYM[:, None], ["c_csym"])
    assert mask.tolist() == (CSYM > DEFAULT_CSYM_THRESH).tolist()
    assert DEFAULT_CSYM_THRESH == 8.0


def test_csym_threshold_can_be_overridden() -> None:
    mask = boundary_atoms(CSYM[:, None], ["csym"], csym_thresh=2.0)
    assert mask.sum() == 5


def test_explicit_column_and_fcc_value() -> None:
    data = np.column_stack([PTM, np.zeros_like(PTM)])
    mask = boundary_atoms(data, ["v_ptm_type", "c_cna"], structure_col="c_cna", fcc_value=0)
    assert not mask.any()


def test_missing_columns_raise_a_clear_error() -> None:
    with pytest.raises(ValueError, match="v_ptm_type.*c_csym.*dump custom"):
        boundary_atoms(np.zeros((3, 2)), ["id", "c_pe"])


def test_unknown_explicit_column_raises() -> None:
    with pytest.raises(ValueError, match="'v_ptm_type' is not in this dump"):
        boundary_atoms(np.zeros((3, 1)), ["c_csym"], structure_col="v_ptm_type")
