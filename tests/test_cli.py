"""End-to-end checks of the gbtrack command line."""

import json

import numpy as np
import pytest
from PIL import Image

from conftest import DATA
from gbtrack.cli import frame_number, main
from gbtrack.synth import make_frames


def test_frame_number_uses_last_integer(tmp_path) -> None:
    assert frame_number(tmp_path / "ni_1000K_step120.png") == 120


def test_images_writes_all_outputs(tmp_path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Unpadded names (gb_10 sorts before gb_2 lexically) prove numeric ordering.
    for i, img in enumerate(make_frames(12, v_px_per_frame=2.0, seed=4,
                                        side_v_px_per_frame=-2.0)):
        Image.fromarray(img).save(frames_dir / f"gb_{i}.png")
    out = tmp_path / "out"
    main(["images", str(frames_dir), "--steps-per-frame", "10", "--ps-per-step", "100",
          "--scale", "0.5", "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_frames"] == 12
    assert summary["velocity_A_per_ps"] == pytest.approx(2.0 * 0.5 / 1000, rel=0.03)
    assert summary["velocity_m_per_s"] == pytest.approx(100 * summary["velocity_A_per_ps"])
    table = np.genfromtxt(out / "distances.csv", delimiter=",", names=True)
    assert table["frame"].tolist() == list(range(12))
    assert (out / "distance_vs_time.png").stat().st_size > 0


def test_arrhenius_command(tmp_path) -> None:
    main(["arrhenius", str(DATA / "nickel_velocities.csv"), "--out", str(tmp_path)])
    result = json.loads((tmp_path / "activation_energy.json").read_text())
    assert 0.28 < result["activation_energy_eV"] < 0.33
    assert (tmp_path / "arrhenius.png").stat().st_size > 0
