"""Command line entry point.

    gbtrack run TRAJECTORY.dmp.gz --every N --dt-ps DT     LAMMPS dump (primary)
    gbtrack images FRAMES_DIR --steps-per-frame N ...       OVITO snapshot images
    gbtrack arrhenius RESULTS.csv                           ln v vs 1/T fit
    gbtrack demo [--images]                                 synthetic end-to-end check
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from . import plots
from .dump import iter_frames, write_dump
from .gbatoms import structure_column
from .synth import DUMP_COLUMNS, make_dump_frames, make_frames
from .track import displacement_px, track_boundaries, track_dump
from .velocity import (angstrom_per_ps_to_m_per_s, arrhenius_fit, fit_velocity,
                       to_angstrom)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGE_COLUMNS = ["frame", "step", "time_ps", "middle_px", "side_px", "middle_A", "side_A",
                 "average_A"]
DUMP_TABLE_COLUMNS = ["frame", "timestep", "time_ps", "middle_A", "side_A", "average_A"]


def frame_number(path: Path) -> int:
    """The last integer in the file name, e.g. 'gb_0120.png' -> 120."""
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"no frame number in file name: {path.name}")
    return int(numbers[-1])


def frame_paths(frames_dir: Path) -> list[Path]:
    """Image files in a directory, ordered by their frame number."""
    paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if len(paths) < 2:
        raise SystemExit(f"need at least two frames in {frames_dir}, found {len(paths)}")
    return sorted(paths, key=frame_number)


def image_table(frame_ids: list[int], positions: np.ndarray, steps_per_frame: float,
                ps_per_step: float, scale: float) -> dict[str, np.ndarray]:
    """Per-frame distances moved by each boundary, in px and angstrom."""
    dist_px = np.abs(displacement_px(positions))
    step = np.asarray(frame_ids, float) * steps_per_frame
    middle_A, side_A = to_angstrom(dist_px[:, 0], scale), to_angstrom(dist_px[:, 1], scale)
    return {"frame": np.asarray(frame_ids), "step": step, "time_ps": step * ps_per_step,
            "middle_px": dist_px[:, 0], "side_px": dist_px[:, 1],
            "middle_A": middle_A, "side_A": side_A, "average_A": (middle_A + side_A) / 2}


def dump_table(timesteps: np.ndarray, positions: np.ndarray, every: int,
               dt_ps: float) -> dict[str, np.ndarray]:
    """Per-frame distances moved by each boundary, in angstrom from box coordinates."""
    dist = np.abs(displacement_px(positions))
    return {"frame": np.arange(len(timesteps)) * every, "timestep": timesteps,
            "time_ps": timesteps * dt_ps, "middle_A": dist[:, 0], "side_A": dist[:, 1],
            "average_A": dist.mean(axis=1)}


def summarize(table: dict[str, np.ndarray]) -> dict:
    """Fit velocity on the averaged distance; fall back to the middle boundary alone."""
    column = "average_A" if np.isfinite(table["average_A"]).sum() >= 2 else "middle_A"
    slope, intercept, r2 = fit_velocity(table["time_ps"], table[column])
    return {"velocity_A_per_ps": slope, "velocity_m_per_s": float(angstrom_per_ps_to_m_per_s(slope)),
            "r2": r2, "intercept_A": intercept, "fit_column": column,
            "n_frames": int(len(table["frame"]))}


def write_table(path: Path, table: dict[str, np.ndarray], columns: list[str]) -> None:
    """CSV with blank cells where a boundary was lost."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for i in range(len(table["frame"])):
            writer.writerow(["" if not np.isfinite(table[c][i]) else f"{table[c][i]:.6g}"
                             for c in columns])


def plot_run(path: Path, table: dict[str, np.ndarray], summary: dict) -> None:
    """Distance vs time for both boundaries, the average, and the fit."""
    fig, ax = plots.new_figure()
    t = table["time_ps"]
    ax.plot(t, table["middle_A"], "o", ms=4, color=plots.SERIES[0], label="middle GB")
    ax.plot(t, table["side_A"], "s", ms=4, color=plots.SERIES[1], label="side GB")
    v = summary["velocity_A_per_ps"]
    plots.plot_distance_fit(ax, t, table[summary["fit_column"]], v, summary["intercept_A"],
                            plots.INK, f"{summary['fit_column']}: v = {v:.3g} A/ps")
    plots.style_axes(ax)
    ax.legend(frameon=False, labelcolor=plots.INK)
    plots.save(fig, path)


def write_outputs(out: Path, table: dict[str, np.ndarray], columns: list[str],
                  extra: dict | None = None) -> dict:
    """distances.csv, summary.json and distance_vs_time.png; returns the summary."""
    out.mkdir(parents=True, exist_ok=True)
    summary = {**summarize(table), **(extra or {})}
    write_table(out / "distances.csv", table, columns)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_run(out / "distance_vs_time.png", table, summary)
    return summary


def run_dump(path: Path, every: int, dt_ps: float, out: Path, axis: str = "x",
             structure_col: str = "auto", window: float = 5.0) -> dict:
    """Track a LAMMPS dump and write all outputs."""
    # Resolve "auto" up front from the first frame's header so the summary records it.
    first_columns = next(iter_frames(path))[2]
    structure_col = structure_column(first_columns, structure_col)
    timesteps, positions = track_dump(path, every, axis, structure_col, window)
    table = dump_table(timesteps, positions, every, dt_ps)
    return write_outputs(out, table, DUMP_TABLE_COLUMNS, {"structure_col": structure_col})


def run_images(paths: list[Path], steps_per_frame: float, ps_per_step: float,
               scale: float, out: Path, window: float = 10.0) -> dict:
    """Track a folder of snapshot images and write all outputs."""
    positions = track_boundaries(paths, search_window_px=window)
    table = image_table([frame_number(p) for p in paths], positions,
                        steps_per_frame, ps_per_step, scale)
    return write_outputs(out, table, IMAGE_COLUMNS)


def cmd_run(args: argparse.Namespace) -> None:
    summary = run_dump(Path(args.trajectory), args.every, args.dt_ps, Path(args.out),
                       args.axis, args.structure_col, args.window)
    print(json.dumps(summary, indent=2))


def cmd_images(args: argparse.Namespace) -> None:
    summary = run_images(frame_paths(Path(args.frames_dir)), args.steps_per_frame,
                         args.ps_per_step, args.scale, Path(args.out), args.window)
    print(json.dumps(summary, indent=2))


def cmd_arrhenius(args: argparse.Namespace) -> None:
    data = np.genfromtxt(args.results, delimiter=",", names=True)
    temps, vel = data["temperature_K"], data["velocity_A_per_ps"]
    ea, prefactor, r2 = arrhenius_fit(temps, vel)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"activation_energy_eV": ea, "prefactor_A_per_ps": prefactor, "r2": r2,
              "temperatures_K": temps.tolist(), "velocities_A_per_ps": vel.tolist()}
    (out / "activation_energy.json").write_text(json.dumps(result, indent=2))
    fig, ax = plots.new_figure()
    plots.plot_arrhenius(ax, temps, vel, ea, prefactor, plots.SERIES[0], Path(args.results).stem)
    plots.style_axes(ax)
    ax.legend(frameon=False, labelcolor=plots.INK)
    plots.save(fig, out / "arrhenius.png")
    print(json.dumps(result, indent=2))


# 25 frames keep the two boundaries, closing in from both sides, >10 A apart in a 60 A box.
DUMP_DEMO = {"n_frames": 25, "v_A_per_frame": 0.35, "speck_fraction": 0.10, "seed": 7,
             "steps_per_frame": 5000, "dt_ps": 0.002}
IMAGE_DEMO = {"n_frames": 60, "v_px_per_frame": 1.5, "speck_fraction": 0.08, "seed": 7,
              "scale_A_per_px": 0.2, "steps_per_frame": 1.0, "ps_per_step": 1000.0}


def report_demo(out: Path, true_v: float, summary: dict) -> None:
    found = summary["velocity_A_per_ps"]
    report = {"true_velocity_A_per_ps": true_v, "recovered_velocity_A_per_ps": found,
              "relative_error": abs(found - true_v) / true_v, "r2": summary["r2"]}
    (out / "demo.json").write_text(json.dumps(report, indent=2))
    print(f"true velocity      {true_v:.6g} A/ps")
    print(f"recovered velocity {found:.6g} A/ps  ({100 * report['relative_error']:.2f}% error)")


def dump_demo(out: Path) -> None:
    """Synthetic dump with both boundaries closing in at the same speed."""
    d = DUMP_DEMO
    out.mkdir(parents=True, exist_ok=True)
    frames = make_dump_frames(d["n_frames"], v_A_per_frame=d["v_A_per_frame"],
                              speck_fraction=d["speck_fraction"], seed=d["seed"],
                              side_v_A_per_frame=-d["v_A_per_frame"],
                              steps_per_frame=d["steps_per_frame"])
    write_dump(out / "demo.dmp.gz", frames, DUMP_COLUMNS)
    summary = run_dump(out / "demo.dmp.gz", 1, d["dt_ps"], out)
    report_demo(out, d["v_A_per_frame"] / (d["steps_per_frame"] * d["dt_ps"]), summary)


def image_demo(out: Path) -> None:
    """Synthetic snapshot images with both boundaries closing in at the same speed."""
    d = IMAGE_DEMO
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = make_frames(d["n_frames"], v_px_per_frame=d["v_px_per_frame"],
                         speck_fraction=d["speck_fraction"], seed=d["seed"],
                         side_v_px_per_frame=-d["v_px_per_frame"])
    for i, img in enumerate(frames):
        Image.fromarray(img).save(frames_dir / f"frame_{i:04d}.png")
    summary = run_images(frame_paths(frames_dir), d["steps_per_frame"], d["ps_per_step"],
                         d["scale_A_per_px"], out)
    true_v = d["v_px_per_frame"] * d["scale_A_per_px"] / (d["steps_per_frame"] * d["ps_per_step"])
    report_demo(out, true_v, summary)


def cmd_demo(args: argparse.Namespace) -> None:
    (image_demo if args.images else dump_demo)(Path(args.out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gbtrack", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="track boundaries in a LAMMPS dump (.dmp or .dmp.gz)")
    run.add_argument("trajectory")
    run.add_argument("--every", type=int, default=1, help="use every Nth frame")
    run.add_argument("--dt-ps", type=float, required=True, help="ps per MD timestep")
    run.add_argument("--axis", choices=["x", "y", "z"], default="x", help="migration axis")
    run.add_argument("--structure-col", default="auto", help="column classifying atoms")
    run.add_argument("--window", type=float, default=5.0, help="search window in angstrom")
    run.add_argument("--out", default="gbtrack_out")
    run.set_defaults(func=cmd_run)
    images = sub.add_parser("images", help="track boundaries in a folder of OVITO snapshots")
    images.add_argument("frames_dir")
    images.add_argument("--steps-per-frame", type=float, required=True)
    images.add_argument("--ps-per-step", type=float, required=True)
    images.add_argument("--scale", type=float, required=True, help="angstrom per pixel")
    images.add_argument("--window", type=float, default=10.0, help="search window in px")
    images.add_argument("--out", default="gbtrack_out")
    images.set_defaults(func=cmd_images)
    arr = sub.add_parser("arrhenius", help="fit ln v vs 1/T from a results CSV")
    arr.add_argument("results")
    arr.add_argument("--out", default="gbtrack_out")
    arr.set_defaults(func=cmd_arrhenius)
    demo = sub.add_parser("demo", help="track synthetic data with a known velocity")
    demo.add_argument("--images", action="store_true", help="use snapshot images, not a dump")
    demo.add_argument("--out", default="gbtrack_demo")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
