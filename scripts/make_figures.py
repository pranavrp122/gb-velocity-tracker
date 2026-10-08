"""Regenerate every figure in figures/ from the data and example frames.

Run from the repository root:  python scripts/make_figures.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from gbtrack import plots
from gbtrack.cli import DUMP_DEMO
from gbtrack.detect import boundary_mask, column_profile, find_boundaries, load_frame, sample_rows
from gbtrack.dump import write_dump
from gbtrack.profile import find_peaks as find_profile_peaks
from gbtrack.synth import DUMP_COLUMNS, make_dump_frames, make_frames
from gbtrack.track import (displacement_px, dump_profiles, frame_profile, track_boundaries,
                           track_periodic)
from gbtrack.velocity import angstrom_per_ps_to_m_per_s, arrhenius_fit, fit_velocity

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGURES, OVITO = ROOT / "data", ROOT / "figures", ROOT / "examples" / "ovito"
TEMPS = [1000, 1100, 1200, 1300]
SYNTH = {"n": 40, "v": 1.7, "specks": 0.10, "seed": 11}


def load(name: str) -> np.ndarray:
    return np.genfromtxt(DATA / name, delimiter=",", names=True)


def velocities(table: np.ndarray) -> list[tuple[float, float, float]]:
    """(slope, intercept, r2) of average distance vs time for each temperature."""
    fits = []
    for temp in TEMPS:
        rows = table[table["temperature_K"] == temp]
        fits.append(fit_velocity(rows["time_ps"], rows["average_A"]))
    return fits


def nickel_distance_figure(nickel: np.ndarray) -> None:
    fig, ax = plots.new_figure(size=(7.5, 4.8))
    for temp, (slope, intercept, _), color in zip(TEMPS, velocities(nickel), plots.SERIES):
        rows = nickel[nickel["temperature_K"] == temp]
        keep = np.isfinite(rows["average_A"])
        v_ms = float(angstrom_per_ps_to_m_per_s(slope))
        plots.plot_distance_fit(ax, rows["time_ps"][keep], rows["average_A"][keep],
                                slope, intercept, color, f"{temp} K   v = {v_ms:.4f} m/s")
    ax.set_title("Nickel: average grain-boundary distance vs time", loc="left")
    plots.style_axes(ax)
    ax.legend(frameon=False, labelcolor=plots.INK)
    plots.save(fig, FIGURES / "nickel_distance_vs_time.png")


def arrhenius_figure(nickel: np.ndarray, nicocr: np.ndarray) -> dict[str, float]:
    fig, ax = plots.new_figure(size=(7, 4.6))
    energies = {}
    for label, table, color in [("Ni", nickel, plots.SERIES[0]),
                                ("NiCoCr (u0 = 0.08)", nicocr, plots.SERIES[1])]:
        v = np.array([fit[0] for fit in velocities(table)])
        ea, prefactor, _ = arrhenius_fit(np.array(TEMPS, float), v)
        plots.plot_arrhenius(ax, np.array(TEMPS), v, ea, prefactor, color, label)
        energies[label] = ea
    ax.set_title("Arrhenius plot of boundary velocity", loc="left")
    plots.style_axes(ax)
    ax.legend(frameon=False, labelcolor=plots.INK)
    plots.save(fig, FIGURES / "arrhenius_nickel_vs_nicocr.png")
    return energies


def show_frame(ax, img: np.ndarray, xs: list[float], title: str) -> None:
    ax.imshow(img)
    for x in xs:
        ax.axvline(x, color=plots.MARKER_RED, lw=1.2)
    ax.set_title(title, loc="left", color=plots.INK, fontsize=10)
    ax.set_axis_off()


def show_profile(ax, profile: np.ndarray, peaks: list[float], title: str) -> None:
    ax.plot(profile, color=plots.INK_MUTED, lw=1.2)
    for x in peaks:
        ax.axvline(x, color=plots.MARKER_RED, lw=1.2)
    ax.set_xlim(0, len(profile))
    ax.set_xlabel("image column (px)")
    ax.set_ylabel("white fraction")
    ax.set_title(title, loc="left", fontsize=10)
    plots.style_axes(ax)


def synthetic_figure() -> tuple[float, float]:
    frames = make_frames(SYNTH["n"], v_px_per_frame=SYNTH["v"],
                         speck_fraction=SYNTH["specks"], seed=SYNTH["seed"])
    positions = track_boundaries(frames)
    profile = column_profile(boundary_mask(frames[0]), sample_rows(frames[0]))
    frame_idx = np.arange(SYNTH["n"])
    dist = displacement_px(positions)[:, 0]
    slope, intercept, _ = fit_velocity(frame_idx, dist)
    fig, axes = plots.new_figure(2, 2, size=(11, 7))
    show_frame(axes[0, 0], frames[0], list(positions[0]), "Synthetic frame 0")
    show_frame(axes[0, 1], frames[-1], list(positions[-1]), f"Synthetic frame {SYNTH['n'] - 1}")
    show_profile(axes[1, 0], profile, find_boundaries(profile, n=2), "Column profile, frame 0")
    ax = axes[1, 1]
    plots.plot_distance_fit(ax, frame_idx, dist, slope, intercept, plots.SERIES[0],
                            f"recovered  {slope:.3f} px/frame")
    ax.plot(frame_idx, SYNTH["v"] * frame_idx, "--", lw=1.5, color=plots.INK_MUTED,
            label=f"true  {SYNTH['v']:.3f} px/frame")
    ax.set_xlabel("frame")
    ax.set_ylabel("middle GB distance (px)")
    ax.set_title("Recovered vs true velocity", loc="left", fontsize=10)
    plots.style_axes(ax)
    ax.legend(frameon=False, labelcolor=plots.INK)
    plots.save(fig, FIGURES / "synthetic_demo.png")
    return SYNTH["v"], slope


def show_position_profile(ax, profile: np.ndarray, centers: np.ndarray, title: str) -> None:
    half_bin = (centers[1] - centers[0]) / 2
    ax.plot(centers, profile, color=plots.INK_MUTED, lw=1.2)
    for x in find_profile_peaks(centers, profile, n=2):
        ax.axvline(x, color=plots.MARKER_RED, lw=1.5, clip_on=False)
        # A peak on the periodic edge sits on the plot border, so label every peak.
        ax.annotate(f"{x:.1f} A", (x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4 if x < centers[len(centers) // 2] else -4, -12),
                    textcoords="offset points", color=plots.INK, fontsize=9,
                    ha="left" if x < centers[len(centers) // 2] else "right")
    ax.set_xlim(centers[0] - half_bin, centers[-1] + half_bin)
    ax.set_xlabel("x (angstrom)")
    ax.set_ylabel("boundary-atom fraction")
    ax.set_title(title, loc="left", fontsize=10)
    plots.style_axes(ax)


def dump_demo_figure() -> tuple[float, float]:
    d = DUMP_DEMO
    frames = make_dump_frames(d["n_frames"], v_A_per_frame=d["v_A_per_frame"],
                              speck_fraction=d["speck_fraction"], seed=d["seed"],
                              side_v_A_per_frame=-d["v_A_per_frame"],
                              steps_per_frame=d["steps_per_frame"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.dmp.gz"
        write_dump(path, frames, DUMP_COLUMNS)
        profiles = list(dump_profiles(path))
        timesteps, positions = track_periodic(profiles)
    time_ps = timesteps * d["dt_ps"]
    dist = np.abs(displacement_px(positions)).mean(axis=1)
    slope, intercept, _ = fit_velocity(time_ps, dist)
    true_v = d["v_A_per_frame"] / (d["steps_per_frame"] * d["dt_ps"])
    fig, axes = plots.new_figure(1, 3, size=(15, 4.4))
    for ax, i, label in [(axes[0], 0, "frame 0"), (axes[1], -1, f"frame {d['n_frames'] - 1}")]:
        show_position_profile(ax, profiles[i][1], profiles[i][2], f"Boundary-atom x histogram, {label}")
    ax = axes[2]
    plots.plot_distance_fit(ax, time_ps, dist, slope, intercept, plots.SERIES[0],
                            f"recovered  {slope:.4f} A/ps")
    ax.plot(time_ps, true_v * time_ps, "--", lw=1.5, color=plots.INK_MUTED,
            label=f"true  {true_v:.4f} A/ps")
    ax.set_title("Average boundary distance vs time", loc="left", fontsize=10)
    plots.style_axes(ax)
    ax.legend(frameon=False, labelcolor=plots.INK)
    plots.save(fig, FIGURES / "dump_demo.png")
    return true_v, slope


def real_profile_figure() -> list[float]:
    names = ["ni_1000K_step000.png", "ni_1000K_step199.png"]
    fig, axes = plots.new_figure(2, 2, size=(12, 6.5), gridspec_kw={"height_ratios": [2, 1]})
    middles = []
    for col, name in enumerate(names):
        profile, (left, right) = frame_profile(OVITO / name)
        peaks = find_boundaries(profile, n=2)
        middle = min(peaks, key=lambda x: abs(x - (left + right) / 2))
        middles.append(middle)
        step = int(name.split("step")[1].split(".")[0])
        show_frame(axes[0, col], load_frame(OVITO / name), peaks,
                   f"Ni 1000 K, step {step}: middle GB at x = {middle:.1f} px")
        show_profile(axes[1, col], profile, peaks, "Column profile")
    plots.save(fig, FIGURES / "real_profile_1000K.png")
    return middles


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    nickel, nicocr = load("nickel_distances.csv"), load("nicocr_u008_distances.csv")
    nickel_distance_figure(nickel)
    energies = arrhenius_figure(nickel, nicocr)
    true_v, found_v = synthetic_figure()
    dump_true, dump_found = dump_demo_figure()
    middles = real_profile_figure()
    for label, ea in energies.items():
        print(f"Ea {label}: {ea:.4f} eV")
    print(f"synthetic image velocity: true {true_v:.4f}, recovered {found_v:.4f} px/frame")
    print(f"synthetic dump velocity: true {dump_true:.5f}, recovered {dump_found:.5f} A/ps")
    print(f"real 1000 K middle GB: {middles[0]:.1f} px -> {middles[1]:.1f} px")


if __name__ == "__main__":
    main()
