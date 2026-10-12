# Grain Boundary Velocity Tracker

A small tool that measures how fast a grain boundary moves in an atomistic simulation. It reads the same LAMMPS dump file OVITO opens, tracks the boundary frame by frame, and fits a velocity. Point it at runs from several temperatures and it also fits the Arrhenius line for the activation energy.

I wrote it during my research assistantship, where I was running hybrid Monte Carlo / molecular dynamics simulations (LAMMPS) of a NiCoCr bicrystal to see how alloying and chemical ordering change how easily a grain boundary moves. This is the cleaned-up version, posted with the data and figures from that work.

![How the tracker works on a synthetic trajectory](figures/dump_demo.png)

## Why I wrote it

Every simulation gives you a trajectory of 200 frames, and the one number you actually want out of it is a slope: how far the boundary moved versus how much time passed.

At first I got that number by hand. Open the trajectory in OVITO, step forward some fixed number of frames, measure where the boundary is now with the measuring tool, type the distance into a spreadsheet, repeat about twenty times, then fit a line. Roughly an hour per run, and I had four temperatures per alloy and several alloys.

So I automated it. The tool reads the trajectory, finds the boundary in every Nth frame, records its position in angstrom, and fits the line. A 40-frame trajectory takes under three seconds. After that, adding another temperature or re-running a condition was no longer something I had to think about.

## How it works

1. **It reads the LAMMPS dump directly.** No exporting, no screenshots. My LAMMPS runs already wrote a per-atom structure type from polyhedral template matching (`v_ptm_type`). Atoms that aren't FCC are the boundary, which is the same thing OVITO is showing you when it paints them white. If the dump has no structure column the tool falls back to centrosymmetry (`c_csym`), and if it has neither it tells you which column to add to your dump command.
2. **It finds the boundaries.** The boundary atoms get binned along the migration axis. Since a grain boundary is a plane, its atoms pile into a few bins and make a sharp peak, while the thermal disorder you get at high temperature just raises the floor a bit. There are two peaks to track in a bicrystal: the boundary in the middle and the one at the periodic edge of the box.
3. **It tracks and fits.** Each boundary is followed frame to frame, and the box is treated as periodic, so a boundary crossing the edge keeps a continuous distance. Distance versus time goes through a least-squares fit for the velocity in angstrom/ps (1 angstrom/ps is 100 m/s). `gbtrack arrhenius` then fits ln(v) against 1/T across temperatures to get the activation energy.

```
gbtrack run mcmd.sro.dmp.gz --every 10 --dt-ps 0.002 --out results/1000K
gbtrack arrhenius velocities.csv --out results
```

## Results from the study

These are the numbers from my spreadsheet at the time, re-fit with this tool.

![Nickel distance vs time](figures/nickel_distance_vs_time.png)

| Alloy | 1000 K | 1100 K | 1200 K | 1300 K |
|---|---|---|---|---|
| Pure nickel | 0.029 m/s | 0.041 m/s | 0.055 m/s | 0.066 m/s |
| NiCoCr, random solid solution | 0.137 m/s | 0.147 m/s | 0.169 m/s | 0.195 m/s |

Alloying made the boundary about 4.7x faster at 1000 K. The Arrhenius fit gives an activation energy of 0.31 eV for nickel and 0.13 eV for the random NiCoCr alloy. In the full study, adding chemical order to the NiCoCr roughly tripled that barrier compared to the random alloy.

![Arrhenius fit](figures/arrhenius_nickel_vs_nicocr.png)

## It works on the pictures too

If all you have are OVITO renders, `gbtrack images` does the same peak-finding on the image: whitish pixels count as boundary, a column profile stands in for the atom histogram, and you give it a scale in angstrom per pixel. Below are the real 1000 K nickel frames at step 0 and step 199 with the boundaries it found. Even at 1400 K, where thermal noise covers the sample in white specks, the boundary's columns come out around 0.75 white against a background near 0.2, so it still stands out.

![Real OVITO frames with detected boundaries](figures/real_profile_1000K.png)

## Checking that it's right

`gbtrack demo` builds a synthetic trajectory with a known boundary velocity and an adjustable amount of thermal noise, then runs the tracker on it. It recovers the velocity to within 1% on the dump path (0.0347 vs 0.0350 angstrom/ps with 10% noise atoms, including a boundary that crosses the periodic edge) and within 0.1% on the image path. There are 61 pytest tests covering the dump parser, how boundary atoms are chosen, peak finding (including a peak sitting right on the periodic edge), tracking with unwrapping, the velocity and Arrhenius fits against the study's recorded numbers, and the real OVITO frames.

The lab's trajectory files aren't included here, so the dump path is checked against synthetic trajectories and the recorded distances from the study, and the image path against the real renders.

## Stack

Python, NumPy, Pillow, matplotlib, pytest. Input is a LAMMPS dump trajectory, plain or gzipped, which is what OVITO loads for these runs.
