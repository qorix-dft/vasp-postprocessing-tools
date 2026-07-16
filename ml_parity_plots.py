#!/usr/bin/env python3
"""
ml_parity_plots.py

Create parity plots (predicted vs. reference) of energies and forces from a VASP
machine-learning force-field training run.

It reads two files produced by VASP:

  * ML_REG  -- the regression file. It holds, in this exact order and format
               (see https://vasp.at/wiki/ML_REG):

                   Total energies (eV)   : 1 value per training structure
                                           col 1 = ab-initio (DFT), col 2 = fitted (ML)
                   Forces (eV ang.^-1)   : 3*N values per structure (x,y,z of every atom),
                                           structures listed one after another
                   Stress (kbar)         : 6 values per structure (not plotted here)

  * ML_AB   -- the training data file (see https://vasp.at/wiki/ML_AB). Used only
               to recover, for every configuration and *in the same order as ML_REG*:
                   - the "System name"  -> used to group the points
                   - "The number of atoms" (N) -> used to slice the flat force list
                                                   into per-structure blocks

Output: ONE energy panel and ONE force panel *per system* (grouped by System name),
so every system can be inspected on its own zoomed axes. The number of images is
therefore 2 * (number of distinct systems). Each panel's title reports its RMSE.
Files are named   parity_energy_<system>.png   and   parity_forces_<system>.png .

The energies in ML_REG are *total* energies, so different-sized systems live at
very different values. By default we plot them per atom (eV/atom), which is the
usual MLFF error metric.

All settings are hardcoded in the CONFIGURATION block below -- edit them there.
"""

import os
import re
import sys
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no interactive display needed; write files directly
import matplotlib.pyplot as plt

# =====================================================================================
# CONFIGURATION -- everything hardcoded here
# =====================================================================================

# --- input files -------------------------------------------------------------------
ML_REG_FILE = "ML_REG"
ML_AB_FILE = "ML_AB"

# --- output ------------------------------------------------------------------------
OUTPUT_DIR = "parity_plots"                     # created if missing
ENERGY_FILE_TEMPLATE = "parity_energy_{system}.png"
FORCES_FILE_TEMPLATE = "parity_forces_{system}.png"

# --- energy handling ---------------------------------------------------------------
# ML_REG stores TOTAL energies (eV). With True, energies are divided by the number
# of atoms in each structure -> eV/atom (recommended, systems become comparable).
ENERGY_PER_ATOM = True

# --- figure appearance -------------------------------------------------------------
FIG_SIZE = (6.5, 6.2)      # inches, per panel
DPI = 200
MARKER_SIZE_ENERGY = 32    # scatter point size for the energy panels
MARKER_SIZE_FORCE = 6      # scatter point size for the force panels (dense)
POINT_ALPHA_ENERGY = 0.85
POINT_ALPHA_FORCE = 0.30   # forces have many points -> use transparency
MARKER_EDGE_WIDTH = 0.3    # thin dark edge keeps light colours visible on white

# Colour-blind-safe categorical palette (re-ordered Okabe-Ito). Each system keeps
# the same colour across its energy and force panels; cycles if there are more
# systems than colours.
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#56B4E9",  # sky blue
    "#CC79A7",  # pink
    "#D55E00",  # vermillion
    "#000000",  # black
]

DIAGONAL_COLOR = "#888888"  # y = x reference line

# =====================================================================================
# ML_AB parsing
# =====================================================================================

def parse_ml_ab(path):
    """Return a list of (system_name, n_atoms) tuples, one per configuration,
    in file order. Robust to the exact number of blank/separator lines: it keys
    off the literal header labels used in ML_AB."""
    names = []
    natoms = []
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError as exc:
        sys.exit(f"ERROR: could not read ML_AB file '{path}': {exc}")

    i = 0
    n = len(lines)
    while i < n:
        label = lines[i].strip()
        # The value always sits two lines below the header:
        #   <label>
        #   ----------
        #   <value>
        if label == "System name":
            names.append(lines[i + 2].strip())
            i += 3
            continue
        # NB: must be the exact string -- "The number of atom types" also starts
        # with "The number of atom", so we compare the whole stripped line.
        if label == "The number of atoms":
            natoms.append(int(lines[i + 2].split()[0]))
            i += 3
            continue
        i += 1

    if len(names) != len(natoms):
        sys.exit(
            f"ERROR: ML_AB parse mismatch: {len(names)} 'System name' entries but "
            f"{len(natoms)} 'The number of atoms' entries."
        )
    if not names:
        sys.exit(f"ERROR: no configurations found in '{path}'. Is it an ML_AB file?")
    return list(zip(names, natoms))

# =====================================================================================
# ML_REG parsing
# =====================================================================================

def _is_data_row(line):
    """A data row is two floating-point numbers (col1 = ab-initio, col2 = fitted)."""
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def parse_ml_reg(path):
    """Return three (M, 2) numpy arrays: energies, forces, stress.
    Column 0 = ab-initio (reference/DFT), column 1 = fitted (ML)."""
    sections = {"energy": [], "force": [], "stress": []}
    current = None
    try:
        with open(path) as fh:
            for line in fh:
                stripped = line.strip()
                low = stripped.lower()
                # Section switches, identified by the header labels in ML_REG.
                if low.startswith("total energies"):
                    current = "energy"
                    continue
                if low.startswith("forces"):
                    current = "force"
                    continue
                if low.startswith("stress"):
                    current = "stress"
                    continue
                if current is None:
                    continue
                row = _is_data_row(line)
                if row is not None:
                    sections[current].append(row)
    except OSError as exc:
        sys.exit(f"ERROR: could not read ML_REG file '{path}': {exc}")

    energies = np.array(sections["energy"], dtype=float)
    forces = np.array(sections["force"], dtype=float)
    stress = np.array(sections["stress"], dtype=float)
    if energies.size == 0:
        sys.exit(f"ERROR: no 'Total energies' data found in '{path}'.")
    return energies, forces, stress

# =====================================================================================
# helpers
# =====================================================================================

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def unique_in_order(seq):
    return list(OrderedDict.fromkeys(seq))


def color_map(system_names):
    """Map each unique system name to a colour, in first-appearance order."""
    uniq = unique_in_order(system_names)
    return OrderedDict(
        (name, PALETTE[k % len(PALETTE)]) for k, name in enumerate(uniq)
    )


def safe_filename(name):
    """Make a system name safe to use in a file name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def make_panel(ref, ml, color, system, kind_title, axis_label,
               rmse_value, rmse_unit, rmse_fmt, marker_size, alpha, out_file):
    """Draw one parity panel for a single system (single data series)."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    both = np.concatenate([ref, ml])
    lo, hi = both.min(), both.max()
    pad = 0.03 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    # y = x reference line (drawn first, underneath the points)
    ax.plot([lo, hi], [lo, hi], "--", color=DIAGONAL_COLOR, lw=1.2, zorder=1)

    ax.scatter(
        ref, ml,
        s=marker_size, alpha=alpha, color=color,
        edgecolors="black", linewidths=MARKER_EDGE_WIDTH,
        rasterized=(ref.size > 5000), zorder=2,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Reference / DFT  {axis_label}")
    ax.set_ylabel(f"ML predicted  {axis_label}")
    ax.set_title(f"{system}  —  {kind_title}")
    ax.grid(True, color="#dddddd", lw=0.6, zorder=0)
    ax.text(
        0.03, 0.97,
        f"RMSE {rmse_fmt.format(rmse_value)} {rmse_unit}\nn = {ref.size}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_file}")

# =====================================================================================
# main
# =====================================================================================

def main():
    print(f"Reading ML_AB  : {ML_AB_FILE}")
    configs = parse_ml_ab(ML_AB_FILE)              # [(name, natoms), ...]
    names = [c[0] for c in configs]
    atoms = np.array([c[1] for c in configs], dtype=int)
    n_config = len(configs)
    colors = color_map(names)
    order = list(colors.keys())                    # first-appearance order
    print(f"  {n_config} configurations, {len(order)} distinct systems")

    print(f"Reading ML_REG : {ML_REG_FILE}")
    energies, forces, stress = parse_ml_reg(ML_REG_FILE)
    print(f"  energy rows={len(energies)}  force rows={len(forces)}  stress rows={len(stress)}")

    # ---- consistency checks between the two files ---------------------------------
    if len(energies) != n_config:
        sys.exit(
            f"ERROR: ML_REG has {len(energies)} energy entries but ML_AB has "
            f"{n_config} configurations. The two files must come from the same run."
        )
    expected_force_rows = int(3 * atoms.sum())
    if len(forces) != expected_force_rows:
        sys.exit(
            f"ERROR: ML_REG has {len(forces)} force rows but ML_AB implies "
            f"3*sum(N) = {expected_force_rows}. The two files must match."
        )

    # ---- ENERGIES: one point per configuration ------------------------------------
    e_ref = energies[:, 0].copy()
    e_ml = energies[:, 1].copy()
    if ENERGY_PER_ATOM:
        e_ref = e_ref / atoms
        e_ml = e_ml / atoms
        energy_axis_label = "energy (eV/atom)"
        energy_rmse_unit = "meV/atom"
        energy_rmse_scale = 1000.0          # eV/atom -> meV/atom
        energy_rmse_fmt = "{:.2f}"
    else:
        energy_axis_label = "total energy (eV)"
        energy_rmse_unit = "eV"
        energy_rmse_scale = 1.0
        energy_rmse_fmt = "{:.4f}"

    # group energy points by system name
    energy_groups = {}
    for name in order:
        mask = np.array([nm == name for nm in names])
        energy_groups[name] = (e_ref[mask], e_ml[mask])

    # ---- FORCES: slice the flat list into per-structure blocks of 3*N -------------
    # block i owns rows [start_i : start_i + 3*N_i); every component inherits the
    # system name of its parent configuration.
    starts = np.concatenate(([0], np.cumsum(3 * atoms)))
    force_ref_by_sys = {name: [] for name in order}
    force_ml_by_sys = {name: [] for name in order}
    for i, name in enumerate(names):
        s, e = starts[i], starts[i + 1]
        force_ref_by_sys[name].append(forces[s:e, 0])
        force_ml_by_sys[name].append(forces[s:e, 1])
    force_groups = {
        name: (np.concatenate(force_ref_by_sys[name]),
               np.concatenate(force_ml_by_sys[name]))
        for name in order
    }

    # ---- draw one energy panel and one force panel per system ---------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Plotting into '{OUTPUT_DIR}/' ({2 * len(order)} images):")
    summary = []
    for name in order:
        color = colors[name]
        safe = safe_filename(name)

        e_r, e_m = energy_groups[name]
        e_rmse = rmse(e_r, e_m) * energy_rmse_scale
        make_panel(
            e_r, e_m, color, name, "Energy parity (ML vs. DFT)",
            energy_axis_label, e_rmse, energy_rmse_unit, energy_rmse_fmt,
            MARKER_SIZE_ENERGY, POINT_ALPHA_ENERGY,
            os.path.join(OUTPUT_DIR, ENERGY_FILE_TEMPLATE.format(system=safe)),
        )

        f_r, f_m = force_groups[name]
        f_rmse = rmse(f_r, f_m)
        make_panel(
            f_r, f_m, color, name, "Force-component parity (ML vs. DFT)",
            "force (eV/Å)", f_rmse, "eV/Å", "{:.4f}",
            MARKER_SIZE_FORCE, POINT_ALPHA_FORCE,
            os.path.join(OUTPUT_DIR, FORCES_FILE_TEMPLATE.format(system=safe)),
        )
        summary.append((name, e_r.size, e_rmse, f_r.size, f_rmse))

    # ---- per-system summary table -------------------------------------------------
    print("\nPer-system RMSE:")
    print(f"  {'system':<16} {'n_struct':>8}  {'E-RMSE':>12}   {'n_force':>9}  {'F-RMSE (eV/Å)':>14}")
    for name, ne, er, nf, fr in summary:
        print(f"  {name:<16} {ne:>8}  {energy_rmse_fmt.format(er):>8} {energy_rmse_unit:<4}"
              f"   {nf:>9}  {fr:>14.4f}")


if __name__ == "__main__":
    main()
