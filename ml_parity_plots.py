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
                   - the "System name"  -> used to group/colour the points
                   - "The number of atoms" (N) -> used to slice the flat force list
                                                   into per-structure blocks

The energies in ML_REG are *total* energies, so different-sized systems live at
very different values. By default we plot them per atom (eV/atom), which puts every
system on a comparable scale and matches the usual MLFF error metric.

Output: two figures, each overlaying all systems with a distinct colour and a legend
that reports the per-system RMSE:
    * parity_energy.png  -- reference vs. ML energy
    * parity_forces.png  -- reference vs. ML force components

All settings are hardcoded in the CONFIGURATION block below -- edit them there.
"""

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

# --- output files ------------------------------------------------------------------
ENERGY_PLOT_FILE = "parity_energy.png"
FORCES_PLOT_FILE = "parity_forces.png"

# --- energy handling ---------------------------------------------------------------
# ML_REG stores TOTAL energies (eV). With True, energies are divided by the number
# of atoms in each structure -> eV/atom (recommended, systems become comparable).
ENERGY_PER_ATOM = True

# --- figure appearance -------------------------------------------------------------
FIG_SIZE = (7.5, 7.0)      # inches, per figure
DPI = 200
MARKER_SIZE_ENERGY = 26    # scatter point size for the energy plot (617 pts here)
MARKER_SIZE_FORCE = 6      # scatter point size for the force plot (~300k pts here)
POINT_ALPHA_ENERGY = 0.85
POINT_ALPHA_FORCE = 0.35   # forces have many points -> use transparency
MARKER_EDGE_WIDTH = 0.3    # thin dark edge keeps light colours visible on white

# Colour-blind-safe categorical palette (re-ordered Okabe-Ito). Assigned to system
# names in fixed order; if there are more systems than colours it cycles.
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


def make_parity_figure(groups, order, colors, title, axis_label,
                       rmse_unit, rmse_scale, rmse_fmt, marker_size,
                       alpha, out_file):
    """Draw one parity plot.

    groups     : dict name -> (ref_array, ml_array)
    order      : iterable of names giving legend/plot order
    colors     : dict name -> colour
    rmse_scale : multiply RMSE (in axis units) to get it in rmse_unit
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    all_vals = []
    for name in order:
        ref, ml = groups[name]
        all_vals.append(ref)
        all_vals.append(ml)
    all_vals = np.concatenate(all_vals)
    lo, hi = all_vals.min(), all_vals.max()
    pad = 0.03 * (hi - lo) if hi > lo else 1.0
    lo, hi = lo - pad, hi + pad

    # y = x reference line (drawn first, sits underneath the points)
    ax.plot([lo, hi], [lo, hi], "--", color=DIAGONAL_COLOR, lw=1.2,
            zorder=1, label="_nolegend_")

    # overall RMSE across every point, shown in the corner
    ref_all = np.concatenate([groups[n][0] for n in order])
    ml_all = np.concatenate([groups[n][1] for n in order])
    overall = rmse(ref_all, ml_all) * rmse_scale

    for name in order:
        ref, ml = groups[name]
        r = rmse(ref, ml) * rmse_scale
        label = f"{name}  (RMSE {rmse_fmt.format(r)} {rmse_unit}, n={ref.size})"
        ax.scatter(
            ref, ml,
            s=marker_size, alpha=alpha, color=colors[name],
            edgecolors="black", linewidths=MARKER_EDGE_WIDTH,
            label=label, rasterized=(ref.size > 5000), zorder=2,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Reference / DFT  {axis_label}")
    ax.set_ylabel(f"ML predicted  {axis_label}")
    ax.set_title(title)
    ax.grid(True, color="#dddddd", lw=0.6, zorder=0)
    ax.text(
        0.03, 0.97,
        f"Overall RMSE: {rmse_fmt.format(overall)} {rmse_unit}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.9),
    )
    leg = ax.legend(
        loc="lower right", frameon=True, fontsize=8.5,
        title="System  (per-structure grouping)", title_fontsize=9,
        markerscale=1.6 if marker_size < 12 else 1.0,
    )
    leg.get_frame().set_edgecolor("#cccccc")

    fig.tight_layout()
    fig.savefig(out_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_file}")
    return overall

# =====================================================================================
# main
# =====================================================================================

def main():
    print(f"Reading ML_AB  : {ML_AB_FILE}")
    configs = parse_ml_ab(ML_AB_FILE)              # [(name, natoms), ...]
    names = [c[0] for c in configs]
    atoms = np.array([c[1] for c in configs], dtype=int)
    n_config = len(configs)
    print(f"  {n_config} configurations, {len(unique_in_order(names))} distinct systems")

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

    colors = color_map(names)
    order = list(colors.keys())  # first-appearance order

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

    # ---- draw ---------------------------------------------------------------------
    print("Plotting:")
    e_overall = make_parity_figure(
        energy_groups, order, colors,
        title="Energy parity  (ML force field vs. DFT)",
        axis_label=energy_axis_label,
        rmse_unit=energy_rmse_unit, rmse_scale=energy_rmse_scale,
        rmse_fmt=energy_rmse_fmt, marker_size=MARKER_SIZE_ENERGY,
        alpha=POINT_ALPHA_ENERGY, out_file=ENERGY_PLOT_FILE,
    )
    f_overall = make_parity_figure(
        force_groups, order, colors,
        title="Force-component parity  (ML force field vs. DFT)",
        axis_label="force (eV/Å)",
        rmse_unit="eV/Å", rmse_scale=1.0,
        rmse_fmt="{:.4f}", marker_size=MARKER_SIZE_FORCE,
        alpha=POINT_ALPHA_FORCE, out_file=FORCES_PLOT_FILE,
    )

    print("\nSummary (overall RMSE):")
    print(f"  energy : {energy_rmse_fmt.format(e_overall)} {energy_rmse_unit}")
    print(f"  forces : {f_overall:.4f} eV/Å")


if __name__ == "__main__":
    main()
