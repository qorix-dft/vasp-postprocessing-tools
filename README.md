# vasp-postprocessing-tools
Python scripts for parsing and visualizing VASP output files.

## `ml_parity_plots.py`

Parity plots (ML predicted vs. DFT reference) for a machine-learning force-field
training run. Reads `ML_REG` for the fitted-vs-ab-initio values and `ML_AB` to
group configurations by their `System name` and recover per-structure atom counts.

Produces two figures — `parity_energy.png` and `parity_forces.png` — each
overlaying every system in a distinct colour with a legend reporting the
per-system RMSE.

```bash
pip install numpy matplotlib
cd /path/to/your/vasp/run   # directory containing ML_REG and ML_AB
python3 /path/to/ml_parity_plots.py
```

All settings (file names, per-atom vs. total energy, colours, marker sizes) are
hardcoded in the `CONFIGURATION` block at the top of the script.
