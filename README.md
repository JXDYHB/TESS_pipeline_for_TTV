# TESS Pipeline for TTV

End-to-end pipeline for fitting TESS transit times and searching for Transit
Timing Variations (TTVs) in warm-Jupiter candidates.

The pipeline downloads TESS light curves with
[lightkurve](https://docs.lightkurve.org), selects per-transit cutout windows
using Transit Least Squares (TLS) plus Gaussian dip refinement, and jointly fits
the transit shape and per-window mid-transit times with NumPyro / JAX using
[jaxoplanet](https://jax.exoplanet.codes) and
[tinygp](https://tinygp.readthedocs.io).

The post-processing step estimates a linear ephemeris by weighted maximum
likelihood and compares it with a sinusoidal TTV model,

$$
t_n =
T_0 + nP +
A \sin\left(
\frac{2\pi (T_0+nP)}{P_{\mathrm{TTV}}}+ \varphi
\right),
$$

where $n$ is the transit epoch, $T_0$ is the reference mid-transit time,
$P$ is the linear orbital period, $A$ is the TTV amplitude,
$P_{\mathrm{TTV}}$ is the sinusoidal TTV period, and $\varphi$ is the phase.
The pipeline reports the improvement in $\chi^2$ relative to the linear
ephemeris.

## Pipeline overview

For each target, the pipeline performs the following steps:

1. Read one target from `data/WJs.csv`.
2. Download available TESS light curves using `lightkurve`.
3. Identify candidate transit windows using TLS and/or the catalogue ephemeris.
4. Refine the approximate transit centers using a Gaussian dip template.
5. Fit all selected windows jointly with a shared transit-shape model and
   per-window mid-transit times.
6. Estimate the best-fit linear ephemeris from the fitted mid-transit times.
7. Construct the O-C diagram relative to the linear ephemeris.
8. Fit an optional sinusoidal TTV model and report the $\Delta\chi^2$
   improvement.

The main scientific output is a set of fitted mid-transit times, their
uncertainties, an O-C diagram, and a quantitative comparison between a linear
ephemeris and a sinusoidal TTV model.

## Layout

```
TTV_Analysis_For_TESS/
├── ppl.py                        # CLI entry point (one CSV row per call)
├── pyproject.toml + uv.lock      # pinned dependencies
├── scripts/mcmc_ast.sbatch       # SLURM array template
└── tesspipe/
    ├── download.py               # lightkurve fetch with per-author fallback
    ├── transit_window.py         # TLS + ephemeris-driven window selection
    ├── model_build.py            # NumPyro joint model + MCMC runner
    ├── data_display.py           # post-processing (linear + sinusoidal TTV)
    ├── env_utils.py              # typed env-var readers
    ├── arviz_runtime.py          # fcntl-based ArviZ import lock for SLURM
    └── modes/                    # one file per baseline model variant
        ├── mode0.py              # quadratic baseline + white noise
        ├── mode1.py              # per-window tinygp Matérn-3/2 baseline
        ├── mode3.py              # mode 1 + per-window duration
        └── noise.py              # hierarchical white-noise blocks (shared)
```

## Output layout

```
OUTPUT/
├── slurm/                                 # raw SLURM logs (gitignored)
└── tic_<TIC>_index_<INDEX>/mode<M>/
    ├── latest -> runs/<latest TS>/
    └── runs/<YYYYMMDD-HHMMSS>/
        ├── idata.pkl, windows.pkl
        ├── summary.json                   # ephemeris, χ², sine fit, file refs
        ├── oc.png                         # linear-ephemeris O-C
        ├── oc_sinusoidal.png              # sinusoidal TTV overlay
        ├── corner.png                     # posterior pair plot
        ├── fit/window_NNN.png             # per-transit fit diagnostics
        └── slurm/slurm_<JOB>_<INDEX>.{log,err}
```

## Installation

Tested on Python 3.12.  Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Or with plain pip:

```bash
pip install -e .
```

## Catalogue input

`ppl.py` reads `data/WJs.csv` (excluded from the repo).  The required columns
are:

| column            | meaning                                |
|-------------------|----------------------------------------|
| `TIC`             | TESS Input Catalogue ID                |
| `P`               | orbital period (days)                  |
| `Duration (hrs)`  | transit duration (hours)               |
| `Depth (ppm)`     | transit depth in ppm                   |
| `Depth_err (ppm)` | 1-σ uncertainty on depth               |
| any of `Tc_BTJD`, `t0`, `T0`, `Epoch`, … | catalogue t0 prior (BTJD or BJD) |

## Usage

Run one target end-to-end:

```bash
python ppl.py --index 17 --mode 1
```

`--mode` choices:

- `0` — quadratic baseline + white noise
- `1` — tinygp per-window GP + shared transit shape (default)
- `3` — mode 1 + per-window transit duration

Optional `--manual-t0 3622.76,2904.02,...` skips TLS and uses the given BTJD
transit times directly.

Re-run only the post-processing on an existing run (no MCMC redo):

```bash
python -m tesspipe.data_display \
    --run-dir OUTPUT/tic_334811204_index_17/mode1/latest
```

## SLURM batch usage

Edit `scripts/mcmc_ast.sbatch` and set `#SBATCH --array=N1,N2,...` to the rows
in `WJs.csv` you want to process, optionally override `MODE` via env var, and:

```bash
sbatch scripts/mcmc_ast.sbatch
```

Each array task produces its own `tic_<TIC>_index_<INDEX>/mode<M>/runs/<TS>/`
directory and updates the `latest` symlink atomically once finished.

## Environment-variable overrides

| variable                     | default | effect                                                       |
|------------------------------|---------|--------------------------------------------------------------|
| `NUMPYRO_NUM_CHAINS`         | 2       | number of MCMC chains                                        |
| `RUN_TIMESTAMP`              | now     | force a specific run-directory timestamp                     |
| `TLS_MIN_DURATION_COVERAGE`  | 0.85    | reject window when < 85% of duration is observed             |
| `TLS_MIN_TC_SCORE`           | 0.02    | reject window when the Gaussian-template score is below this |
| `TLS_MAX_TC_SHIFT_FRAC`      | 0.5     | reject window if `\|tc_fit − tc_linear\| > frac × duration`    |
| `TLS_EPHEM_ONLY`             | 0       | skip the TLS global search when an ephemeris is given        |
| `TLS_USE_EPHEM_PRIOR`        | 1       | seed window selection from the catalogue ephemeris           |
| `DOWNLOAD_MAX_TRIES`         | 3       | per-sector download retry budget                             |

## Output: `summary.json`

```jsonc
{
  "tic_id": 334811204,
  "p_prior": 21.07766,
  "ephemeris": {
    "p_grid": 21.07665, "p_grid_min": …, "p_grid_max": …,
    "p_mle":  21.077634, "p_mle_err": 2.1e-05,
    "t0_mle": 1743.999927, "t0_mle_err": 0.000816,
    "chi2": 7.52, "dof": 8, "redchi2": 0.94,
    "chi2_kept": 7.52, "dof_kept": 8, "redchi2_kept": 0.94
  },
  "sinusoidal_ttv": {
    "T0": 1744.0, "P": 21.077, "A_minutes": 4.66,
    "P_ttv_days": 48.4, "phi_rad": 0.41,
    "n_free_params": 5,
    "chi2": 0.5, "dof": 3, "redchi2": 0.17,
    "delta_chi2_vs_linear": 7.02,
    "lm_converged": true
  },
  "windows": { "n_total": 8, "n_kept": 8 },
  "files": { … }
}
```

