"""Pipeline entry point — run one row from WJs.csv through MCMC.

Usage
-----
    python ppl.py --index 5 --mode 1
    python ppl.py --index 5 --mode 1 --manual-t0 3622.76,2904.02

Environment variables
---------------------
NUMPYRO_NUM_CHAINS   number of MCMC chains (default: 2)
RUN_TIMESTAMP        override the timestamp used for result directories
"""

import argparse
import os
import pickle
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

print = partial(print, flush=True)
os.environ.setdefault("TESSPIPE_IMPORT_T0", str(time.perf_counter()))

CSV_PATH    = "data/WJs.csv"
OUTPUT_ROOT = Path("OUTPUT")

# Output layout:
#   OUTPUT/
#   └── tic_<TIC>_index_<INDEX>/mode<M>/
#       ├── latest -> runs/<latest TS>/
#       └── runs/<YYYYMMDD-HHMMSS>/
#           ├── idata.pkl, windows.pkl, summary.json
#           ├── oc.png, corner.png
#           └── fit/window_NNN.png

# Column names that may hold the catalogue t0 value.
T0_COLUMNS = (
    "Tc_BTJD", "TOI List Tc_BTJD", "t0", "T0", "Tc", "Epoch",
    "T0 (BTJD)", "t0_btjd", "T0_BTJD", "T0 (BJD)", "t0_bjd",
)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _to_positive_float(value) -> float | None:
    """Return a positive float, or None for missing / non-positive values."""
    try:
        out = float(value)
    except Exception:
        return None
    return out if pd.notna(out) and out > 0 else None


def _find_t0_prior(row) -> float | None:
    """Extract the t0 prior from known column names, with fuzzy fallback."""
    for key in T0_COLUMNS:
        if key in row.index:
            try:
                v = float(row.get(key))
                if pd.notna(v):
                    return v
            except Exception:
                continue
    for key in row.index:
        key_l = str(key).lower()
        if "t0" in key_l or "epoch" in key_l:
            try:
                v = float(row.get(key))
                if pd.notna(v):
                    return v
            except Exception:
                continue
    return None


def _parse_manual_t0(text: str | None) -> list | None:
    """Parse a comma-separated BTJD string into a float list."""
    if text is None:
        return None
    out = [float(s.strip()) for s in str(text).split(",") if s.strip()]
    return out if out else None


def _import_elapsed_s() -> float:
    """Seconds since TESSPIPE_IMPORT_T0 was set."""
    try:
        return time.perf_counter() - float(os.environ["TESSPIPE_IMPORT_T0"])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _make_run_dir(tic_id: int, row_index: int, mode: int, run_timestamp: str) -> Path:
    """Build (and create) the canonical run directory for this row+target+mode."""
    target_dir = f"tic_{tic_id}_index_{row_index}"
    run_dir = OUTPUT_ROOT / target_dir / f"mode{mode}" / "runs" / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _update_latest_symlink(run_dir: Path) -> None:
    """Point ``<target>/<mode>/latest`` at the newest run.

    Uses a relative target so the symlink keeps working if the OUTPUT/
    directory is moved or mounted elsewhere.
    """
    latest = run_dir.parent.parent / "latest"
    target = Path("runs") / run_dir.name
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(target)


def _save_pickle(obj, path: Path) -> None:
    """Pickle ``obj`` to ``path`` and print one short confirmation line."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Saved] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """End-to-end pipeline: download → MCMC → post-process — all in-process."""
    parser = argparse.ArgumentParser(description="Run one WJs.csv row through the TESS transit pipeline.")
    parser.add_argument("--index", type=int, required=True,
                        help="Row index in data/WJs.csv (0-based).")
    parser.add_argument(
        "--mode", type=int, default=1, choices=[0, 1, 3],
        help=(
            "Baseline model: "
            "0 = quadratic baseline + white noise (shared transit shape); "
            "1 = tinygp per-window GP + shared transit shape; "
            "3 = tinygp per-window GP + independent duration per window."
        ),
    )
    parser.add_argument(
        "--manual-t0", type=str, default=None,
        help="Comma-separated BTJD transit times, e.g. '3622.76,2904.02'. "
             "When provided, skips TLS and uses these times directly.",
    )
    args = parser.parse_args()

    row_index   = int(args.index)
    mode        = int(args.mode)
    manual_time = _parse_manual_t0(args.manual_t0)

    num_chains    = max(1, int(os.getenv("NUMPYRO_NUM_CHAINS", "2")))
    run_timestamp = os.getenv("RUN_TIMESTAMP") or datetime.now().strftime("%Y%m%d-%H%M%S")

    # --- 0. Catalogue row -> TIC, priors, output paths ---------------------
    df = pd.read_csv(CSV_PATH)
    if not (0 <= row_index < len(df)):
        raise IndexError(f"Index {row_index} out of bounds for CSV with {len(df)} rows")
    row    = df.iloc[row_index]
    tic_id = int(row["TIC"])
    run_dir = _make_run_dir(tic_id, row_index, mode, run_timestamp)

    depth_ppm      = float(row["Depth (ppm)"])
    depth_err_ppm  = float(row["Depth_err (ppm)"])
    duration_hours = _to_positive_float(row.get("Duration (hrs)", None))
    period_days    = _to_positive_float(row.get("P", None))
    t0_days_prior  = _find_t0_prior(row)

    window_days = (3.0 * duration_hours / 24.0) if duration_hours is not None else 0.5
    if duration_hours is None:
        print("[WARN] Duration (hrs) missing or invalid; using fallback window=0.5 days")

    depth_lo = max((depth_ppm - depth_err_ppm) / 1e6, 1e-9)
    depth_hi = (depth_ppm + depth_err_ppm) / 1e6

    print(f"[INFO] TIC {tic_id}  mode={mode}  chains={num_chains}")
    print(f"[INFO] run_dir = {run_dir}")

    # --- 1. Download TESS light curves -------------------------------------
    from tesspipe.download    import download_sector
    from tesspipe.model_build import build_model_mcmc
    print(f"[import] tesspipe ok (+{_import_elapsed_s():.1f}s)")

    print(f"[run] downloading TIC {tic_id} ...")
    data_all = download_sector(int(tic_id), author="AUTO", sectors=None,
                               target_exptime=None, verbose=True)
    if not data_all:
        raise RuntimeError(f"download_sector returned no products for TIC {tic_id}")

    # Remap (sector, exptime) -> (sector, tic_label) as build_model_mcmc expects.
    TIC_LABEL   = 120.0
    sector_data = {(int(sec), TIC_LABEL): lc for (sec, _exp), lc in data_all.items()}
    sectors     = sorted({s for (s, _) in sector_data.keys()})
    sector_exptimes = {int(s): float(e) for (s, e) in data_all.keys()}
    print(f"[run] downloaded {len(sector_data)} sector(s): {sectors}")
    print(f"[run] per-sector exptime (s): {sector_exptimes}")

    # --- 2. MCMC -----------------------------------------------------------
    windows, mcmc, samples, idata, summary = build_model_mcmc(
        sector_data=sector_data,
        tic=TIC_LABEL,
        window=window_days,
        sectors=sectors,
        max_windows_per_sector=10,
        depth_lo=depth_lo, depth_hi=depth_hi,
        duration_hours=duration_hours,
        period_days_prior=period_days,
        t0_days_prior=t0_days_prior,
        time=manual_time,
        min_duration_coverage=0.85,
        debug_windows=True,
        num_warmup=1500, num_samples=1000, num_chains=num_chains,
        target_accept_prob=0.92, max_tree_depth=10,
        rng_seed=0, platform="cpu", enable_x64=False,
        use_catalog_duration_prior=True,
        mode=mode,
        tic_id=tic_id, run_index=row_index,
    )
    print("[INFO] MCMC finished")

    # --- 3. Save MCMC artefacts --------------------------------------------
    _save_pickle(idata,   run_dir / "idata.pkl")
    _save_pickle(windows, run_dir / "windows.pkl")

    # --- 4. Post-processing (inline; no subprocess re-importing JAX) -------
    from tesspipe.data_display import run_post_processing
    try:
        run_post_processing(run_dir, idata=idata, windows=windows,
                            wjs_csv=Path(CSV_PATH))
    except Exception as e:
        print(f"[WARN] post-processing failed: {e}")

    # --- 5. Latest symlink (only after everything else succeeded) -----------
    try:
        _update_latest_symlink(run_dir)
        print(f"[DONE] latest -> {run_dir.name}")
    except Exception as e:
        print(f"[WARN] could not update 'latest' symlink: {e}")


if __name__ == "__main__":
    main()
