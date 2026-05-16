"""Transit-window selection: ephemeris-first with TLS fallback.

Entry point: collect_windows().

Strategy
--------
1. If explicit transit times are provided (``time=``), build windows around
   those times directly — no TLS needed.
2. If a catalogue ephemeris (period + t0) is available, build windows from it.
   An optional narrow period scan (FIXED_T0_SCAN_PERIOD=1) refines the period.
3. If neither yields windows, run Transit Least Squares (TLS) on the combined
   light curve and use its best-fit ephemeris.

After window construction, each candidate is refined with a Gaussian dip-
template fit (_refine_tc_in_window) and filtered by duration coverage and
template-fit score.  A linear + quadratic O-C fit is attached to every window
for diagnostics.

Public API
----------
collect_windows(sectors, sector_data, ...) -> list[dict]
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from transitleastsquares import transitleastsquares

from tesspipe.env_utils import env_bool, env_int, env_float


# ---------------------------------------------------------------------------
# Window-building configuration (shared across helpers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _WindowBuildConfig:
    """Parameters that are constant for the duration of one collect_windows call.

    Bundled into a dataclass so the helper functions receive one typed object
    instead of five separate keyword arguments.
    """
    window: float
    max_windows_per_sector: int
    min_duration_coverage: float
    debug: bool
    pics_root: Path


# ---------------------------------------------------------------------------
# Cadence utilities
# ---------------------------------------------------------------------------

def _median_cadence_days(t: np.ndarray) -> float:
    """Median positive cadence, or NaN if fewer than 3 points."""
    if t.size < 3:
        return np.nan
    dt = np.diff(np.asarray(t, float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size > 0 else np.nan


# ---------------------------------------------------------------------------
# Transit-time refinement
# ---------------------------------------------------------------------------

def _refine_tc_in_window(
    t: np.ndarray,
    f: np.ndarray,
    tc_linear: float,
    duration_days: float,
    window_days: float,
) -> dict:
    """Refine the transit centre in one window via a Gaussian dip template.

    Scans a grid of centre positions and picks the one that minimises χ² of
    a ``1 - depth * Gaussian`` model.  Returns a dict with keys:
      accepted, tc_fit, score, reason, depth_fit.
    """
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]

    if t.size < 7:
        return {"accepted": False, "tc_fit": float(tc_linear), "score": 0.0,
                "reason": "too_few_points", "depth_fit": 0.0}

    cad = _median_cadence_days(t)
    if not np.isfinite(cad) or cad <= 0:
        cad = max(1e-4, float(window_days) / 200.0)

    dur = float(duration_days) if np.isfinite(duration_days) and duration_days > 0 \
        else max(2.0 * cad, float(window_days) / 8.0)

    span   = min(0.45 * float(window_days), max(2.0 * cad, 1.5 * dur))
    if span <= cad:
        span = 2.0 * cad
    ngrid  = int(np.clip(np.ceil(2.0 * span / cad), 31, 301))
    c_grid = np.linspace(float(tc_linear) - span, float(tc_linear) + span, ngrid)
    sigma  = max(1.5 * cad, dur / 5.0)

    chi2_flat = float(np.sum((f - 1.0) ** 2))
    if not np.isfinite(chi2_flat) or chi2_flat <= 0:
        return {"accepted": False, "tc_fit": float(tc_linear), "score": 0.0,
                "reason": "bad_chi2_flat", "depth_fit": 0.0}

    one_minus_f = 1.0 - f
    best = None
    for c in c_grid:
        g    = np.exp(-0.5 * ((t - c) / sigma) ** 2)
        gg   = float(np.sum(g * g))
        if gg <= 0:
            continue
        depth = max(0.0, float(np.sum(g * one_minus_f) / gg))
        if not np.isfinite(depth):
            continue
        chi2 = float(np.sum((f - (1.0 - depth * g)) ** 2))
        if best is None or chi2 < best["chi2"]:
            best = {"tc": float(c), "chi2": chi2, "depth": depth}

    if best is None:
        return {"accepted": False, "tc_fit": float(tc_linear), "score": 0.0,
                "reason": "template_fit_failed", "depth_fit": 0.0}

    score = float((chi2_flat - best["chi2"]) / chi2_flat)
    if not np.isfinite(score):
        score = 0.0

    if score < 0.005:
        return {"accepted": False, "tc_fit": float(tc_linear), "score": score,
                "reason": "low_tc_score", "depth_fit": float(best["depth"])}

    return {"accepted": True, "tc_fit": float(best["tc"]), "score": score,
            "reason": "accepted", "depth_fit": float(best["depth"])}


# ---------------------------------------------------------------------------
# Duration-coverage fraction
# ---------------------------------------------------------------------------

def _duration_coverage_fraction(t: np.ndarray, tc: float, duration_days: float) -> float:
    """Estimate fractional coverage of one transit duration centred on tc."""
    t   = np.asarray(t, float)
    t   = t[np.isfinite(t)]
    dur = float(duration_days)
    if t.size < 3 or not np.isfinite(dur) or dur <= 0:
        return 0.0
    half = 0.5 * dur
    tin  = t[(t >= float(tc) - half) & (t <= float(tc) + half)]
    if tin.size == 0:
        return 0.0
    cad = _median_cadence_days(t)
    if not np.isfinite(cad) or cad <= 0:
        cad = max(1e-4, dur / 100.0)
    return float(np.clip((np.max(tin) - np.min(tin) + cad) / dur, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Ephemeris fitting (for O-C diagnostics)
# ---------------------------------------------------------------------------

def _fit_linear_ephemeris(
    epochs: np.ndarray,
    tc_obs: np.ndarray,
    weights: np.ndarray = None,
) -> dict | None:
    """Weighted linear least-squares fit: tc = t0 + epoch * P."""
    epochs = np.asarray(epochs, float)
    tc_obs = np.asarray(tc_obs, float)
    m      = np.isfinite(epochs) & np.isfinite(tc_obs)
    epochs, tc_obs = epochs[m], tc_obs[m]
    if weights is None:
        weights = np.ones_like(tc_obs)
    else:
        weights = np.asarray(weights, float)[m]
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    if epochs.size < 2 or np.unique(epochs).size < 2:
        return None
    try:
        p, t0 = np.polyfit(epochs, tc_obs, deg=1, w=np.sqrt(weights))
    except Exception:
        return None
    return {"period_days": float(p), "t0_days": float(t0)} \
        if np.isfinite(p) and np.isfinite(t0) else None


def _fit_quadratic_ttv(
    epochs: np.ndarray,
    tc_obs: np.ndarray,
    weights: np.ndarray = None,
) -> dict | None:
    """Weighted quadratic fit for coarse TTV diagnostics."""
    epochs = np.asarray(epochs, float)
    tc_obs = np.asarray(tc_obs, float)
    m      = np.isfinite(epochs) & np.isfinite(tc_obs)
    epochs, tc_obs = epochs[m], tc_obs[m]
    if weights is None:
        weights = np.ones_like(tc_obs)
    else:
        weights = np.asarray(weights, float)[m]
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    if epochs.size < 3 or np.unique(epochs).size < 3:
        return None
    try:
        c2, c1, c0 = np.polyfit(epochs, tc_obs, deg=2, w=np.sqrt(weights))
    except Exception:
        return None
    if not all(np.isfinite(v) for v in (c0, c1, c2)):
        return None
    return {
        "c0": float(c0), "c1": float(c1), "c2": float(c2),
        "t0_days":      float(c0),
        "period_days":  float(c1 + c2),
        "dperiod_days": float(2.0 * c2),
    }


# ---------------------------------------------------------------------------
# Phase-distance helper
# ---------------------------------------------------------------------------

def _phase_distance_days(period_days: float, t0_ref: float, t0_new: float) -> float:
    """Absolute phase difference between two transit epochs, in days."""
    p = float(period_days)
    if not np.isfinite(p) or p <= 0:
        return np.inf
    d = float(t0_new) - float(t0_ref)
    return float(abs(d - np.round(d / p) * p))


# ---------------------------------------------------------------------------
# Window builders (raw cutouts, no quality filtering)
# ---------------------------------------------------------------------------

def _build_windows_from_tls(
    t, f, period, t0, duration_days, window_days, max_windows, f_err=None
) -> list:
    """Slice per-transit cutouts centred on t0 + n*period."""
    tmin = float(np.min(t))
    tmax = float(np.max(t))
    n_lo = int(np.floor((tmin - t0) / period)) - 1
    n_hi = int(np.ceil((tmax  - t0) / period)) + 1
    centers = sorted(
        tc for n in range(n_lo, n_hi + 1)
        if tmin <= (tc := float(t0 + n * period)) <= tmax
    )[:int(max_windows)]

    half_w = 0.5 * float(window_days)
    windows = []
    for tc in centers:
        m = (t >= tc - half_w) & (t <= tc + half_w) & np.isfinite(t) & np.isfinite(f)
        if np.sum(m) < 5:
            continue
        f_err_win = (
            np.full(int(np.sum(m)), np.nan, dtype=float) if f_err is None
            else np.asarray(np.asarray(f_err, float)[m], float)
        )
        windows.append({
            "t_center":           float(tc),
            "t_left":             tc - half_w,
            "t_right":            tc + half_w,
            "t":                  np.asarray(t[m], float),
            "f":                  np.asarray(f[m], float),
            "f_err":              f_err_win,
            "tls_period":         float(period),
            "tls_t0":             float(t0),
            "tls_duration_days":  float(duration_days),
        })
    return windows


def _build_windows_from_times(t, f, centers, duration_days, window_days, f_err=None) -> list:
    """Slice per-transit cutouts centred on explicit transit-time guesses."""
    centers = sorted(float(c) for c in np.ravel(np.asarray(centers, float)) if np.isfinite(c))
    half_w  = 0.5 * float(window_days)
    windows = []
    for tc in centers:
        m = (t >= tc - half_w) & (t <= tc + half_w) & np.isfinite(t) & np.isfinite(f)
        if np.sum(m) < 5:
            continue
        f_err_win = (
            np.full(int(np.sum(m)), np.nan, dtype=float) if f_err is None
            else np.asarray(np.asarray(f_err, float)[m], float)
        )
        windows.append({
            "t_center":          float(tc),
            "t_left":            tc - half_w,
            "t_right":           tc + half_w,
            "t":                 np.asarray(t[m], float),
            "f":                 np.asarray(f[m], float),
            "f_err":             f_err_win,
            "tls_period":        np.nan,
            "tls_t0":            np.nan,
            "tls_duration_days": float(duration_days),
        })
    return windows


# ---------------------------------------------------------------------------
# Quality filtering and refinement (top-level helpers)
# ---------------------------------------------------------------------------

def _apply_cov_score_filter(
    one_lc_windows: list,
    rec: dict,
    dur_days: float,
    source_tag: str,
    cfg: _WindowBuildConfig,
) -> list:
    """Refine each candidate window and keep those passing quality cuts.

    Mutates each passing window in-place to add refined tc, coverage, and
    sector/author/exptime metadata.
    """
    cov_thr   = float(np.clip(env_float("TLS_MIN_DURATION_COVERAGE", cfg.min_duration_coverage), 0.0, 1.0))
    score_thr = max(0.0, env_float("TLS_MIN_TC_SCORE", 0.02))
    # Reject windows where the Gaussian template moved tc_fit away from
    # tc_linear by more than this fraction of the transit duration — such a
    # large shift means the template found something *other than* the
    # predicted transit (e.g. a noise feature in a window with a data gap).
    shift_frac_thr = max(0.0, env_float("TLS_MAX_TC_SHIFT_FRAC", 0.5))
    kept      = []

    for w in one_lc_windows:
        tc_linear = float(w["t_center"])
        tc_ref    = _refine_tc_in_window(w["t"], w["f"], tc_linear, float(dur_days), float(cfg.window))
        tc_fit    = float(tc_ref["tc_fit"])
        half_w    = 0.5 * float(cfg.window)

        w.update({
            "t_center":          tc_fit,
            "t_left":            tc_fit - half_w,
            "t_right":           tc_fit + half_w,
            "tc_linear":         tc_linear,
            "tc_fit":            tc_fit,
            "tc_refine_score":   float(tc_ref["score"]),
            "tc_refine_reason":  str(tc_ref["reason"]),
            "tc_depth_fit":      float(tc_ref["depth_fit"]),
            "duration_coverage": _duration_coverage_fraction(w["t"], tc_fit, float(dur_days)),
            "sector":            int(rec["sector"]),
            "exptime":           float(rec["exptime"]),
            "author":            str(rec.get("author", "UNKNOWN")),
        })

        score_i    = float(w["tc_refine_score"])
        shift_days = abs(tc_fit - tc_linear)
        shift_max  = shift_frac_thr * float(dur_days)

        passed_cov   = w["duration_coverage"] >= cov_thr
        passed_score = np.isfinite(score_i) and score_i >= score_thr
        passed_shift = shift_days <= shift_max

        if passed_cov and passed_score and passed_shift:
            kept.append(w)
            if cfg.debug:
                print(
                    f"[cut_window] source={source_tag} sector={w['sector']} "
                    f"tc_linear={tc_linear:.6f} tc_fit={tc_fit:.6f} "
                    f"cov={w['duration_coverage']:.3f} score={score_i:.4f} "
                    f"shift={shift_days:.4f}d"
                )
        elif cfg.debug:
            reasons = []
            if not passed_cov:   reasons.append(f"cov={w['duration_coverage']:.3f}<{cov_thr:.2f}")
            if not passed_score: reasons.append(f"score={score_i:.4f}<{score_thr:.4f}")
            if not passed_shift: reasons.append(
                f"shift={shift_days:.4f}d>{shift_max:.4f}d (={shift_frac_thr:.2f}×duration)"
            )
            print(
                f"[reject_window] source={source_tag} sector={int(rec['sector'])} "
                f"tc_linear={tc_linear:.6f} tc_fit={tc_fit:.6f}: {', '.join(reasons)}"
            )
    return kept


def _build_and_refine(
    records: list,
    period: float,
    t0: float,
    dur_days: float,
    cfg: _WindowBuildConfig,
    sde_val: float = np.nan,
    depth_val: float = np.nan,
    source_tag: str = "ephem_or_tls",
) -> list:
    """Build windows from a linear ephemeris, refine, and filter all records."""
    local = []
    for rec in records:
        raw = _build_windows_from_tls(
            rec["t"], rec["f"], float(period), float(t0),
            float(dur_days), float(cfg.window), int(cfg.max_windows_per_sector),
            f_err=rec.get("f_err"),
        )
        for w in raw:
            w["depth_est_trigger"] = float(depth_val) if np.isfinite(depth_val) else np.nan
            w["snr_est_trigger"]   = float(sde_val)   if np.isfinite(sde_val)   else np.nan
            w["tls_sde"]           = float(sde_val)   if np.isfinite(sde_val)   else np.nan
            w["tls_depth"]         = float(depth_val) if np.isfinite(depth_val) else np.nan

        kept = _apply_cov_score_filter(raw, rec, dur_days, source_tag, cfg)
        local.extend(kept)
        if cfg.debug:
            _save_sector_plot(rec["t"], rec["f"], kept, rec["sector"],
                              rec["exptime"], cfg.pics_root, "tls", rec.get("author", "UNKNOWN"))
    return local


def _build_and_refine_from_times(
    records: list,
    time_centers,
    dur_days: float,
    cfg: _WindowBuildConfig,
) -> list:
    """Build windows from explicit transit times, refine, and filter all records."""
    local = []
    for rec in records:
        raw = _build_windows_from_times(
            rec["t"], rec["f"], time_centers,
            float(dur_days), float(cfg.window), f_err=rec.get("f_err"),
        )
        for w in raw:
            w["depth_est_trigger"] = w["snr_est_trigger"] = np.nan
            w["tls_sde"] = w["tls_depth"] = np.nan

        kept = _apply_cov_score_filter(raw, rec, dur_days, "manual_time", cfg)
        local.extend(kept)
        if cfg.debug:
            _save_sector_plot(rec["t"], rec["f"], kept, rec["sector"],
                              rec["exptime"], cfg.pics_root, "manual_time", rec.get("author", "UNKNOWN"))
    return local


# ---------------------------------------------------------------------------
# Period scoring (used only with FIXED_T0_SCAN_PERIOD=1)
# ---------------------------------------------------------------------------

def _score_period_fixed_t0(records, period, t0, dur_days, window, max_windows) -> tuple:
    """Sum refined template scores for a trial period with fixed t0."""
    score_sum = 0.0
    n_w       = 0
    for rec in records:
        for w in _build_windows_from_tls(rec["t"], rec["f"], float(period), float(t0),
                                          float(dur_days), float(window), int(max_windows)):
            result = _refine_tc_in_window(w["t"], w["f"], float(w["t_center"]),
                                          float(dur_days), float(window))
            score_sum += max(0.0, float(result.get("score", 0.0)))
            n_w       += 1
    return float(score_sum), int(n_w)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup_windows_by_epoch(windows: list, t0: float, period: float) -> list:
    """Keep only the highest-quality window for each integer transit epoch."""
    if not windows:
        return windows
    p = float(period)
    if not np.isfinite(p) or p <= 0:
        return windows

    best: dict = {}
    for w in windows:
        tc = float(w.get("tc_fit", w.get("t_center", np.nan)))
        if not np.isfinite(tc):
            continue
        e    = int(np.rint((tc - float(t0)) / p))
        rank = (float(w.get("tc_refine_score", -np.inf)),
                float(w.get("duration_coverage", -np.inf)),
                -float(w.get("exptime", np.inf)))
        cur = best.get(e)
        if cur is None or rank > cur[0]:
            best[e] = (rank, w)

    return [v[1] for _, v in sorted(best.items(), key=lambda kv: kv[0])]


# ---------------------------------------------------------------------------
# Debug plots
# ---------------------------------------------------------------------------

def _save_sector_plot(t, f, windows, sector, exptime, outdir, mode_tag, author):
    """Save a per-sector debug plot with selected windows highlighted."""
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=160)
    ax.plot(t, f, ".", ms=2, alpha=0.65, color="black")
    y0, y1 = float(np.nanmin(f)), float(np.nanmax(f))
    y_text = y0 + 0.08 * (y1 - y0 if y1 > y0 else 1.0)
    for i, w in enumerate(windows, start=1):
        c_fit  = float(w.get("tc_fit",    w["t_center"]))
        c_lin  = float(w.get("tc_linear", w["t_center"]))
        score  = float(w.get("tc_refine_score", np.nan))
        cov    = float(w.get("duration_coverage", np.nan))
        reason = str(w.get("tc_refine_reason", "na"))
        ax.axvspan(float(w["t_left"]), float(w["t_right"]), color="tab:orange", alpha=0.18)
        ax.axvline(c_lin, color="tab:blue", lw=1.0, alpha=0.7)
        ax.axvline(c_fit, color="tab:red",  lw=1.1, alpha=0.9)
        label = f"w{i} s={score:.3f}" if np.isfinite(score) else f"w{i}"
        if np.isfinite(cov):
            label += f" cov={cov:.2f}"
        if reason != "accepted":
            label += f" {reason}"
        ax.text(c_fit, y_text, label, fontsize=7, ha="center", va="bottom",
                color="tab:red", rotation=90)
    ax.set_title(f"{mode_tag} | sector={int(sector)} exptime={float(exptime):.0f}s author={author}")
    ax.set_xlabel("Time [BTJD]")
    ax.set_ylabel("Normalised flux")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{mode_tag}_sector{int(sector)}_expt{int(round(float(exptime)))}.png")
    plt.close(fig)


def _save_oc_plot(windows: list, outdir: Path, tic_id, mode_tag: str = "tls_ttv"):
    """Save an O-C diagnostic scatter from the selected windows."""
    rows = [
        (float(w["ttv_epoch"]), float(w.get("oc_refit_linear_days", w.get("oc_linear_days", np.nan))))
        for w in windows
        if "ttv_epoch" in w
    ]
    rows = [(e, oc) for e, oc in rows if np.isfinite(e) and np.isfinite(oc)]
    if not rows:
        return
    arr = np.asarray(rows, float)
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    ax.axhline(0.0, color="0.5", lw=1.0, alpha=0.7)
    ax.plot(arr[:, 0], arr[:, 1] * 1440.0, "o", ms=4, alpha=0.85, color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("O-C (linear) [min]")
    ax.set_title(f"{mode_tag} O-C | TIC {int(tic_id) if tic_id is not None else 0}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{mode_tag}_oc_tic{int(tic_id) if tic_id is not None else 0}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect_windows(
    sectors,
    sector_data: dict,
    duration_hours=None,
    period_days_prior=None,
    t0_days_prior=None,
    time=None,
    min_duration_coverage: float = 0.6,
    window: float = 0.5,
    max_windows_per_sector: int = 10,
    debug: bool = False,
    tic_id=None,
    run_index=None,
) -> list:
    """Collect transit windows using ephemeris-first logic with TLS fallback.

    Returns a list of window dicts, each containing the time/flux arrays,
    window bounds, refined transit centre, and O-C diagnostic fields.
    """
    idx_for_path = (
        int(run_index) if run_index is not None
        else (int(tic_id) if tic_id is not None else 0)
    )
    pics_root = Path("results") / "tls" / "pics" / str(idx_for_path)

    cfg = _WindowBuildConfig(
        window=float(window),
        max_windows_per_sector=int(max_windows_per_sector),
        min_duration_coverage=float(min_duration_coverage),
        debug=bool(debug),
        pics_root=pics_root,
    )

    # ------------------------------------------------------------------
    # Build normalised per-sector records for TLS / window construction.
    # ------------------------------------------------------------------
    tls_downsample_step = env_int("TLS_DOWNSAMPLE_STEP", default=4, min_value=1)
    records = []
    for sector in sectors:
        for (sec, exptime), lc in sector_data.items():
            if int(sec) != int(sector):
                continue
            t = np.asarray(lc.time.value, float)
            f = np.asarray(lc.flux.value, float)
            try:
                ferr = np.asarray(lc.flux_err.value, float)
            except Exception:
                ferr = np.full_like(f, np.nan)
            if ferr.shape != f.shape:
                ferr = np.full_like(f, np.nan)

            m = np.isfinite(t) & np.isfinite(f)
            t, f, ferr = t[m], f[m], ferr[m]
            if t.size < 50:
                continue
            med = np.nanmedian(f)
            if not np.isfinite(med) or med == 0:
                continue
            fn    = f / med
            ferrn = np.abs(ferr / med)
            records.append({
                "sector":  int(sector),
                "exptime": float(exptime),
                "author":  str(lc.meta.get("AUTHOR", lc.meta.get("author",
                           lc.meta.get("ORIGIN", "UNKNOWN")))),
                "t":       t,
                "f":       fn,
                "f_err":   ferrn,
                "t_tls":   t[::tls_downsample_step],
                "f_tls":   fn[::tls_downsample_step],
            })

    if not records:
        if debug:
            print("[TLS] no valid sector_data")
        return []

    # ------------------------------------------------------------------
    # Path 1: explicit manual transit times
    # ------------------------------------------------------------------
    period            = np.nan
    t0                = np.nan
    dur_days          = np.nan
    ephem_windows: list  = []
    manual_time_mode  = False
    windows: list        = []

    if time is not None:
        t_manual = np.asarray(time, float).ravel()
        t_manual = t_manual[np.isfinite(t_manual)]
        t_manual = np.where(t_manual > 2.4e6, t_manual - 2457000.0, t_manual)

        if t_manual.size > 0:
            manual_time_mode = True
            if duration_hours is not None and np.isfinite(duration_hours) and float(duration_hours) > 0:
                dur_days = float(duration_hours) / 24.0
            elif period_days_prior is not None and np.isfinite(period_days_prior) and float(period_days_prior) > 0:
                dur_days = max(0.06, min(0.6, 0.015 * float(period_days_prior)))
            else:
                dur_days = max(0.04, min(0.6, 0.30 * float(window)))
            if debug:
                print(f"[MANUAL time] {t_manual.size} centres, duration_days={dur_days:.6f}")
            manual_windows = _build_and_refine_from_times(records, t_manual, dur_days, cfg)
            windows.extend(manual_windows)
            if debug:
                print(f"[MANUAL time] accepted windows: {len(manual_windows)}")

    # ------------------------------------------------------------------
    # Path 2: catalogue ephemeris (period + t0 prior)
    # ------------------------------------------------------------------
    use_ephem_prior = env_bool("TLS_USE_EPHEM_PRIOR", default=True)

    if (
        not manual_time_mode
        and use_ephem_prior
        and period_days_prior is not None
        and t0_days_prior is not None
        and np.isfinite(period_days_prior)
        and np.isfinite(t0_days_prior)
        and float(period_days_prior) > 0
    ):
        period = float(period_days_prior)
        t0     = float(t0_days_prior)
        if t0 > 2.4e6:
            t0 -= 2457000.0

        if duration_hours is not None and np.isfinite(duration_hours) and float(duration_hours) > 0:
            dur_days = float(duration_hours) / 24.0
        else:
            dur_days = max(0.06, min(0.6, 0.015 * period))

        if debug:
            print(f"[EPHEM prior] P={period:.6f} t0={t0:.6f} dur={dur_days:.6f}")

        if env_bool("FIXED_T0_SCAN_PERIOD", default=False):
            frac_lo = env_float("TLS_PRIOR_FRAC_LO", 0.9) if os.getenv("TLS_PRIOR_FRAC_LO") else 0.9
            frac_hi = env_float("TLS_PRIOR_FRAC_HI", 1.1) if os.getenv("TLS_PRIOR_FRAC_HI") else 1.1
            p_lo    = max(0.0, period * frac_lo)
            p_hi    = max(p_lo + 0.1, period * frac_hi)
            n_steps = env_int("FIXED_T0_PERIOD_SCAN_STEPS", default=121, min_value=11)
            best_item = None
            for p_try in np.linspace(p_lo, p_hi, n_steps):
                s, n = _score_period_fixed_t0(records, p_try, t0, dur_days, window, max_windows_per_sector)
                item = (s, n, float(p_try))
                if best_item is None or item[0] > best_item[0] or (item[0] == best_item[0] and item[1] > best_item[1]):
                    best_item = item
            if best_item is not None:
                period = float(best_item[2])
                if debug:
                    print(f"[P-scan fixed t0] period_best={period:.8f} score={best_item[0]:.4f}")
        elif debug:
            print(f"[P-fixed] using prior period directly: P={period:.8f}")

        ephem_windows = _build_and_refine(records, period, t0, dur_days, cfg, source_tag="ephem_prior")
        if ephem_windows:
            windows.extend(ephem_windows)
        elif debug:
            print("[EPHEM prior] no windows found; will attempt TLS fallback.")

    # ------------------------------------------------------------------
    # Path 3: TLS global search
    # ------------------------------------------------------------------
    t_all = np.concatenate([r["t_tls"] for r in records])
    f_all = np.concatenate([r["f_tls"] for r in records])
    order = np.argsort(t_all)
    t_all, f_all = t_all[order], f_all[order]

    tls_max_pts = (
        None if not os.getenv("TLS_MAX_POINTS")
        else env_int("TLS_MAX_POINTS", default=1000, min_value=1000)
    )
    if tls_max_pts is not None and t_all.size > tls_max_pts:
        step  = int(np.ceil(t_all.size / tls_max_pts))
        t_all = t_all[::step]
        f_all = f_all[::step]
        if debug:
            print(f"[TLS cap] downsampled to {t_all.size} points (step={step})")

    tspan = float(np.max(t_all) - np.min(t_all))
    if period_days_prior is not None and np.isfinite(period_days_prior) and float(period_days_prior) > 0:
        p0       = float(period_days_prior)
        frac_lo  = env_float("TLS_PRIOR_FRAC_LO", 0.7) if os.getenv("TLS_PRIOR_FRAC_LO") else 0.7
        frac_hi  = env_float("TLS_PRIOR_FRAC_HI", 1.3) if os.getenv("TLS_PRIOR_FRAC_HI") else 1.3
        period_min = max(0.6, p0 * frac_lo)
        period_max = max(period_min + 0.1, p0 * frac_hi)
    else:
        period_min = max(0.6, min(10.0, 0.2 * tspan))
        period_max = max(period_min + 0.1, min(20.0, 0.8 * tspan))
    if os.getenv("TLS_PERIOD_MIN"):
        period_min = env_float("TLS_PERIOD_MIN", period_min)
    if os.getenv("TLS_PERIOD_MAX"):
        period_max = env_float("TLS_PERIOD_MAX", period_max)
    if period_max <= period_min:
        period_max = period_min + 0.1
    if debug:
        print(f"[TLS range] period_min={period_min:.4f} period_max={period_max:.4f}")

    run_tls = not ephem_windows and not env_bool("TLS_EPHEM_ONLY", default=False) and not manual_time_mode

    if run_tls:
        tls_threads   = env_int("TLS_THREADS", default=1, min_value=1)
        tls_oversamp  = env_int("TLS_OVERSAMPLING_FACTOR", default=10, min_value=1)
        tls_show_prog = env_bool("TLS_SHOW_PROGRESS", default=False)

        model       = transitleastsquares(t_all, f_all)
        trial_ranges = [
            (float(period_min), float(period_max), int(tls_oversamp)),
            (float(period_min), float(min(period_max, max(period_min + 0.1, 0.6 * tspan))),
             max(2, int(tls_oversamp) // 2)),
            (float(max(0.6, min(period_min, 10.0))), float(max(10.1, min(200.0, 0.8 * tspan))),
             max(2, int(tls_oversamp) // 3)),
        ]
        res      = None
        last_exc = None
        for i, (pmin_i, pmax_i, over_i) in enumerate(trial_ranges, start=1):
            if pmax_i <= pmin_i:
                pmax_i = pmin_i + 0.1
            try:
                if debug:
                    print(f"[TLS try {i}] pmin={pmin_i:.4f} pmax={pmax_i:.4f} oversampling={over_i}")
                res = model.power(
                    show_progress_bar=bool(tls_show_prog),
                    use_threads=int(tls_threads),
                    period_min=float(pmin_i),
                    period_max=float(pmax_i),
                    n_transits_min=1,
                    oversampling_factor=int(over_i),
                )
                break
            except ValueError as e:
                last_exc = e
                if "zero-size array" in str(e).lower() and "minimum" in str(e).lower():
                    if debug:
                        print(f"[TLS try {i}] empty grid: {e}; retrying")
                    continue
                raise

        if res is None:
            raise RuntimeError(f"TLS failed after fallback retries; last error: {last_exc}")

        period   = float(res.period)
        t0       = float(res.T0)
        dur_days = float(res.duration)
        if duration_hours is not None and np.isfinite(duration_hours) and float(duration_hours) > 0:
            dur_days = float(duration_hours) / 24.0
        if debug:
            print(f"[TLS global] P={period:.6f} t0={t0:.6f} dur={dur_days:.6f} SDE={float(res.SDE):.3f}")

        tls_windows = _build_and_refine(records, period, t0, dur_days, cfg,
                                         sde_val=float(res.SDE), depth_val=float(res.depth),
                                         source_tag="tls_global")

        # Gate: reject TLS result when it disagrees with an existing ephem prior.
        use_tls = True
        if ephem_windows and np.isfinite(period_days_prior) and np.isfinite(t0_days_prior):
            p_prior   = float(period_days_prior)
            t0_prior  = float(t0_days_prior)
            if t0_prior > 2.4e6:
                t0_prior -= 2457000.0
            phase_dist = _phase_distance_days(p_prior, t0_prior, t0)
            p_rel_err  = abs(period - p_prior) / max(1e-9, p_prior)
            phase_tol  = max(2.0 * float(dur_days), 0.25 * float(window))
            p_rel_tol  = 0.03
            if os.getenv("TLS_PRIOR_PHASE_TOL_DAYS"):
                phase_tol = max(0.0, env_float("TLS_PRIOR_PHASE_TOL_DAYS", phase_tol))
            if os.getenv("TLS_PRIOR_REL_PERIOD_TOL"):
                p_rel_tol = max(0.0, env_float("TLS_PRIOR_REL_PERIOD_TOL", p_rel_tol))
            if phase_dist > phase_tol or p_rel_err > p_rel_tol:
                use_tls = False
                if debug:
                    print(
                        f"[TLS gate] reject TLS: phase_dist={phase_dist:.4f}d (tol={phase_tol:.4f}d), "
                        f"p_rel_err={p_rel_err:.4f} (tol={p_rel_tol:.4f}); keeping EPHEM windows."
                    )

        if use_tls and tls_windows:
            windows.extend(tls_windows)
        elif ephem_windows:
            windows.extend(ephem_windows)
        elif tls_windows:
            windows.extend(tls_windows)

    elif ephem_windows:
        if debug:
            print("[EPHEM prior] TLS_EPHEM_ONLY=1; skipping TLS search.")
        windows.extend(ephem_windows)

    if not windows:
        if debug:
            _save_oc_plot(windows, pics_root, tic_id)
        return windows

    # ------------------------------------------------------------------
    # Post-processing: dedup, epoch assignment, O-C fits.
    # ------------------------------------------------------------------
    has_linear_ephem = np.isfinite(period) and np.isfinite(t0) and float(period) > 0

    if env_bool("TLS_DEDUP_BY_EPOCH", default=True) and has_linear_ephem:
        n_before = len(windows)
        windows  = _dedup_windows_by_epoch(windows, t0=t0, period=period)
        if debug and len(windows) != n_before:
            print(f"[TLS dedup] {n_before} -> {len(windows)} windows")

    if has_linear_ephem:
        epochs, tc_obs, weights = [], [], []
        for w in windows:
            tc     = float(w.get("tc_fit", w["t_center"]))
            e      = int(np.round((tc - t0) / period))
            tc_lin = float(t0 + e * period)
            w["ttv_epoch"]      = int(e)
            w["oc_linear_days"] = float(tc - tc_lin)
            epochs.append(float(e))
            tc_obs.append(float(tc))
            weights.append(max(1e-4, float(w.get("tc_refine_score", 0.01))))

        lin_fit = _fit_linear_ephemeris(epochs, tc_obs, weights)
        if lin_fit is not None:
            if debug:
                print(f"[TLS refit-linear] P={lin_fit['period_days']:.8f} t0={lin_fit['t0_days']:.8f}")
            for w in windows:
                e        = float(w["ttv_epoch"])
                tc_obs_i = float(w.get("tc_fit", w["t_center"]))
                tc_refit = float(lin_fit["t0_days"] + e * lin_fit["period_days"])
                w["tls_refit_period_days"] = float(lin_fit["period_days"])
                w["tls_refit_t0_days"]     = float(lin_fit["t0_days"])
                w["oc_refit_linear_days"]  = float(tc_obs_i - tc_refit)
        else:
            for w in windows:
                w["tls_refit_period_days"] = np.nan
                w["tls_refit_t0_days"]     = np.nan
                w["oc_refit_linear_days"]  = np.nan

        ttv_fit = _fit_quadratic_ttv(epochs, tc_obs, weights)
        if ttv_fit is not None:
            if debug:
                print(f"[TLS+TTV] P0={ttv_fit['period_days']:.8f} dP={ttv_fit['dperiod_days']:.3e} d/epoch")
            for w in windows:
                e        = float(w["ttv_epoch"])
                tc_obs_i = float(w.get("tc_fit", w["t_center"]))
                tc_model = float(ttv_fit["c2"] * e**2 + ttv_fit["c1"] * e + ttv_fit["c0"])
                w["tc_ttv_model"]         = tc_model
                w["oc_ttv_resid_days"]    = float(tc_obs_i - tc_model)
                w["tls_ttv_period0_days"] = float(ttv_fit["period_days"])
                w["tls_ttv_dperiod_days"] = float(ttv_fit["dperiod_days"])
                w["tls_ttv_t0_days"]      = float(ttv_fit["t0_days"])
        else:
            for w in windows:
                w["tc_ttv_model"]         = float(w.get("tc_fit", w["t_center"]))
                w["oc_ttv_resid_days"]    = np.nan
                w["tls_ttv_period0_days"] = np.nan
                w["tls_ttv_dperiod_days"] = np.nan
                w["tls_ttv_t0_days"]      = np.nan
    else:
        for w in windows:
            w.setdefault("ttv_epoch",           np.nan)
            w.setdefault("oc_linear_days",       np.nan)
            w.setdefault("tls_refit_period_days", np.nan)
            w.setdefault("tls_refit_t0_days",     np.nan)
            w.setdefault("oc_refit_linear_days",  np.nan)
            w["tc_ttv_model"]         = float(w.get("tc_fit", w["t_center"]))
            w.setdefault("oc_ttv_resid_days",    np.nan)
            w.setdefault("tls_ttv_period0_days", np.nan)
            w.setdefault("tls_ttv_dperiod_days", np.nan)
            w.setdefault("tls_ttv_t0_days",      np.nan)

    if debug:
        _save_oc_plot(windows, pics_root, tic_id)

    return windows
