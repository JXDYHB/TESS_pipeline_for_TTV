"""Post-processing and visualisation for completed MCMC runs.

Can be run as a module::

    python -m tesspipe.data_display --index 5 --results-dir OUTPUT/results-...

or called inline from ppl.py via save_fit_plots().
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

try:
    from tinygp import GaussianProcess
    from tinygp.kernels import quasisep
except Exception:
    GaussianProcess = None
    quasisep        = None

from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit

# ---------------------------------------------------------------------------
# Physical / timing constant
# ---------------------------------------------------------------------------
MINUTES_PER_DAY = 1440.0   # used throughout to convert days ↔ minutes


# ---------------------------------------------------------------------------
# File loading helper
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    """Read a pickled artefact from ``path``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_period_prior_from_wjs(tic_id, csv_path) -> float | None:
    """Read the period prior P for a TIC from the WJs catalogue CSV."""
    if tic_id is None:
        return None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(float(row.get("TIC", -1))) == int(tic_id):
                    p = row.get("P", "")
                    return float(p) if p else None
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Posterior extraction helpers
# ---------------------------------------------------------------------------

def posterior_scalar(post, name: str, k: int = None, use: str = "median") -> float:
    """Extract a scalar summary for one posterior variable."""
    key = f"{name}_{k}" if k is not None else name
    if key not in post.data_vars:
        raise KeyError(f"posterior missing variable '{key}'")
    arr = post[key]
    if use == "mean":
        return float(arr.mean().values)
    return float(np.median(arr.stack(sample=("chain", "draw")).values))


def posterior_indexed_scalar(
    post, name: str, index, use: str = "median", default: float = 0.0
) -> float:
    """Extract a scalar from a vector-valued posterior variable."""
    if index is None or int(index) < 0 or name not in post.data_vars:
        return float(default)
    vals = np.asarray(post[name].stack(sample=("chain", "draw")).values, float)
    idx  = int(index)
    if vals.ndim == 1:
        out = vals[idx] if idx < vals.shape[0] else float(default)
    else:
        out = vals[..., idx] if idx < vals.shape[-1] else float(default)
    return float(np.mean(out) if use == "mean" else np.median(out))


# ---------------------------------------------------------------------------
# Transit model helpers (used for visualisation)
# ---------------------------------------------------------------------------

def kipping_q_to_u(q1, q2):
    """Convert Kipping (q1, q2) to quadratic LD (u1, u2)."""
    s = jnp.sqrt(q1)
    return 2 * s * q2, s * (1 - 2 * q2)


def transit_delta(t_rel, p: float, duration, r, b, q1, q2, t0) -> np.ndarray:
    """Compute transit-only flux perturbation on a relative-time grid."""
    u1, u2 = kipping_q_to_u(q1, q2)
    orbit  = TransitOrbit(
        period=float(p), duration=duration, time_transit=t0,
        impact_param=b, radius_ratio=r,
    )
    return np.asarray(limb_dark_light_curve(orbit, jnp.array([u1, u2]))(jnp.asarray(t_rel)))


def contact_times_rel(t0_rel: float, duration_days: float, b: float, r: float) -> dict | None:
    """Approximate t1..t4 contact times relative to the window reference."""
    t14 = float(duration_days)
    if not np.isfinite(t14) or t14 <= 0:
        return None
    b, r     = float(b), abs(float(r))
    chord_14 = max(0.0, (1.0 + r) ** 2 - b ** 2)
    chord_23 = max(0.0, (1.0 - r) ** 2 - b ** 2)
    if chord_14 <= 0:
        return None
    t23 = t14 * np.sqrt(chord_23 / chord_14) if chord_23 > 0 else 0.0
    return {
        "t1": float(t0_rel) - 0.5 * t14,
        "t2": float(t0_rel) - 0.5 * t23,
        "t3": float(t0_rel) + 0.5 * t23,
        "t4": float(t0_rel) + 0.5 * t14,
    }


# ---------------------------------------------------------------------------
# Transit-time extraction
# ---------------------------------------------------------------------------

def get_abs_t0_and_rows(
    idata, windows, use: str = "median", ref_key: str = "tc_fit"
) -> list:
    """Build sorted transit-time rows from posterior t0 samples."""
    post = idata.posterior
    rows = []
    for k, w in enumerate(windows):
        t_ref = float(w[ref_key]) if ref_key in w else float(w["t_center"])
        name  = f"t0_{k}"
        if name not in post.data_vars:
            raise KeyError(f"posterior missing '{name}'")
        arr    = post[name].stack(sample=("chain", "draw")).values
        t0_rel = float(np.median(arr) if use == "median" else np.mean(arr))
        rows.append({
            "k":        k,
            "sector":   int(w.get("sector", -1)),
            "t0_abs":   t_ref + t0_rel,
            "epoch_raw": w.get("ttv_epoch", np.nan),
        })
    rows.sort(key=lambda x: x["t0_abs"])
    return rows


def get_sigma_t_minutes_from_idata(idata, sorted_rows) -> np.ndarray:
    """Per-transit timing uncertainty in minutes from posterior samples."""
    post = idata.posterior
    out  = np.zeros(len(sorted_rows), dtype=float)
    for i, r in enumerate(sorted_rows):
        name = f"t0_{int(r['k'])}"
        if name not in post.data_vars:
            raise KeyError(f"posterior missing '{name}'")
        samples = post[name].stack(sample=("chain", "draw")).values.astype(float)
        out[i]  = np.std(samples, ddof=1) * MINUTES_PER_DAY
    return out


# ---------------------------------------------------------------------------
# Period search (used in post-processing)
# ---------------------------------------------------------------------------

def all_pairwise_dts(t: np.ndarray) -> np.ndarray:
    """All positive pairwise time differences."""
    t   = np.sort(np.asarray(t, float))
    out = [t[j] - t[i] for i in range(len(t)) for j in range(i + 1, len(t))]
    return np.asarray(out, float)


def generate_p_candidates_pairwise(
    t, p_min: float, p_max: float,
    p_prior=None, k_max: int = 800, top: int = 12000, k_window: int = 2,
) -> np.ndarray:
    """Candidate periods from pairwise time differences within a period range."""
    dts = all_pairwise_dts(t)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    if dts.size == 0:
        return np.array([])

    cands = []
    for dt in dts:
        if p_prior is not None and np.isfinite(p_prior) and p_prior > 0:
            k0 = int(np.rint(dt / float(p_prior)))
            ks = np.arange(max(1, k0 - k_window), min(k_max, max(1, k0 + k_window)) + 1)
        else:
            k_hi = int(min(k_max, np.floor(dt / p_min)))
            if k_hi < 1:
                continue
            ks = np.arange(1, k_hi + 1)
        ps = dt / ks
        m  = (ps >= p_min) & (ps <= p_max)
        if np.any(m):
            cands.append(ps[m])

    if not cands:
        return np.array([])
    p = np.sort(np.unique(np.concatenate(cands)))
    if p.size > top:
        p = p[np.unique(np.linspace(0, p.size - 1, top).astype(int))]
    return p


def score_p_by_intervals(
    t, sigma_days, p: float,
    penalty_strength: float = 0.2,
    p_prior=None, prior_rel_sigma: float = 0.05, prior_strength: float = 25.0,
) -> dict | None:
    """Score a trial period using adjacent-interval residuals."""
    t, s   = np.asarray(t, float), np.asarray(sigma_days, float)
    order  = np.argsort(t)
    t, s   = t[order], s[order]
    dt     = np.diff(t)
    sig_dt = np.sqrt(s[:-1] ** 2 + s[1:] ** 2)
    m      = np.isfinite(dt) & np.isfinite(sig_dt) & (dt > 0) & (sig_dt > 0)
    dt, sig_dt = dt[m], sig_dt[m]
    if dt.size < 2:
        return None

    k_raw = np.rint(dt / p).astype(int)
    use   = k_raw >= 1
    if not np.any(use):
        return None
    k, dt_use, sig_use = k_raw[use], dt[use], sig_dt[use]
    resid      = dt_use - k * p
    chi2       = float(np.sum((resid / sig_use) ** 2))
    score      = chi2 + penalty_strength * np.log1p(np.sum(k))
    prior_term = 0.0
    if p_prior is not None and np.isfinite(p_prior) and p_prior > 0:
        rel        = (float(p) - float(p_prior)) / max(1e-12, float(p_prior))
        prior_term = float(prior_strength) * (rel / max(1e-4, float(prior_rel_sigma))) ** 2
        score     += prior_term

    return {"P": float(p), "chi2": chi2, "score": float(score), "prior_term": prior_term}


def find_best_period_prior_guided(
    t, sigma_days, p_prior: float,
    frac_lo: float = 0.8, frac_hi: float = 1.2,
    k_max: int = 800, keep_top: int = 50,
    refine_steps: int = 3, refine_span: float = 0.02,
    prior_rel_sigma: float = 0.05, prior_strength: float = 25.0,
) -> tuple:
    """Find the best period near a prior using candidate scoring + local refinement."""
    if p_prior is None or not np.isfinite(p_prior) or p_prior <= 0:
        raise ValueError("Invalid period prior from WJs.csv")

    p_min   = float(p_prior) * float(frac_lo)
    p_max   = float(p_prior) * float(frac_hi)
    p_cands = generate_p_candidates_pairwise(t, p_min, p_max, p_prior=p_prior,
                                             k_max=k_max, k_window=2)
    p_dense = np.linspace(p_min, p_max, 800)
    p_cands = np.unique(np.concatenate([p_cands, p_dense])) if p_cands.size > 0 else p_dense

    scored = [
        r for p in p_cands
        if (r := score_p_by_intervals(t, sigma_days, p, p_prior=p_prior,
                                       prior_rel_sigma=prior_rel_sigma,
                                       prior_strength=prior_strength)) is not None
    ]
    if not scored:
        raise RuntimeError("All candidate periods failed scoring")

    scored.sort(key=lambda d: d["score"])
    top_list = scored[:min(keep_top, len(scored))]
    best     = top_list[0]

    for _ in range(refine_steps):
        improved = False
        for cand in top_list:
            p0 = cand["P"]
            for p in np.linspace(p0 * (1 - refine_span), p0 * (1 + refine_span), 161):
                if not (p_min <= p <= p_max):
                    continue
                out = score_p_by_intervals(t, sigma_days, p, p_prior=p_prior,
                                           prior_rel_sigma=prior_rel_sigma,
                                           prior_strength=prior_strength)
                if out is not None and out["score"] < best["score"]:
                    best, improved = out, True
        if not improved:
            break

    return best["P"], best, (p_min, p_max)


def keep_mask_from_intervals(
    t, sigma_min, p: float,
    k_interval: float = 6.0, g_mode: str = "sqrt", g_max: float = 6.0,
) -> tuple:
    """Build a keep-mask from interval residual consistency under a trial period."""
    t, sig = np.asarray(t, float), np.asarray(sigma_min, float)
    order  = np.argsort(t)
    t_s, sig_s = t[order], sig[order]

    dt    = np.diff(t_s)
    k     = np.clip(np.rint(dt / p).astype(int), 0, None)
    resid = (dt - k * p) * MINUTES_PER_DAY          # days → minutes
    base  = np.sqrt(sig_s[:-1] ** 2 + sig_s[1:] ** 2)
    g     = np.minimum(np.sqrt(k.astype(float)), float(g_max)) if g_mode == "sqrt" else 1.0
    thr   = float(k_interval) * base * g
    good  = np.isfinite(resid) & np.isfinite(thr) & (np.abs(resid) <= thr)
    good  = np.where(k == 0, True, good)

    keep_sorted = np.ones(len(t_s), dtype=bool)
    if len(t_s) >= 2:
        keep_sorted[0]  = good[0]
        keep_sorted[-1] = good[-1]
        if len(t_s) > 2:
            keep_sorted[1:-1] = good[:-1] & good[1:]

    keep = np.zeros(len(t), dtype=bool)
    keep[order] = keep_sorted
    dbg = {"order": order, "k": k, "resid_min": resid, "thr": thr, "good_interval": good}
    return keep, dbg


def fit_linear_ephemeris_mle(
    t_obs,
    sigma_t,
    p_init: float,
    max_iter: int = 10,
) -> dict | None:
    """Iteratively fit the linear ephemeris ``t = T0 + n·P`` by maximum likelihood.

    Each iteration assigns integer epochs ``n_i = round((t_i - T0) / P)`` and
    refits ``(P, T0)`` by inverse-variance-weighted least squares — the MLE
    for Gaussian transit-time errors.  Convergence is reached when the
    epoch numbering stops changing.

    Parameters
    ----------
    t_obs : array
        Mid-transit times (e.g. days, BTJD).
    sigma_t : array
        1-σ uncertainties on each ``t_obs``, same units as ``t_obs``.
    p_init : float
        Initial period guess (e.g. from a prior-guided grid search).
    max_iter : int
        Max number of epoch-reassignment iterations.

    Returns
    -------
    dict with keys:
        P, P_err, T0, T0_err, epochs, chi2, dof, redchi2
    or ``None`` when the fit cannot be performed (too few finite points or
    a degenerate epoch matrix).
    """
    t = np.asarray(t_obs, float)
    s = np.asarray(sigma_t, float)
    finite = np.isfinite(t) & np.isfinite(s) & (s > 0)
    t, s = t[finite], s[finite]
    if t.size < 2:
        return None

    p     = float(p_init)
    t_ref = float(t.min())
    epochs_prev = None
    nbar = tbar = den = wsum = 0.0
    epochs = np.zeros(t.size, dtype=int)
    t0 = float(t_ref)

    for _ in range(max_iter):
        epochs = np.rint((t - t_ref) / p).astype(int)
        if epochs_prev is not None and np.array_equal(epochs, epochs_prev):
            break
        epochs_prev = epochs

        # Weighted LS on t = T0 + n·P (closed form via centred normal equations).
        w    = 1.0 / s ** 2
        wsum = float(w.sum())
        nbar = float((w * epochs).sum() / wsum)
        tbar = float((w * t).sum()      / wsum)
        den  = float((w * (epochs - nbar) ** 2).sum())
        if den <= 0:
            return None
        p  = float((w * (epochs - nbar) * (t - tbar)).sum() / den)
        t0 = float(tbar - p * nbar)
        t_ref = t0   # use updated T0 as reference for the next epoch round

    if not (np.isfinite(p) and np.isfinite(t0)):
        return None

    # χ² of the linear model and reduced χ².
    resid    = t - (t0 + epochs * p)
    chi2     = float(((resid / s) ** 2).sum())
    dof      = max(1, int(t.size - 2))
    redchi2  = float(chi2 / dof)

    # Formal 1-σ uncertainties from the inverse Fisher matrix
    # (σ_P² = 1/Σ w(n-n̄)²;  σ_T0² = 1/Σw + n̄²·σ_P²).
    p_err  = float(np.sqrt(1.0 / den))
    t0_err = float(np.sqrt(1.0 / wsum + nbar ** 2 / den))

    return {
        "P":       p,
        "P_err":   p_err,
        "T0":      t0,
        "T0_err":  t0_err,
        "epochs":  epochs,
        "chi2":    chi2,
        "dof":     dof,
        "redchi2": redchi2,
    }


def fit_sinusoidal_ttv(
    t_obs,
    sigma_t,
    epochs,
    p_lin: float,
    t0_lin: float,
    n_grid_p_ttv: int = 400,
) -> dict | None:
    """Fit the sinusoidal TTV ephemeris

        t_n = T0 + n*P + A * sin(2π * (T0 + n*P) / P_TTV + φ)

    Strategy
    --------
    1. Grid-scan ``P_TTV`` between ``2 P`` and ``5 × baseline_span`` on a log grid.
       For each trial value the residuals from the linear ephemeris are fitted
       by weighted LS in the basis ``[sin(2πt/P_TTV), cos(2πt/P_TTV)]`` —
       i.e.  the (A, φ) sub-problem is closed-form linear and very fast.
    2. Pick the ``P_TTV`` with smallest χ², then refine all 5 parameters
       jointly with ``scipy.optimize.least_squares`` (Levenberg–Marquardt).

    Returns ``None`` if the fit cannot be performed (too few transits or
    a degenerate system).
    """
    from scipy.optimize import least_squares

    t = np.asarray(t_obs, float)
    s = np.asarray(sigma_t, float)
    n = np.asarray(epochs, int)
    finite = np.isfinite(t) & np.isfinite(s) & np.isfinite(n) & (s > 0)
    t, s, n = t[finite], s[finite], n[finite]

    # We need at least 4 points to fit the 3-parameter sub-problem (A, P_TTV, φ)
    # with (T0, P) held fixed.  Below 4 the system is under-determined.
    if t.size < 4:
        return None
    # The full 5-parameter joint refinement only makes sense when there is
    # enough redundancy beyond the parameter count; otherwise we keep
    # (T0, P) fixed at the linear-MLE values and only refit (A, P_TTV, φ).
    do_full_refine = t.size >= 7

    # ---- 1. Grid scan over P_TTV -------------------------------------------
    t_lin   = float(t0_lin) + n * float(p_lin)
    resid   = t - t_lin
    sqrt_w  = 1.0 / s
    span    = float(t.max() - t.min())
    p_ttv_lo = max(1e-3, 2.0 * float(p_lin))
    p_ttv_hi = max(p_ttv_lo * 2, 5.0 * span)
    p_ttv_grid = np.geomspace(p_ttv_lo, p_ttv_hi, int(n_grid_p_ttv))

    best = None
    for p_ttv in p_ttv_grid:
        ang = 2.0 * np.pi * t_lin / p_ttv
        # Weighted linear LS:   resid ≈ a·sin + b·cos
        X  = np.column_stack([np.sin(ang), np.cos(ang)])
        Xw = X * sqrt_w[:, None]
        yw = resid * sqrt_w
        try:
            sol, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except Exception:
            continue
        a, b   = float(sol[0]), float(sol[1])
        model  = a * np.sin(ang) + b * np.cos(ang)
        chi2   = float(np.sum(((resid - model) / s) ** 2))
        if best is None or chi2 < best["chi2"]:
            best = {"P_ttv": float(p_ttv), "A": float(np.hypot(a, b)),
                    "phi": float(np.arctan2(b, a)), "chi2": chi2}

    if best is None:
        return None

    # ---- 2. Refinement -----------------------------------------------------
    # Full 5-parameter LM refine only when we have enough data points;
    # otherwise refit just (A, P_TTV, φ) with (T0, P) pinned to the
    # linear-MLE values to keep the problem well-posed.
    if do_full_refine:
        def _residuals(theta):
            T0_, P_, A_, P_ttv_, phi_ = theta
            t_pred = T0_ + n * P_ + A_ * np.sin(2.0 * np.pi * (T0_ + n * P_) / P_ttv_ + phi_)
            return (t - t_pred) / s
        x0 = np.array([t0_lin, p_lin, best["A"], best["P_ttv"], best["phi"]], float)
        n_free = 5
    else:
        def _residuals(theta):
            A_, P_ttv_, phi_ = theta
            t_pred = t_lin + A_ * np.sin(2.0 * np.pi * t_lin / P_ttv_ + phi_)
            return (t - t_pred) / s
        x0 = np.array([best["A"], best["P_ttv"], best["phi"]], float)
        n_free = 3

    try:
        res = least_squares(_residuals, x0, method="lm", max_nfev=5000)
        if do_full_refine:
            T0_f, P_f, A_f, Pttv_f, phi_f = (float(v) for v in res.x)
        else:
            T0_f, P_f = float(t0_lin), float(p_lin)
            A_f, Pttv_f, phi_f = (float(v) for v in res.x)
        chi2 = float(np.sum(res.fun ** 2))
        success = bool(res.success)
    except Exception:
        T0_f, P_f = float(t0_lin), float(p_lin)
        A_f, Pttv_f, phi_f = best["A"], best["P_ttv"], best["phi"]
        chi2 = best["chi2"]
        success = False

    # Wrap φ into [-π, π] and ensure A is positive (flip sign + π if needed).
    if A_f < 0:
        A_f, phi_f = -A_f, phi_f + np.pi
    phi_f = float(((phi_f + np.pi) % (2 * np.pi)) - np.pi)

    dof = max(1, t.size - n_free)
    return {
        "T0":             T0_f,
        "P":              P_f,
        "A":              A_f,
        "P_ttv":          Pttv_f,
        "phi":            phi_f,
        "chi2":           chi2,
        "dof":            int(dof),
        "redchi2":        float(chi2 / dof),
        "success":        success,
        "n_free_params":  n_free,        # 5 with full refine, 3 when (T0,P) pinned
    }


def sinusoidal_ttv_predict(epochs, T0: float, P: float,
                            A: float, P_ttv: float, phi: float) -> np.ndarray:
    """Predict transit times from the sinusoidal TTV model."""
    epochs = np.asarray(epochs, float)
    t_lin = T0 + epochs * P
    return t_lin + A * np.sin(2.0 * np.pi * t_lin / P_ttv + phi)


# ---------------------------------------------------------------------------
# Fit-window plots
# ---------------------------------------------------------------------------

def save_fit_plots(
    windows: list,
    idata,
    p_fixed: float,
    out_dir: Path,
    use: str = "median",
) -> list:
    """Save per-window fit diagnostic plots into ``out_dir``.

    Filenames are simply ``window_<k:03d>.png`` — the directory already
    encodes target / mode / run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    post = idata.posterior

    if "ttv_P" in post.data_vars:
        p_fixed = float(posterior_scalar(post, "ttv_P", use=use))

    # Shared transit parameters (present when the mode doesn't have per-window versions).
    r_shared         = posterior_scalar(post, "r",         use=use) if "r"         in post.data_vars else None
    b_shared         = posterior_scalar(post, "b",         use=use) if "b"         in post.data_vars else None
    q1_shared        = posterior_scalar(post, "q1",        use=use) if "q1"        in post.data_vars else None
    q2_shared        = posterior_scalar(post, "q2",        use=use) if "q2"        in post.data_vars else None
    c2_shared        = posterior_scalar(post, "c2",        use=use) if "c2"        in post.data_vars else 0.0
    sigma_jit_shared = posterior_scalar(post, "sigma_jit", use=use) if "sigma_jit" in post.data_vars else None
    duration_shared  = (
        float(np.exp(posterior_scalar(post, "logD", use=use)))
        if "logD" in post.data_vars else None
    )

    def _get(name: str, k: int, shared, default: float) -> float:
        """Return the per-window posterior if available, else shared, else default."""
        if f"{name}_{k}" in post.data_vars:
            return posterior_scalar(post, name, k=k, use=use)
        return float(shared) if shared is not None else float(default)

    plot_payloads: list = []
    y_values: list      = []

    for k, w in enumerate(windows):
        if "t" not in w or "f" not in w:
            continue
        t     = np.asarray(w["t"], float)
        f     = np.asarray(w["f"], float)
        t_ref = float(w.get("tc_fit", w.get("t_center", np.median(t))))
        t_rel = t - t_ref

        t0_rel = posterior_scalar(post, "t0", k=k, use=use)
        t0_abs = t_ref + float(t0_rel)
        c0_k   = posterior_scalar(post, "c0", k=k, use=use) if f"c0_{k}" in post.data_vars else 1.0

        # Duration: prefer per-window logD_k, else shared logD, else TLS fallback.
        if f"logD_{k}" in post.data_vars:
            duration_k = float(np.exp(posterior_scalar(post, "logD", k=k, use=use)))
        elif duration_shared is not None:
            duration_k = duration_shared
        else:
            duration_k = float(w.get("tls_duration_days", 0.20))
        if not np.isfinite(duration_k) or duration_k <= 0:
            duration_k = 0.20

        b_k         = _get("b",         k, b_shared,         0.5)
        r_k         = _get("r",         k, r_shared,         0.05)
        q1_k        = _get("q1",        k, q1_shared,        0.3)
        q2_k        = _get("q2",        k, q2_shared,        0.3)
        sigma_jit_k = _get("sigma_jit", k, sigma_jit_shared, 0.0)
        c2_k        = _get("c2",        k, c2_shared,        0.0)

        t_grid     = np.linspace(t.min(), t.max(), 600)
        t_grid_rel = t_grid - t_ref
        delta_grid = transit_delta(t_grid_rel, p_fixed, duration_k, r_k, b_k, q1_k, q2_k, t0_rel)

        # Compute baseline trend: GP if available, quadratic otherwise.
        has_gp_k     = f"gp_amp_{k}" in post.data_vars and f"gp_ell_{k}" in post.data_vars
        has_gp_shared = "gp_amp" in post.data_vars and "gp_ell" in post.data_vars
        trend_label   = "Trend"

        if (has_gp_k or has_gp_shared) and GaussianProcess is not None and quasisep is not None:
            amp_k = posterior_scalar(post, "gp_amp", k=k, use=use) if has_gp_k \
                    else posterior_scalar(post, "gp_amp", use=use)
            ell_k = posterior_scalar(post, "gp_ell", k=k, use=use) if has_gp_k \
                    else posterior_scalar(post, "gp_ell", use=use)

            ferr = np.asarray(w.get("f_err", []), float).ravel()
            ferr = ferr[np.isfinite(ferr) & (ferr > 0)]
            default_ferr = float(np.median(ferr)) if ferr.size > 0 else 5e-4
            sigma_tot_sq = float(sigma_jit_k) ** 2 + default_ferr ** 2
            for var_name, idx_key in [
                ("sigma_sector_extra",  "sector_index"),
                ("sigma_author_extra",  "author_index"),
                ("sigma_cadence_extra", "cadence_index"),
            ]:
                sigma_tot_sq += posterior_indexed_scalar(
                    post, var_name, w.get(idx_key, -1), use=use, default=0.0
                ) ** 2
            sigma_tot  = float(np.sqrt(max(sigma_tot_sq, 1e-12)))

            delta_data = transit_delta(t_rel, p_fixed, duration_k, r_k, b_k, q1_k, q2_k, t0_rel)
            y_base     = np.ravel(f - delta_data)
            kernel     = float(amp_k) ** 2 * quasisep.Matern32(scale=float(ell_k))
            gp_obj     = GaussianProcess(
                kernel,
                jnp.asarray(np.ravel(t_rel)),
                diag=jnp.asarray(np.full(len(t_rel), sigma_tot ** 2)),
                mean=float(c0_k),
            )
            trend_grid = np.asarray(
                gp_obj.predict(
                    jnp.asarray(y_base),
                    X_test=jnp.asarray(np.ravel(t_grid_rel)),
                    return_var=False,
                ),
                float,
            )
            trend_label = "Trend (tinygp)"
        else:
            trend_grid = np.asarray(c0_k * (1.0 + c2_k * (t_grid_rel - t0_rel) ** 2), float)

        contacts_rel = contact_times_rel(t0_rel, duration_k, b_k, r_k)
        y_values.extend([f.ravel(), trend_grid.ravel(), (trend_grid + delta_grid).ravel()])
        # x-axis: show the actual data extent with a small margin on each side.
        # Using per-window bounds avoids one outlier window (where tc_fit is near the edge)
        # inflating the x range for all other windows.
        x_margin = max(0.005, 0.05 * float(t.max() - t.min()))
        plot_payloads.append({
            "k":           k,
            "sector":      w.get("sector", -1),
            "t":           t,
            "f":           f,
            "t_grid":      t_grid,
            "trend_grid":  np.asarray(trend_grid, float),
            "f_model":     np.asarray(trend_grid + delta_grid, float),
            "t0_abs":      t0_abs,
            "t_ref":       t_ref,
            "trend_label": trend_label,
            "xlim":        (float(t.min()) - x_margin, float(t.max()) + x_margin),
            "contacts_abs": None if contacts_rel is None
                else {name: t_ref + val for name, val in contacts_rel.items()},
        })

    if y_values:
        y_all = np.concatenate(y_values)
        y_lo  = float(np.nanmin(y_all))
        y_hi  = float(np.nanmax(y_all))
        y_pad = max(1e-4, 0.05 * (y_hi - y_lo))
        common_ylim = (y_lo - y_pad, y_hi + y_pad)
    else:
        common_ylim = None

    saved = []
    for item in plot_payloads:
        plt.figure(figsize=(6, 3))
        plt.plot(item["t"], item["f"], "k.", ms=2, label="Data", zorder=1)
        plt.plot(item["t_grid"], item["f_model"], lw=2, color="tab:orange",
                 label="Trend+Transit", zorder=2)
        plt.plot(item["t_grid"], item["trend_grid"], lw=2.2, ls="--", color="tab:green",
                 alpha=0.95, label=item["trend_label"], zorder=3)
        if item["contacts_abs"] is not None:
            c = item["contacts_abs"]
            plt.axvspan(c["t1"], c["t4"], color="tab:blue", alpha=0.08, zorder=0)
            for name, colour in [("t1", "tab:blue"), ("t2", "tab:cyan"),
                                  ("t3", "tab:cyan"),  ("t4", "tab:blue")]:
                plt.axvline(c[name], ls=":", lw=1.1, alpha=0.9, color=colour, zorder=2.5)
        plt.axvline(item["t0_abs"], ls="--", lw=1, alpha=0.8, label=f"t0={item['t0_abs']:.4f}")
        plt.title(f"Window {item['k']} (sector={item['sector']})")
        plt.xlabel("Time [BTJD]")
        plt.ylabel("Flux")
        plt.xlim(*item["xlim"])
        if common_ylim is not None:
            plt.ylim(*common_ylim)
        plt.legend(fontsize=8)
        plt.tight_layout()
        p = out_dir / f"window_{item['k']:03d}.png"
        plt.savefig(p, dpi=200)
        plt.close()
        saved.append(str(p))

    return saved


# ---------------------------------------------------------------------------
# O-C and corner plots
# ---------------------------------------------------------------------------

def save_oc_plot(
    epoch, oc_min, sigma_min, keep, sector, out_path: Path,
    p_fit: float,
    redchi2: float = None,
    p_err: float = None,
    title_suffix: str = None,
):
    """Save an O-C scatter plot with uncertainty bars and keep-mask highlights.

    The title shows the MLE-fitted period (with optional 1-σ uncertainty)
    and reduced χ² of the linear-ephemeris fit.
    """
    order = np.argsort(epoch)
    ep, oc, sg, kp = epoch[order], oc_min[order], sigma_min[order], keep[order]
    sec = sector[order] if sector is not None else None

    plt.figure(figsize=(7.8, 4.8))
    plt.axhline(0, lw=1.2, color="k", alpha=0.7)
    plt.errorbar(ep, oc, yerr=sg, fmt="none", ecolor="0.55", elinewidth=1.3,
                 capsize=4, alpha=0.9, zorder=1)
    plt.errorbar(ep[kp], oc[kp], yerr=sg[kp], fmt="o", markersize=6,
                 ecolor="tab:red", elinewidth=2.2, capsize=7, zorder=3)
    plt.scatter(ep[~kp], oc[~kp], marker="x", s=70, color="tab:gray", zorder=2)
    if sec is not None:
        for x, y, s in zip(ep, oc, sec):
            plt.text(x, y, str(int(s)), fontsize=9, alpha=0.85)

    p_str = f"P={p_fit:.6f}"
    if p_err is not None and np.isfinite(p_err):
        p_str += f"±{p_err:.2e}"
    title = f"O-C ({p_str} d)  kept={int(np.sum(kp))}/{len(kp)}"
    if redchi2 is not None and np.isfinite(redchi2):
        title += f"  χ²/dof={redchi2:.2f}"
    if title_suffix:
        title += f" {title_suffix}"
    plt.xlabel("Epoch number (n)")
    plt.ylabel("O – C residual (min)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_oc_plot_sinusoidal(
    epoch, oc_min, sigma_min, keep, sector,
    out_path: Path,
    p_lin: float,
    t0_lin: float,
    sine_fit: dict,
    chi2_lin: float,
    dof_lin: int,
):
    """O-C plot with the sinusoidal TTV model overlaid in epoch space.

    The black points and error bars are the same as in the linear O-C plot
    (residuals from the linear ephemeris ``T0_lin + n·P_lin``).  The purple
    curve is the sinusoidal model evaluated on a dense epoch grid, *also*
    referenced to the linear ephemeris — so both share the same y-axis and
    "0" line means "consistent with linear timing".
    """
    order = np.argsort(epoch)
    ep, oc, sg, kp = epoch[order], oc_min[order], sigma_min[order], keep[order]
    sec = sector[order] if sector is not None else None

    plt.figure(figsize=(8.0, 5.0))
    plt.axhline(0, lw=1.2, color="k", alpha=0.7, label="Linear ephemeris")

    # ----- Sinusoidal model overlay -----
    n_dense = np.linspace(int(epoch.min()) - 1, int(epoch.max()) + 1, 1000)
    t_sine  = sinusoidal_ttv_predict(
        n_dense,
        T0=sine_fit["T0"], P=sine_fit["P"],
        A=sine_fit["A"],   P_ttv=sine_fit["P_ttv"],
        phi=sine_fit["phi"],
    )
    # Express as O-C relative to the *linear* reference shown by the data points.
    sine_oc_min = (t_sine - (t0_lin + n_dense * p_lin)) * MINUTES_PER_DAY
    plt.plot(n_dense, sine_oc_min, "-", color="tab:purple", lw=2.0, alpha=0.9,
             label=(f"Sinusoidal: A={sine_fit['A'] * MINUTES_PER_DAY:.2f} min, "
                    f"P_TTV={sine_fit['P_ttv']:.2f} d"))

    # ----- Data points -----
    plt.errorbar(ep, oc, yerr=sg, fmt="none", ecolor="0.55", elinewidth=1.3,
                 capsize=4, alpha=0.9, zorder=1)
    plt.errorbar(ep[kp], oc[kp], yerr=sg[kp], fmt="o", markersize=6,
                 ecolor="tab:red", elinewidth=2.0, capsize=6, zorder=3, label="Kept")
    plt.scatter(ep[~kp], oc[~kp], marker="x", s=70, color="tab:gray", zorder=2, label="Rejected")
    if sec is not None:
        for x, y, s in zip(ep, oc, sec):
            plt.text(x, y, str(int(s)), fontsize=9, alpha=0.85)

    chi2_sin    = float(sine_fit["chi2"])
    dof_sin     = int(sine_fit["dof"])
    redchi2_sin = float(sine_fit["redchi2"])
    redchi2_lin = float(chi2_lin / max(1, dof_lin))
    delta_chi2  = float(chi2_lin) - chi2_sin     # > 0 means sinusoid improves the fit

    # Number of extra free parameters in the sine fit relative to the linear
    # model (linear has 2: T0, P).  Significance thresholds for χ²(Δdof):
    #   Δdof=1 → Δχ² > 3.84 (2σ),  9.0 (3σ)
    #   Δdof=3 → Δχ² > 7.81 (2σ), 14.2 (3σ)
    n_free_sin   = int(sine_fit.get("n_free_params", 5))
    delta_dof    = max(1, n_free_sin - 2)
    sigma3_thr   = {1: 9.0, 3: 14.2}.get(delta_dof, None)
    threshold_str = f", Δχ² > {sigma3_thr:.1f} ≈ 3σ" if sigma3_thr else ""

    title = (
        f"O-C   linear: χ²/dof = {redchi2_lin:.2f} ({chi2_lin:.1f}/{dof_lin})   |   "
        f"sine: χ²/dof = {redchi2_sin:.2f} ({chi2_sin:.1f}/{dof_sin})\n"
        f"Δχ² = χ²_lin − χ²_sin = {delta_chi2:.2f}   "
        f"({delta_dof} extra param{'s' if delta_dof != 1 else ''}{threshold_str})"
    )
    plt.xlabel("Epoch number (n)")
    plt.ylabel("O – C residual (min)")
    plt.title(title, fontsize=10)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_corner_plot(idata, out_path: Path) -> str | None:
    """Save a corner-style posterior pair plot."""
    from tesspipe.arviz_runtime import configure_arviz_runtime, release_arviz_runtime
    configure_arviz_runtime()
    try:
        import arviz as az
    finally:
        release_arviz_runtime()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vars_ok = list(idata.posterior.data_vars.keys())
    if len(vars_ok) < 2:
        return None
    try:
        az.plot_pair(idata, var_names=vars_ok, kind="kde", marginals=True)
    except Exception as e:
        keep_n = min(20, len(vars_ok))
        print(f"[WARN] Full corner plot failed ({e}); falling back to first {keep_n} variables.")
        az.plot_pair(idata, var_names=vars_ok[:keep_n], kind="kde", marginals=True)
    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return str(out_path)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Post-processing entry point  (callable both inline from ppl.py and via CLI)
# ---------------------------------------------------------------------------

def _tic_from_run_dir(run_dir: Path) -> int | None:
    """Recover the TIC ID from a run-dir of the form
    ``OUTPUT/tic_<TIC>/mode<M>/runs/<TS>/`` — returns ``None`` on failure.
    """
    for part in run_dir.resolve().parts:
        if part.startswith("tic_"):
            try:
                return int(part[len("tic_"):])
            except ValueError:
                return None
    return None


def run_post_processing(
    run_dir: Path,
    idata=None,
    windows=None,
    wjs_csv: Path = None,
    frac_lo: float = 0.8,
    frac_hi: float = 1.2,
    k_interval: float = 6.0,
) -> dict:
    """Run all period/ephemeris post-processing on one MCMC run.

    Either pass ``idata`` and ``windows`` directly (no disk round-trip — used
    by ``ppl.py`` after a fresh MCMC), or omit them and let the function read
    ``idata.pkl`` / ``windows.pkl`` from ``run_dir`` (used by the CLI when
    re-running post-processing on archived results).

    Writes ``oc.png``, ``oc_sinusoidal.png``, ``corner.png``, ``fit/window_*.png``
    and ``summary.json`` into ``run_dir``.  Returns the summary payload dict.
    """
    run_dir = Path(run_dir).resolve()
    if wjs_csv is None:
        wjs_csv = Path(__file__).resolve().parents[1] / "data" / "WJs.csv"

    if idata is None or windows is None:
        idata_file   = run_dir / "idata.pkl"
        windows_file = run_dir / "windows.pkl"
        if not idata_file.exists() or not windows_file.exists():
            raise FileNotFoundError(f"Missing idata.pkl / windows.pkl in {run_dir}")
        print(f"[load] {idata_file}")
        idata = _load_pickle(idata_file)
        print(f"[load] {windows_file}")
        windows = _load_pickle(windows_file)

    tic_id  = _tic_from_run_dir(run_dir)
    p_prior = get_period_prior_from_wjs(tic_id, wjs_csv)
    print(f"[post] TIC={tic_id}  P_prior={p_prior}")
    if p_prior is None:
        raise RuntimeError(f"Could not look up P prior for TIC {tic_id} in {wjs_csv}")

    # --- Posterior transit times -------------------------------------------
    rows       = get_abs_t0_and_rows(idata, windows, use="median", ref_key="tc_fit")
    t_obs      = np.asarray([r["t0_abs"] for r in rows], float)
    sector     = np.asarray([r["sector"] for r in rows], int)
    sigma_min  = get_sigma_t_minutes_from_idata(idata, rows)
    sigma_days = sigma_min / MINUTES_PER_DAY
    print(f"[post] N transits = {len(t_obs)}   baseline = {t_obs.max() - t_obs.min():.2f} d")

    # --- Prior-guided P grid search (initialises the MLE) ------------------
    p_best, info_best, (p_min, p_max) = find_best_period_prior_guided(
        t_obs, sigma_days, p_prior,
        frac_lo=frac_lo, frac_hi=frac_hi,
        k_max=800, keep_top=50, refine_steps=3, refine_span=0.02,
    )
    print(
        f"[grid] P_best={p_best:.8f} d  score={info_best['score']:.3f} "
        f"chi2={info_best['chi2']:.3f}  range=[{p_min:.4f}, {p_max:.4f}]"
    )

    keep_u, _ = keep_mask_from_intervals(t_obs, sigma_min, p_best, k_interval=k_interval)
    print(f"[mask] kept {int(np.sum(keep_u))}/{len(keep_u)}")

    # --- MLE linear ephemeris (iterative epoch + weighted LS) --------------
    mle = fit_linear_ephemeris_mle(t_obs, sigma_days, p_init=p_best)
    if mle is None:
        print("[MLE] failed; falling back to prior-guided P with cumulative-Δt epochs.")
        t_sort_idx        = np.argsort(t_obs)
        k_sorted          = np.clip(np.rint(np.diff(t_obs[t_sort_idx]) / p_best).astype(int), 0, None)
        epoch_sorted      = np.zeros(len(t_obs), dtype=int)
        epoch_sorted[1:]  = np.cumsum(k_sorted)
        epoch             = np.zeros(len(t_obs), dtype=int)
        epoch[t_sort_idx] = epoch_sorted
        p_fit, t0_fit     = float(p_best), float(t_obs[0])
        p_err = t0_err = chi2 = redchi2 = float("nan")
        dof = max(1, len(t_obs) - 2)
    else:
        epoch   = mle["epochs"]
        p_fit   = mle["P"]
        t0_fit  = mle["T0"]
        p_err   = mle["P_err"]
        t0_err  = mle["T0_err"]
        chi2    = mle["chi2"]
        dof     = mle["dof"]
        redchi2 = mle["redchi2"]
        print(
            f"[MLE] P  = {p_fit:.8f} ± {p_err:.2e} d   "
            f"T0 = {t0_fit:.6f} ± {t0_err:.6f}\n"
            f"[MLE] chi2/dof = {chi2:.2f}/{dof} = {redchi2:.3f}"
        )

    oc_min = (t_obs - (t0_fit + epoch * p_fit)) * MINUTES_PER_DAY

    # --- Sinusoidal TTV fit ------------------------------------------------
    sine = fit_sinusoidal_ttv(
        t_obs=t_obs, sigma_t=sigma_days, epochs=epoch,
        p_lin=p_fit, t0_lin=t0_fit,
    )
    if sine is not None:
        delta_chi2 = (chi2 - sine["chi2"]) if np.isfinite(chi2) else float("nan")
        print(
            f"[sine] A = {sine['A'] * MINUTES_PER_DAY:.2f} min   "
            f"P_TTV = {sine['P_ttv']:.3f} d   phi = {sine['phi']:.3f} rad\n"
            f"[sine] chi2/dof = {sine['chi2']:.2f}/{sine['dof']} = {sine['redchi2']:.3f}   "
            f"Δχ²(lin−sin) = {delta_chi2:.2f}  "
            f"(n_free_extra={sine.get('n_free_params', 5) - 2})"
        )
    else:
        print("[sine] not enough transits for a sinusoidal TTV fit (need ≥ 4).")

    # --- Plots --------------------------------------------------------------
    fit_dir      = run_dir / "fit"
    oc_path      = run_dir / "oc.png"
    oc_sine_path = run_dir / "oc_sinusoidal.png"
    cor_path     = run_dir / "corner.png"

    fit_images = save_fit_plots(windows, idata, p_fixed=p_best, out_dir=fit_dir, use="median")
    save_oc_plot(
        epoch, oc_min, sigma_min, keep_u, sector, oc_path,
        p_fit=p_fit, p_err=p_err, redchi2=redchi2,
    )
    if sine is not None:
        save_oc_plot_sinusoidal(
            epoch, oc_min, sigma_min, keep_u, sector, oc_sine_path,
            p_lin=p_fit, t0_lin=t0_fit,
            sine_fit=sine, chi2_lin=chi2, dof_lin=dof,
        )
    corner_plot = save_corner_plot(idata, cor_path)

    # χ² on the kept points only.
    n_kept       = int(np.sum(keep_u))
    chi2_kept    = float(((oc_min[keep_u] / sigma_min[keep_u]) ** 2).sum()) if n_kept > 0 else float("nan")
    dof_kept     = max(1, n_kept - 2)
    redchi2_kept = float(chi2_kept / dof_kept) if np.isfinite(chi2_kept) else float("nan")

    # --- summary.json -------------------------------------------------------
    sinusoidal_summary = None if sine is None else {
        "T0":            sine["T0"],
        "P":             sine["P"],
        "A_days":        sine["A"],
        "A_minutes":     sine["A"] * MINUTES_PER_DAY,
        "P_ttv_days":    sine["P_ttv"],
        "phi_rad":       sine["phi"],
        "n_free_params": int(sine.get("n_free_params", 5)),
        "chi2":          sine["chi2"],
        "dof":           sine["dof"],
        "redchi2":       sine["redchi2"],
        "delta_chi2_vs_linear": (
            float(chi2 - sine["chi2"]) if np.isfinite(chi2) else None
        ),
        "lm_converged":  sine["success"],
    }

    payload = {
        "tic_id":  tic_id,
        "p_prior": p_prior,
        "ephemeris": {
            "p_grid":       float(p_best),
            "p_grid_min":   float(p_min),
            "p_grid_max":   float(p_max),
            "p_mle":        float(p_fit),
            "p_mle_err":    float(p_err)        if np.isfinite(p_err)        else None,
            "t0_mle":       float(t0_fit),
            "t0_mle_err":   float(t0_err)       if np.isfinite(t0_err)       else None,
            "chi2":         float(chi2)         if np.isfinite(chi2)         else None,
            "dof":          int(dof),
            "redchi2":      float(redchi2)      if np.isfinite(redchi2)      else None,
            "chi2_kept":    float(chi2_kept)    if np.isfinite(chi2_kept)    else None,
            "dof_kept":     int(dof_kept),
            "redchi2_kept": float(redchi2_kept) if np.isfinite(redchi2_kept) else None,
        },
        "sinusoidal_ttv": sinusoidal_summary,
        "windows": {
            "n_total": int(len(t_obs)),
            "n_kept":  n_kept,
        },
        "files": {
            "idata":         "idata.pkl",
            "windows":       "windows.pkl",
            "oc":            "oc.png",
            "oc_sinusoidal": "oc_sinusoidal.png" if sine is not None else None,
            "corner":        "corner.png"        if corner_plot is not None else None,
            "fit":           [f"fit/{Path(p).name}" for p in fit_images],
        },
    }
    out_json = run_dir / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Saved] summary       -> {out_json}")
    print(f"[Saved] fit plots     -> {fit_dir}  ({len(fit_images)} files)")
    print(f"[Saved] O-C           -> {oc_path}")
    if sine is not None:
        print(f"[Saved] O-C (sine)    -> {oc_sine_path}")
    if corner_plot:
        print(f"[Saved] corner        -> {cor_path}")
    return payload


def main():
    """CLI wrapper for ``run_post_processing`` — reprocess an existing run."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Reprocess one MCMC run: redo period fit + plots + summary.json."
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to the run directory (contains idata.pkl, windows.pkl).")
    parser.add_argument("--wjs-csv", type=Path, default=project_root / "data" / "WJs.csv",
                        help="WJs catalogue CSV used to look up the period prior.")
    parser.add_argument("--frac-lo",    type=float, default=0.8)
    parser.add_argument("--frac-hi",    type=float, default=1.2)
    parser.add_argument("--k-interval", type=float, default=6.0)
    args = parser.parse_args()
    run_post_processing(
        run_dir=args.run_dir, wjs_csv=args.wjs_csv,
        frac_lo=args.frac_lo, frac_hi=args.frac_hi, k_interval=args.k_interval,
    )


if __name__ == "__main__":
    main()
