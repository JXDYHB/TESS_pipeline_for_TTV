"""MCMC model builder: window selection + NumPyro/JAX inference.

Entry point: build_model_mcmc().

The model structure is:
  - shared transit-shape parameters (r, b, duration, LD) sampled once
  - per-window transit-time offset t0_k ~ TruncatedNormal(0, 20 min)
  - per-window baseline (quadratic for mode 0, tinygp for modes 1/3)
  - hierarchical white-noise terms per sector / author / cadence group
"""

import importlib
import os
import time
from dataclasses import dataclass, field, fields as dc_fields
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS

try:
    from tinygp import GaussianProcess
    from tinygp.kernels import quasisep
except Exception:
    GaussianProcess = None
    quasisep        = None

print = partial(print, flush=True)

# ---------------------------------------------------------------------------
# Physical / timing constants
# ---------------------------------------------------------------------------
MINUTES_PER_DAY    = 1440.0
T0_PRIOR_SIGMA_DAYS = 20.0 / MINUTES_PER_DAY   # 20-minute prior σ on per-window t0

# ---------------------------------------------------------------------------
# Module-level caches / state
# ---------------------------------------------------------------------------
_IMPORT_STATUS    = {"collect_windows": False, "arviz": False}
_MODE_MODULE_CACHE: dict = {}


# ---------------------------------------------------------------------------
# ModelContext — static MCMC configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelContext:
    """All static parameters needed by every mode's build_context / run_window.

    Created once in build_model_mcmc() and shared (read-only) across the
    joint_transit_model closure and all mode functions.  Treat as immutable
    after construction.
    """
    # Transit shape priors
    r_lo: float
    r_hi: float
    duration_prior_days: float | None
    duration_prior_low: float | None
    duration_prior_high: float | None
    duration_prior_sigma_days: float | None
    window: float

    # Limb-darkening helper: (q1, q2) -> (u1, u2)
    q_to_u: object

    # GP backend — None when tinygp is not installed
    GaussianProcess: type | None
    quasisep: object | None

    # White-noise metadata
    default_flux_err: float
    n_sectors: int
    n_authors: int
    n_cadences: int
    sector_white_prior_scales:  np.ndarray = field(repr=False, compare=False, hash=False)
    author_white_prior_scales:  np.ndarray = field(repr=False, compare=False, hash=False)
    cadence_white_prior_scales: np.ndarray = field(repr=False, compare=False, hash=False)

    def as_dict(self) -> dict:
        """Shallow dict representation — used to merge with mode-specific samples."""
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


# ---------------------------------------------------------------------------
# Kipping (q1, q2) → quadratic LD (u1, u2)
# ---------------------------------------------------------------------------

def kipping_q_to_u(q1, q2):
    """Convert Kipping limb-darkening parameters to the standard (u1, u2) form."""
    s = jnp.sqrt(q1)
    return 2 * s * q2, s * (1 - 2 * q2)


# ---------------------------------------------------------------------------
# NumPy statistics helpers
# ---------------------------------------------------------------------------

def _mad_sigma_np(x: np.ndarray) -> float:
    """Robust σ estimate via MAD, falling back to std when MAD is zero."""
    x   = np.asarray(x, float)
    x   = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
    s   = 1.4826 * mad
    return float(s) if np.isfinite(s) and s > 0 else float(np.nanstd(x))


def _positive_median(x, fallback: float) -> float:
    """Median of positive finite values, or fallback when none exist."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.median(x)) if x.size > 0 else float(fallback)


def _rounded_exptime_key(exptime) -> str:
    """Stable string key for grouping products by cadence (e.g. '120s')."""
    try:
        v = float(exptime)
    except Exception:
        return "unknown"
    return "unknown" if not np.isfinite(v) or v <= 0 else f"{int(round(v))}s"


# ---------------------------------------------------------------------------
# Timing / lazy-import helpers
# ---------------------------------------------------------------------------

def _import_elapsed_s() -> float:
    """Seconds since TESSPIPE_IMPORT_T0 was set in the environment."""
    try:
        return time.perf_counter() - float(os.environ["TESSPIPE_IMPORT_T0"])
    except Exception:
        return float("nan")


def _get_mode_module(mode: int):
    """Import the selected mode module on first use and cache it."""
    mode = int(mode)
    if mode not in _MODE_MODULE_CACHE:
        mod = importlib.import_module(f"tesspipe.modes.mode{mode}")
        _MODE_MODULE_CACHE[mode] = mod
        print(f"[import] tesspipe.modes.mode{mode} ok (+{_import_elapsed_s():.1f}s)")
    return _MODE_MODULE_CACHE[mode]


def _get_arviz():
    """Import ArviZ lazily — only needed after MCMC finishes."""
    from tesspipe.arviz_runtime import configure_arviz_runtime, release_arviz_runtime
    configure_arviz_runtime()
    try:
        mod = importlib.import_module("arviz")
        if not _IMPORT_STATUS["arviz"]:
            print(f"[import] arviz ok (+{_import_elapsed_s():.1f}s)")
            _IMPORT_STATUS["arviz"] = True
        return mod
    finally:
        release_arviz_runtime()


def _collect_windows(*args, **kwargs):
    """Import transit-window collection lazily to avoid slow startup."""
    from tesspipe.transit_window import collect_windows
    if not _IMPORT_STATUS["collect_windows"]:
        print(f"[import] tesspipe.transit_window ok (+{_import_elapsed_s():.1f}s)")
        _IMPORT_STATUS["collect_windows"] = True
    return collect_windows(*args, **kwargs)


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------

def estimate_sector_sigma_map(
    sector_data: dict,
    sectors: list,
    tic_label: float,
    low_quantile: float = 0.20,
) -> dict:
    """Estimate per-sector flux scatter as a white-noise scale proxy.

    Drops the lowest `low_quantile` fraction of flux values to reduce
    the influence of transits and flares before computing the MAD σ.
    """
    sigma_map = {}
    for s in sectors:
        lc = sector_data.get((int(s), float(tic_label)))
        if lc is None:
            sigma_map[int(s)] = np.nan
            continue
        f = np.asarray(lc.flux.value, float)
        f = f[np.isfinite(f)]
        if f.size < 50:
            sigma_map[int(s)] = np.nan
            continue
        q    = float(np.nanquantile(f, float(low_quantile)))
        keep = f >= q
        sigma_map[int(s)] = _mad_sigma_np(f[keep] if np.any(keep) else f)
    return sigma_map


def build_noise_metadata(windows: list, sigma_sector_map: dict) -> dict:
    """Attach white-noise metadata and per-group prior scales to each window.

    Mutates each window in-place with sector_index, author_index, and
    cadence_index fields, then returns a metadata dict.
    """
    # Collect all ferr values to compute a global default.
    all_ferr = []
    for w in windows:
        ferr = np.asarray(w.get("f_err", []), float).ravel()
        ferr = ferr[np.isfinite(ferr) & (ferr > 0)]
        if ferr.size > 0:
            all_ferr.append(ferr)
    default_flux_err = (
        _positive_median(np.concatenate(all_ferr), 5e-4) if all_ferr else 5e-4
    )

    sector_values  = sorted({int(w.get("sector", -1)) for w in windows if int(w.get("sector", -1)) >= 0})
    author_values  = sorted({str(w.get("author", "UNKNOWN")).upper() for w in windows})
    cadence_values = sorted({_rounded_exptime_key(w.get("exptime", np.nan)) for w in windows})

    sector_index  = {s: i for i, s in enumerate(sector_values)}
    author_index  = {a: i for i, a in enumerate(author_values)}
    cadence_index = {c: i for i, c in enumerate(cadence_values)}

    def _group_scales(group_keys, key_fn):
        """Prior scale for each group: median ferr of windows in that group."""
        scales = []
        for key in group_keys:
            ferr_list = []
            for w in windows:
                if key_fn(w) != key:
                    continue
                ferr = np.asarray(w.get("f_err", []), float).ravel()
                ferr = ferr[np.isfinite(ferr) & (ferr > 0)]
                if ferr.size > 0:
                    ferr_list.append(ferr)
            scale = _positive_median(np.concatenate(ferr_list), default_flux_err) if ferr_list else default_flux_err
            scales.append(max(scale, 0.5 * default_flux_err))
        return scales

    sector_prior_scales = []
    for sec in sector_values:
        sigma_s = float(sigma_sector_map.get(int(sec), np.nan))
        scale   = sigma_s if np.isfinite(sigma_s) and sigma_s > 0 else default_flux_err
        sector_prior_scales.append(max(0.25 * scale, 0.5 * default_flux_err))

    author_prior_scales  = _group_scales(author_values,  lambda w: str(w.get("author", "UNKNOWN")).upper())
    cadence_prior_scales = _group_scales(cadence_values, lambda w: _rounded_exptime_key(w.get("exptime", np.nan)))

    for w in windows:
        sector  = int(w.get("sector", -1))
        author  = str(w.get("author", "UNKNOWN")).upper()
        cadence = _rounded_exptime_key(w.get("exptime", np.nan))
        w["sector_index"]  = int(sector_index.get(sector, -1))
        w["author"]        = author
        w["author_index"]  = int(author_index.get(author, -1))
        w["cadence_group"] = cadence
        w["cadence_index"] = int(cadence_index.get(cadence, -1))

    return {
        "default_flux_err":            float(default_flux_err),
        "sector_values":               sector_values,
        "author_values":               author_values,
        "cadence_values":              cadence_values,
        "sector_white_prior_scales":   np.asarray(sector_prior_scales, float),
        "author_white_prior_scales":   np.asarray(author_prior_scales, float),
        "cadence_white_prior_scales":  np.asarray(cadence_prior_scales, float),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_model_mcmc(
    sector_data: dict,
    tic: float = 120.0,
    window: float = 0.5,
    sectors=None,
    max_windows_per_sector: int = 10,

    # Window selection parameters.
    W: int = 30,
    k_sigma: float = 3.0,
    min_run: int = 3,
    depth_lo: float = 0.01,
    depth_hi: float = 0.03,
    baseline_frac: float = 0.70,
    debug_windows: bool = False,
    duration_hours=None,
    period_days_prior=None,
    t0_days_prior=None,
    time=None,
    min_duration_coverage: float = 0.60,

    # Centre-refinement parameters.
    n_low: int = 15,
    refine_halfspan_cadences: int = 3,
    n_scan: int = 21,
    min_pairs: int = 8,
    baseline_min_pts: int = 10,

    # MCMC parameters.
    num_warmup: int = 400,
    num_samples: int = 2000,
    num_chains: int = 1,
    target_accept_prob: float = 0.92,
    max_tree_depth: int = 9,
    rng_seed: int = 0,
    platform: str = "gpu",
    enable_x64: bool = False,
    use_catalog_duration_prior: bool = True,
    mode: int = 0,

    # Run identity (used for debug plot paths).
    tic_id=None,
    run_index=None,
) -> tuple:
    """Select transit windows and run a joint NumPyro MCMC model.

    Parameters
    ----------
    mode:
        0 – quadratic baseline + white noise (shared transit shape)
        1 – tinygp Matérn-3/2 per-window baseline (shared transit shape)
        3 – mode 1 + independent per-window transit duration

    Returns
    -------
    (windows, mcmc, samples, idata, summary)
    """
    # --- JAX / NumPyro setup ---
    jax.config.update("jax_enable_x64", bool(enable_x64))
    if str(platform).lower() == "cpu" and int(num_chains) > 1:
        try:
            numpyro.set_host_device_count(int(num_chains))
        except Exception as e:
            print(f"[WARN] Failed to set host device count to {int(num_chains)}: {e}")
    numpyro.set_platform(platform)
    print("JAX devices:", jax.devices())

    if sectors is None:
        sectors = sorted({int(sec) for (sec, _) in sector_data.keys()})
    else:
        sectors = [int(s) for s in sectors]

    tic_label        = float(tic)
    sigma_sector_map = estimate_sector_sigma_map(sector_data, sectors, tic_label)
    print(f"[build_model_mcmc] sigma_sector_map: {sigma_sector_map}")

    # --- Duration prior from catalogue ---
    duration_prior_days = duration_prior_sigma_days = duration_prior_low = duration_prior_high = None
    if (
        bool(use_catalog_duration_prior)
        and duration_hours is not None
        and np.isfinite(duration_hours)
        and float(duration_hours) > 0
    ):
        # Clip the catalogue duration to a reasonable range, but allow up to
        # 2 days so long-duration targets (e.g. P~260d planets with ~30 h
        # transits) are not artificially compressed into a 0.8-day prior.
        dp     = float(np.clip(float(duration_hours) / 24.0, 0.03, 2.0))
        dp_sig = float(max(0.01, 0.15 * dp))
        dp_lo  = float(max(0.03, 0.4 * dp))
        dp_hi  = float(min(3.0, 2.5 * dp))
        if dp_lo >= dp_hi:
            dp_lo = float(max(0.03, dp - 3.0 * dp_sig))
            dp_hi = float(min(1.5, dp + 3.0 * dp_sig))
        duration_prior_days       = dp
        duration_prior_sigma_days = dp_sig
        duration_prior_low        = dp_lo
        duration_prior_high       = dp_hi
        print(
            f"[build_model_mcmc] duration prior: "
            f"TruncNormal(loc={dp:.4g} d, σ={dp_sig:.4g} d, [{dp_lo:.4g}, {dp_hi:.4g}])"
        )
    else:
        print("[build_model_mcmc] duration prior: free (no catalogue prior)")

    # --- Radius-ratio prior from transit depth ---
    if np.isfinite(depth_lo) and np.isfinite(depth_hi) and float(depth_hi) > float(depth_lo) > 0:
        r_lo = float(np.sqrt(float(depth_lo)))
        r_hi = float(np.sqrt(float(depth_hi)))
    else:
        r_lo, r_hi = 0.001, 0.2
    r_lo = float(np.clip(r_lo, 5e-4, 0.5))
    r_hi = float(np.clip(max(r_hi, r_lo * 1.05), r_lo * 1.05, 0.5))
    print(
        f"[build_model_mcmc] radius-ratio prior: r ∈ [{r_lo:.4g}, {r_hi:.4g}] "
        f"(depth ∈ [{float(depth_lo):.4g}, {float(depth_hi):.4g}])"
    )

    # --- Window selection ---
    windows = _collect_windows(
        sectors=sectors,
        sector_data=sector_data,
        tic=float(tic_label),
        W=int(W),
        k_sigma=float(k_sigma),
        min_run=int(min_run),
        max_candidates_per_sector=30,
        depth_lo=float(depth_lo),
        depth_hi=float(depth_hi),
        window=float(window),
        n_low=int(n_low),
        refine_halfspan_cadences=int(refine_halfspan_cadences),
        n_scan=int(n_scan),
        min_pairs=int(min_pairs),
        baseline_frac=float(baseline_frac),
        baseline_min_pts=int(baseline_min_pts),
        duration_hours=float(duration_hours) if duration_hours is not None else None,
        period_days_prior=float(period_days_prior) if period_days_prior is not None else None,
        t0_days_prior=float(t0_days_prior) if t0_days_prior is not None else None,
        time=time,
        min_duration_coverage=float(min_duration_coverage),
        max_windows_per_sector=int(max_windows_per_sector),
        debug=bool(debug_windows),
        print_diagnostics=False,
        tic_id=tic_id,
        run_index=run_index,
    )
    print(f"[build_model_mcmc] collected windows: {len(windows)}")
    if len(windows) == 0:
        raise RuntimeError("No accepted windows.")

    # --- Mode validation ---
    mode = int(mode)
    if mode not in (0, 1, 3):
        raise ValueError(f"Unsupported mode={mode}. Expected 0, 1, or 3.")
    if mode in (1, 3) and (GaussianProcess is None or quasisep is None):
        raise RuntimeError("mode=1/3 requires tinygp to be installed.")
    mode_names = {0: "quadratic", 1: "tinygp-per-window", 3: "tinygp-per-window+duration-per-window"}
    print(f"[build_model_mcmc] mode={mode} ({mode_names[mode]})")

    noise_meta = build_noise_metadata(windows, sigma_sector_map)
    print(
        f"[build_model_mcmc] noise groups — "
        f"sectors={noise_meta['sector_values']} "
        f"authors={noise_meta['author_values']} "
        f"cadences={noise_meta['cadence_values']} "
        f"default_σ={noise_meta['default_flux_err']:.4g}"
    )

    # --- Pre-process window arrays onto device ---
    for w in windows:
        if "t" not in w:
            continue
        t_np    = np.asarray(w["t"], float).ravel()
        f_np    = np.asarray(w.get("f", []), float).ravel()
        ferr_np = np.asarray(w.get("f_err", np.full_like(f_np, np.nan)), float).ravel()
        if ferr_np.shape != f_np.shape:
            ferr_np = np.full_like(f_np, np.nan)

        keep               = np.isfinite(t_np) & np.isfinite(f_np)
        t_np, f_np, ferr_np = t_np[keep], f_np[keep], ferr_np[keep]

        if t_np.size >= 2:
            idx               = np.argsort(t_np)
            t_s, f_s, ferr_s  = t_np[idx], f_np[idx], ferr_np[idx]
            dt                = np.diff(t_s)
            dt                = dt[np.isfinite(dt) & (dt > 0)]
            cadence           = float(np.median(np.abs(dt))) if dt.size > 0 else np.nan
            span              = float(max(1e-4, t_s[-1] - t_s[0]))
        else:
            t_s, f_s, ferr_s  = t_np, f_np, ferr_np
            cadence = span = np.nan

        cadence = cadence if np.isfinite(cadence) and cadence > 0 else 1e-4
        span    = span    if np.isfinite(span)    and span    > 0 else max(1e-4, float(window))

        tc_ref          = float(w.get("tc_fit", w.get("t_center", np.nanmedian(t_s) if t_s.size else 0.0)))
        t_jax           = jax.device_put(jnp.asarray(t_s))
        w["t"]          = t_jax
        w["f"]          = jax.device_put(jnp.asarray(f_s))
        w["f_err"]      = jax.device_put(jnp.asarray(ferr_s))
        w["t_rel"]      = t_jax - tc_ref
        w["tc_ref_abs"] = float(tc_ref)
        w["t_cadence"]  = float(cadence)
        w["t_span"]     = float(span)

    # --- Build ModelContext and load mode module ---
    base_ctx = ModelContext(
        r_lo=r_lo, r_hi=r_hi,
        duration_prior_days=duration_prior_days,
        duration_prior_low=duration_prior_low,
        duration_prior_high=duration_prior_high,
        duration_prior_sigma_days=duration_prior_sigma_days,
        window=window,
        q_to_u=kipping_q_to_u,
        GaussianProcess=GaussianProcess,
        quasisep=quasisep,
        default_flux_err=noise_meta["default_flux_err"],
        n_sectors=len(noise_meta["sector_values"]),
        n_authors=len(noise_meta["author_values"]),
        n_cadences=len(noise_meta["cadence_values"]),
        sector_white_prior_scales=noise_meta["sector_white_prior_scales"],
        author_white_prior_scales=noise_meta["author_white_prior_scales"],
        cadence_white_prior_scales=noise_meta["cadence_white_prior_scales"],
    )
    # Pre-compute once; mode_ctx merges this with the mode's sampled variables.
    base_ctx_dict = base_ctx.as_dict()

    mode_module        = _get_mode_module(mode)
    mode_runner        = mode_module.run_window
    mode_build_context = mode_module.build_context

    def joint_transit_model(windows_local):
        """NumPyro model: shared transit shape, independent per-window transit times."""
        mode_ctx = {**base_ctx_dict, **mode_build_context(base_ctx)}
        for k, w in enumerate(windows_local):
            t_rel = jnp.ravel(w["t_rel"])
            f     = jnp.ravel(w["f"])
            t0    = numpyro.sample(
                f"t0_{k}",
                dist.TruncatedNormal(low=-0.05, high=0.05, loc=0.0, scale=T0_PRIOR_SIGMA_DAYS),
            )
            mode_runner(k, w, t_rel, f, t0, mode_ctx)

    # --- NUTS kernel + MCMC ---
    kernel = NUTS(
        model=joint_transit_model,
        target_accept_prob=float(target_accept_prob),
        max_tree_depth=int(max_tree_depth),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        progress_bar=True,
    )
    print(
        f"[build_model_mcmc] mcmc.run start "
        f"(warmup={num_warmup}, samples={num_samples}, chains={num_chains})"
    )
    mcmc.run(jax.random.PRNGKey(int(rng_seed)), windows)
    print("[build_model_mcmc] mcmc.run done")

    # --- Diagnostics ---
    try:
        diag = numpyro_summary(mcmc.get_samples(group_by_chain=True), prob=0.90)
        print()
        print("                         mean           std        median"
              "          5.0%         95.0%         n_eff         r_hat")
        for name in sorted(diag.keys()):
            row  = diag[name]
            vals = [float(np.asarray(row[k])) for k in ("mean", "std", "median", "5.0%", "95.0%", "n_eff", "r_hat")]
            print(f"{name:>15} " + " ".join(f"{v:>14.10f}" for v in vals))
        print()
    except Exception:
        mcmc.print_summary()

    samples = mcmc.get_samples()
    az      = _get_arviz()
    idata   = az.from_numpyro(mcmc)

    # Restore windows to NumPy arrays for downstream plotting / serialisation.
    # Keep JAX arrays alive until az.from_numpyro() finishes, because ArviZ
    # may re-trace the model with the original window arguments.
    for w in windows:
        for key in ("t", "f", "f_err"):
            if key in w:
                w[key] = np.asarray(w[key], float)
        w.pop("t_rel", None)

    if "sigma_jit" in samples:
        sj = np.asarray(samples["sigma_jit"])
    else:
        sj_list = [np.asarray(v) for k, v in samples.items() if str(k).startswith("sigma_jit_")]
        sj = np.concatenate([x.ravel() for x in sj_list]) if sj_list else np.asarray([np.nan])

    summary = {
        "sigma_jit_mean":   float(sj.mean()),
        "sigma_jit_std":    float(sj.std()),
        "n_windows":        int(len(windows)),
        "mode":             int(mode),
        "sigma_sector_map": {int(k): float(v) for k, v in sigma_sector_map.items()},
    }
    return windows, mcmc, samples, idata, summary
