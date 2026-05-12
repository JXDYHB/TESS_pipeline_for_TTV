"""Mode 3 — tinygp baseline + independent transit duration per window.

Same as mode 1 (per-window tinygp), but duration_k is sampled independently
for each window instead of being a single shared parameter.  The impact
parameter b and radius ratio r remain shared.

A changing duration across epochs is a signature of orbital precession.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit

from tesspipe.modes.noise import build_noise_context, white_noise_sigma

_DUMMY_PERIOD = 10.0


def build_context(base_ctx) -> dict:
    """Sample parameters shared across all windows (duration is NOT here).

    ``base_ctx`` is a ``ModelContext`` instance.
    """
    logR = numpyro.sample("logR", dist.Uniform(jnp.log(base_ctx.r_lo), jnp.log(base_ctx.r_hi)))
    r    = numpyro.deterministic("r", jnp.exp(logR))
    b    = numpyro.sample("b",  dist.Uniform(0.0, 1.1))
    q1   = numpyro.sample("q1", dist.Uniform(0.0, 1.0))
    q2   = numpyro.sample("q2", dist.Uniform(0.0, 1.0))
    u1, u2 = base_ctx.q_to_u(q1, q2)

    sigma_jit = numpyro.sample("sigma_jit", dist.HalfNormal(0.01))

    return {
        "r":         r,
        "b":         b,
        "u":         jnp.array([u1, u2]),
        "sigma_jit": sigma_jit,
        **build_noise_context(base_ctx),
    }


def run_window(k: int, w: dict, t_rel, f, t0, ctx: dict) -> None:
    """Sample per-window duration and evaluate the GP log-likelihood."""
    if ctx["duration_prior_days"] is not None:
        duration_k = numpyro.sample(
            f"duration_{k}",
            dist.TruncatedNormal(
                low=ctx["duration_prior_low"],
                high=ctx["duration_prior_high"],
                loc=ctx["duration_prior_days"],
                scale=ctx["duration_prior_sigma_days"],
            ),
        )
    else:
        duration_k = numpyro.sample(
            f"duration_{k}",
            dist.TruncatedNormal(low=0.03, high=0.8, loc=0.20, scale=0.12),
        )
    numpyro.deterministic(f"logD_{k}", jnp.log(duration_k))

    orbit = TransitOrbit(
        period=float(_DUMMY_PERIOD),
        duration=duration_k,
        time_transit=t0,
        impact_param=ctx["b"],
        radius_ratio=ctx["r"],
    )
    delta   = jnp.ravel(limb_dark_light_curve(orbit, ctx["u"])(t_rel))
    sigma   = white_noise_sigma(w, ctx, ctx["sigma_jit"])
    c0_k    = numpyro.sample(f"c0_{k}", dist.Normal(1.0, 0.0005))

    cadence = jnp.maximum(jnp.asarray(float(w.get("t_cadence", 1e-4))), 1e-6)

    # Same prior tweaks as mode 1: keep gp_ell BELOW transit duration and
    # tighten gp_amp scale so the GP absorbs only short-time correlated noise
    # and cannot mimic a transit-shaped bump.
    duration_ref = (
        float(ctx["duration_prior_days"])
        if ctx.get("duration_prior_days") is not None
        else float(ctx["window"]) / 3.0
    )
    duration_ref = jnp.asarray(max(1e-4, duration_ref))
    ell_loc      = jnp.maximum(2.0 * cadence, 0.3 * duration_ref)
    ell_high     = jnp.maximum(2.0 * cadence, 1.0 * duration_ref)

    amp_k = numpyro.sample(
        f"gp_amp_{k}",
        dist.TruncatedNormal(low=0.0, high=0.001, loc=0.0, scale=0.00015),
    )
    ell_k = numpyro.sample(
        f"gp_ell_{k}",
        dist.TruncatedNormal(
            low=2.0 * cadence, high=ell_high,
            loc=ell_loc, scale=ell_loc,
        ),
    )

    y_base = jnp.ravel(f - delta)
    kernel = amp_k * amp_k * ctx["quasisep"].Matern32(scale=ell_k)
    gp     = ctx["GaussianProcess"](kernel, t_rel, diag=sigma ** 2, mean=c0_k)
    numpyro.factor(f"gp_ll_{k}", gp.log_probability(y_base))
