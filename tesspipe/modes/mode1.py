"""Mode 1 — tinygp Matérn-3/2 baseline, independent GP hyperparameters per window.

All transit-shape parameters (r, b, duration, q1, q2) are shared across
every window.  Each window has its own GP amplitude gp_amp_k and length-scale
gp_ell_k, allowing the systematics to look different in different sectors or
at different times.

The GP is conditioned on the residuals after subtracting the transit model,
so the sampler jointly infers the transit shape and the correlated noise.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit

from tesspipe.modes.noise import build_noise_context, white_noise_sigma

_DUMMY_PERIOD = 10.0


def build_context(base_ctx) -> dict:
    """Sample all parameters that are shared across windows.

    ``base_ctx`` is a ``ModelContext`` instance.
    """
    if base_ctx.duration_prior_days is not None:
        duration = numpyro.sample(
            "duration",
            dist.TruncatedNormal(
                low=base_ctx.duration_prior_low,
                high=base_ctx.duration_prior_high,
                loc=base_ctx.duration_prior_days,
                scale=base_ctx.duration_prior_sigma_days,
            ),
        )
    else:
        duration = numpyro.sample(
            "duration", dist.TruncatedNormal(low=0.03, high=0.8, loc=0.20, scale=0.12)
        )
    numpyro.deterministic("logD", jnp.log(duration))

    logR = numpyro.sample("logR", dist.Uniform(jnp.log(base_ctx.r_lo), jnp.log(base_ctx.r_hi)))
    r    = numpyro.deterministic("r", jnp.exp(logR))
    b    = numpyro.sample("b",  dist.Uniform(0.0, 1.1))
    q1   = numpyro.sample("q1", dist.Uniform(0.0, 1.0))
    q2   = numpyro.sample("q2", dist.Uniform(0.0, 1.0))
    u1, u2 = base_ctx.q_to_u(q1, q2)

    sigma_jit = numpyro.sample("sigma_jit", dist.HalfNormal(0.01))

    return {
        "duration":  duration,
        "r":         r,
        "b":         b,
        "u":         jnp.array([u1, u2]),
        "sigma_jit": sigma_jit,
        **build_noise_context(base_ctx),
    }


def run_window(k: int, w: dict, t_rel, f, t0, ctx: dict) -> None:
    """Sample per-window GP hyperparameters and evaluate the GP log-likelihood."""
    orbit = TransitOrbit(
        period=float(_DUMMY_PERIOD),
        duration=ctx["duration"],
        time_transit=t0,
        impact_param=ctx["b"],
        radius_ratio=ctx["r"],
    )
    delta   = jnp.ravel(limb_dark_light_curve(orbit, ctx["u"])(t_rel))
    sigma   = white_noise_sigma(w, ctx, ctx["sigma_jit"])
    c0_k    = numpyro.sample(f"c0_{k}", dist.Normal(1.0, 0.0005))

    cadence = jnp.maximum(jnp.asarray(float(w.get("t_cadence", 1e-4))), 1e-6)

    # GP length-scale prior: stay BELOW the transit duration so the GP only
    # absorbs short-time correlated noise and cannot mimic a transit-shaped
    # bump.  Centring the prior at duration (as in the previous version) put
    # the GP in the same frequency band as the transit and produced an
    # anti-correlated "GP up + transit deeper" solution for sparse targets.
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
        # Tighter scale (0.00015 vs 0.0003) keeps gp_amp near 0 unless data
        # genuinely demands correlated structure — prevents the GP from
        # quietly absorbing transit-shape residuals.
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
