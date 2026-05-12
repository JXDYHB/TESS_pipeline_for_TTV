"""Mode 0 — quadratic baseline + white-noise likelihood.

All transit-shape parameters (r, b, duration, q1, q2) are shared across
every window.  Each window contributes only its own flux offset c0_k and
transit-time offset t0_k.  The baseline is a second-order polynomial:

    baseline_k(t) = c0_k * (1 + c2 * (t_rel - t0_k)²)

This is the fastest mode and serves as a sanity-check reference.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit

from tesspipe.modes.noise import build_noise_context, white_noise_sigma

# TransitOrbit requires a period, but only duration controls the transit shape
# when using the duration parameterisation — so a dummy value is fine.
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
    c2        = numpyro.sample("c2",        dist.Normal(0.0, 0.1))

    return {
        "duration":  duration,
        "r":         r,
        "b":         b,
        "u":         jnp.array([u1, u2]),
        "sigma_jit": sigma_jit,
        "c2":        c2,
        **build_noise_context(base_ctx),
    }


def run_window(k: int, w: dict, t_rel, f, t0, ctx: dict) -> None:
    """Evaluate the quadratic-baseline likelihood for one transit window."""
    orbit = TransitOrbit(
        period=float(_DUMMY_PERIOD),
        duration=ctx["duration"],
        time_transit=t0,
        impact_param=ctx["b"],
        radius_ratio=ctx["r"],
    )
    delta    = jnp.ravel(limb_dark_light_curve(orbit, ctx["u"])(t_rel))
    sigma    = white_noise_sigma(w, ctx, ctx["sigma_jit"])
    c0_k     = numpyro.sample(f"c0_{k}", dist.Normal(1.0, 0.005))
    baseline = c0_k * (1.0 + ctx["c2"] * (t_rel - t0) ** 2)
    numpyro.sample(f"obs_{k}", dist.Normal(baseline + delta, sigma), obs=f)
