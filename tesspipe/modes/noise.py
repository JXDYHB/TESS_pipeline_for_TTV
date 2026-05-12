"""Hierarchical white-noise model shared across all window modes."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# (name, scales_attr, n_attr, condition_for_sampling)
# condition: sector always sampled; author/cadence only when there are >1 groups,
# because a single group provides no data to constrain its own extra variance.
_NOISE_GROUPS = [
    ("sigma_sector_extra",  "sector_white_prior_scales",  "n_sectors",  lambda n: n > 0),
    ("sigma_author_extra",  "author_white_prior_scales",  "n_authors",  lambda n: n > 1),
    ("sigma_cadence_extra", "cadence_white_prior_scales", "n_cadences", lambda n: n > 1),
]


def build_noise_context(base_ctx) -> dict:
    """Sample hierarchical white-noise extra-variance terms shared across windows.

    Returns a dict with three entries (sigma_sector_extra, sigma_author_extra,
    sigma_cadence_extra), each a JAX array indexed by the corresponding group index.
    When a group has only one member its term is fixed to zero.

    ``base_ctx`` is a ``ModelContext`` instance (see model_build.py).
    """
    result = {}
    for name, scales_attr, n_attr, should_sample in _NOISE_GROUPS:
        scales = jnp.asarray(getattr(base_ctx, scales_attr))
        n      = int(getattr(base_ctx, n_attr))
        if should_sample(n):
            result[name] = numpyro.sample(name, dist.HalfNormal(scales).to_event(1))
        else:
            result[name] = jnp.zeros((n,), dtype=jnp.float32)
    return result


def white_noise_sigma(w: dict, ctx: dict, sigma_jit) -> jnp.ndarray:
    """Build the per-point white-noise scale for one window.

    Combines the catalogue flux errors with the jitter term and any
    hierarchical extra-variance terms for the window's sector, author,
    and cadence group.

    ``ctx`` is the merged mode context dict (static config + sampled values).
    """
    f_err        = jnp.ravel(w.get("f_err", jnp.asarray([], dtype=jnp.float32)))
    default_ferr = jnp.asarray(float(ctx["default_flux_err"]))
    f_err        = jnp.where(jnp.isfinite(f_err) & (f_err > 0), f_err, default_ferr)
    var          = f_err ** 2 + jnp.asarray(sigma_jit) ** 2

    for extra_key, idx_key, n_key in [
        ("sigma_sector_extra",  "sector_index",  "n_sectors"),
        ("sigma_author_extra",  "author_index",  "n_authors"),
        ("sigma_cadence_extra", "cadence_index", "n_cadences"),
    ]:
        idx = int(w.get(idx_key, -1))
        if 0 <= idx < int(ctx[n_key]):
            var = var + ctx[extra_key][idx] ** 2

    return jnp.sqrt(jnp.maximum(var, 1e-12))
