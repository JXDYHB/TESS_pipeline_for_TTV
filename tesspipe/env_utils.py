"""Typed helpers for reading environment variables with defaults and bounds.

Centralised here so download.py and transit_window.py share one definition
instead of maintaining two slightly-different copies.
"""

import os


def env_int(name: str, default: int, min_value: int = None) -> int:
    """Read an integer env-var, falling back to `default`.  Optionally clamp below."""
    val = os.getenv(name)
    if val is None:
        return int(default)
    try:
        out = int(val)
    except ValueError:
        return int(default)
    return out if min_value is None else max(int(min_value), out)


def env_float(name: str, default: float, min_value: float = None) -> float:
    """Read a float env-var, falling back to `default`.  Optionally clamp below."""
    val = os.getenv(name)
    if val is None:
        return float(default)
    try:
        out = float(val)
    except ValueError:
        return float(default)
    return out if min_value is None else max(float(min_value), out)


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean env-var (1/true/yes/y/on → True), falling back to `default`."""
    val = os.getenv(name)
    return bool(default) if val is None else val.strip().lower() in {"1", "true", "yes", "y", "on"}
