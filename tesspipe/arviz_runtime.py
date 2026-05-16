"""File-level lock for ArviZ imports on shared SLURM filesystems.

Multiple array-job tasks that start simultaneously can race to write
ArviZ's warning-metadata cache.  Serialising the import with fcntl avoids
transient tmp-file collisions.  The lock is process-local; it does not
provide cross-node protection.
"""

from pathlib import Path

try:
    import fcntl
except Exception:          # Windows / non-POSIX
    fcntl = None

_ARVIZ_IMPORT_LOCK_FH = None


def _arviz_warning_dir() -> Path:
    """Return the directory ArviZ uses for cache / warning metadata."""
    try:
        from platformdirs import user_cache_dir
        return Path(user_cache_dir("arviz", "arviz"))
    except Exception:
        return Path.home() / ".cache" / "arviz"


def configure_arviz_runtime() -> None:
    """Create the ArviZ cache directory and acquire the import lock."""
    global _ARVIZ_IMPORT_LOCK_FH
    warning_dir = _arviz_warning_dir()
    warning_dir.mkdir(parents=True, exist_ok=True)

    if fcntl is None or _ARVIZ_IMPORT_LOCK_FH is not None:
        return

    lock_path = warning_dir / "import.lock"
    fh = open(lock_path, "a+", encoding="utf-8")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    _ARVIZ_IMPORT_LOCK_FH = fh


def release_arviz_runtime() -> None:
    """Release the import lock after ArviZ has been imported."""
    global _ARVIZ_IMPORT_LOCK_FH
    if fcntl is None or _ARVIZ_IMPORT_LOCK_FH is None:
        return

    fcntl.flock(_ARVIZ_IMPORT_LOCK_FH.fileno(), fcntl.LOCK_UN)
    _ARVIZ_IMPORT_LOCK_FH.close()
    _ARVIZ_IMPORT_LOCK_FH = None
