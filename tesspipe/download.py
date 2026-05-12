"""TESS light-curve downloader with per-author fallback and retry logic."""

import os
import random
import shutil
import time

import numpy as np
import lightkurve as lk

from tesspipe.env_utils import env_int, env_float

# Preferred pipeline authors tried in order before falling back to others.
AUTHOR_PRIORITY = ["SPOC", "QLP"]


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _is_network_error(e: Exception) -> bool:
    """Return True if the exception looks like a transient network failure."""
    msg = repr(e)
    return any(tag in msg for tag in (
        "RemoteDisconnected", "Connection aborted", "ReadTimeout",
        "ConnectTimeout", "SSLError", "MaxRetryError",
        "ConnectionError", "TimeoutError",
    ))


def _is_corrupt_file_error(e: Exception) -> bool:
    """Return True if the exception indicates a truncated or corrupt cache file."""
    msg = repr(e).lower()
    return any(tag in msg for tag in (
        "file may be corrupt due to an interrupted download",
        "file may have been truncated",
        "truncated",
    ))


# ---------------------------------------------------------------------------
# Download with retry
# ---------------------------------------------------------------------------

def _download_with_retry(
    row,
    tic_id: int,
    sec: int,
    exptime: float,
    author: str,
    max_tries: int,
    base_sleep: float,
    jitter: float,
    per_sector_sleep,
    verbose: bool,
):
    """Download one search row, retrying on network or corrupt-cache errors."""
    max_corrupt_retries = env_int("DOWNLOAD_CORRUPT_RETRIES", default=2, min_value=1)
    max_backoff_s       = env_float("DOWNLOAD_MAX_BACKOFF_S", default=20.0, min_value=0.0)

    last_err = None
    corrupt_hits = 0
    unparsed_corrupt_hits = 0

    for i in range(max_tries):
        try:
            if verbose:
                print(
                    f"[download_sector] TIC {tic_id} sector {int(sec)} "
                    f"exptime={float(exptime):.0f}s author={author} "
                    f"(try {i + 1}/{max_tries})"
                )
            lc = row.download(flux_column="pdcsap_flux").remove_nans().normalize()

            if per_sector_sleep is not None:
                lo, hi = per_sector_sleep
                time.sleep(lo + (hi - lo) * random.random())
            return lc

        except Exception as e:
            last_err = e

            if _is_corrupt_file_error(e):
                corrupt_hits += 1
                path = None
                token = "Data product "
                if token in str(e):
                    path = str(e).split(token, 1)[1].split(" of type", 1)[0].strip()

                if path and os.path.exists(path):
                    bad_dir = os.path.dirname(path)
                    if verbose:
                        print(f"  [corrupt cache] removing {bad_dir}")
                    try:
                        shutil.rmtree(bad_dir)
                    except Exception:
                        try:
                            os.remove(path)
                        except Exception as ee:
                            if verbose:
                                print(f"  [corrupt cache] failed to remove: {ee}")
                else:
                    unparsed_corrupt_hits += 1
                    if verbose:
                        print("  [corrupt cache] detected but could not parse local path; retrying.")
                    if unparsed_corrupt_hits >= 2:
                        raise

                if corrupt_hits >= max_corrupt_retries:
                    raise
                time.sleep(1.0 + random.random())
                continue

            if _is_network_error(e):
                sleep_s = min(max_backoff_s, base_sleep * (2 ** i) * (1.0 + jitter * random.random()))
                if verbose:
                    print(f"  [network] {type(e).__name__}: {e}")
                    print(f"  -> sleep {sleep_s:.1f}s then retry...")
                time.sleep(sleep_s)
                continue

            raise

    raise last_err


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_sector(
    tic_id: int,
    author: str = "AUTO",
    sectors=None,
    target_exptime=None,
    max_tries: int = 3,
    base_sleep: float = 2.0,
    jitter: float = 0.35,
    per_sector_sleep=(0.6, 1.3),
    continue_on_fail: bool = True,
    verbose: bool = True,
) -> dict:
    """Download TESS light curves for one TIC target, organised by sector.

    Parameters
    ----------
    tic_id:
        TESS Input Catalogue identifier.
    author:
        ``"AUTO"`` tries SPOC then QLP with per-sector fallback.
        A specific name (e.g. ``"SPOC"``) uses only that pipeline.
    sectors:
        List of sector numbers to download.  ``None`` downloads all available.
    target_exptime:
        If given, keep only products whose exposure time matches this value
        (in seconds).  ``None`` picks the shortest cadence per sector.
    continue_on_fail:
        If ``True``, skip failed sectors instead of raising.

    Returns
    -------
    dict
        ``{(sector, exptime): LightCurve}``
    """
    data       = {}
    author_str = str(author).upper() if author is not None else "AUTO"
    max_tries              = env_int("DOWNLOAD_MAX_TRIES", default=max_tries, min_value=1)
    max_authors_per_sector = env_int("DOWNLOAD_MAX_AUTHORS_PER_SECTOR", default=4, min_value=1)

    # maps[author][sector] = (exptime, search_row)
    maps: dict[str, dict] = {}

    if author_str == "AUTO":
        try:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
            tbl    = search.table
        except Exception as e:
            if verbose:
                print(f"[download_sector] search failed for TIC {tic_id}: {e}")
            return data

        if len(tbl) == 0:
            if verbose:
                print(f"[download_sector] No products found for TIC {tic_id}.")
            return data

        if "author" not in tbl.colnames:
            if verbose:
                print("[download_sector] 'author' column missing; cannot prioritise by author.")
            return data

        all_authors = sorted({str(a).upper() for a in tbl["author"]})
        authors = [a for a in AUTHOR_PRIORITY if a in all_authors] + \
                  [a for a in all_authors if a not in AUTHOR_PRIORITY]

        for auth in authors:
            mask_auth = np.array([str(a).upper() == auth for a in tbl["author"]])
            if not np.any(mask_auth):
                maps[auth] = {}
                continue
            tbl_auth = tbl[mask_auth]
            m_auth   = {}
            for sec in np.unique(tbl_auth["sequence_number"]):
                sub = tbl_auth[tbl_auth["sequence_number"] == sec]
                if target_exptime is not None:
                    mask_exp = np.isclose(sub["exptime"], float(target_exptime))
                    if not np.any(mask_exp):
                        continue
                    exptime = float(sub["exptime"][mask_exp][0])
                else:
                    exptime = float(np.min(np.unique(sub["exptime"])))
                mask_pick = mask_auth & (tbl["sequence_number"] == sec) & np.isclose(tbl["exptime"], exptime)
                if np.any(mask_pick):
                    m_auth[int(sec)] = (float(exptime), search[mask_pick][0])
            maps[auth] = m_auth
    else:
        authors = [author_str]
        try:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author=author_str)
            tbl    = search.table
        except Exception as e:
            if verbose:
                print(f"[download_sector] search failed for author={author_str}: {e}")
            return data

        m_auth = {}
        for sec in np.unique(tbl["sequence_number"]):
            sub = tbl[tbl["sequence_number"] == sec]
            if target_exptime is not None:
                mask_exp = np.isclose(sub["exptime"], float(target_exptime))
                if not np.any(mask_exp):
                    continue
                exptime = float(sub["exptime"][mask_exp][0])
            else:
                exptime = float(np.min(np.unique(sub["exptime"])))
            mask_pick = (tbl["sequence_number"] == sec) & np.isclose(tbl["exptime"], exptime)
            if np.any(mask_pick):
                m_auth[int(sec)] = (float(exptime), search[mask_pick][0])
        maps[author_str] = m_auth

    # Collect all sectors across authors, then apply the sector filter.
    all_sectors = sorted({sec for auth in authors for sec in maps.get(auth, {})})
    if sectors is not None:
        requested   = {int(s) for s in sectors}
        all_sectors = [s for s in all_sectors if s in requested]
        if verbose:
            print(f"[download_sector] sector filter active: {sorted(requested)}")

    if not all_sectors:
        if verbose:
            print(f"[download_sector] No products found for TIC {tic_id} (sectors={sectors}, authors={authors}).")
        return data

    for sec in all_sectors:
        got       = False
        last_err  = None
        attempted = 0

        for auth in authors:
            m_auth = maps.get(auth, {})
            if sec not in m_auth:
                continue
            if attempted >= max_authors_per_sector:
                if verbose:
                    print(f"[download_sector] sector={sec} reached author-attempt limit; skipping rest.")
                break
            exptime, row = m_auth[sec]
            attempted   += 1
            try:
                lc = _download_with_retry(
                    row=row, tic_id=tic_id, sec=sec, exptime=exptime, author=auth,
                    max_tries=max_tries, base_sleep=base_sleep, jitter=jitter,
                    per_sector_sleep=per_sector_sleep, verbose=verbose,
                )
                data[(int(sec), float(exptime))] = lc
                got = True
                break
            except Exception as e:
                last_err = e
                if verbose:
                    print(f"[download_sector] FAILED sector={sec} author={auth}: {e}")

        if not got and not continue_on_fail:
            raise last_err

    if verbose:
        print(f"[download_sector] Done. Downloaded {len(data)} sector(s) for TIC {tic_id}.")
    return data
