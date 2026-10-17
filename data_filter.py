#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared data-integrity filter for the Tebon primary cohort.

Single source of truth (added 2026-07-16). A handful of source tracks in the Tebon
MOESM6 xlsx (concentrated in Lapatinib 10uM) are corrupted: their reported mass
equals the timestamp (mass=24 at 24h, mass=72 at 72h) or spikes to hundreds of times
the 6h baseline -- physically impossible for a normalized dry-mass ratio, and absent
from the same conditions in the independent 4,066-organoid export. We drop any
organoid whose normalized mass ratio exceeds 5x at any horizon (extremely lenient;
5-fold growth). This removes 38 tracks: 1888 -> 1850.

Every script that parses the xlsx must apply this so all analyses agree. `recs`
tuples vary in arity across scripts, but the trajectory (t, m) arrays are always the
LAST TWO elements -- filter_recs relies only on that.
"""
import numpy as np

RATIO_CEILING = 5.0


def ratio_max(t, m):
    ok = np.isfinite(m)
    if ok.sum() < 2:
        return np.inf
    base = np.interp(6, t[ok], m[ok])
    if base <= 1e-3:
        return np.inf
    return float(np.nanmax(m[ok]) / base)


def filter_recs(recs, verbose=True):
    """Drop corrupted organoids. Assumes each rec has (t, m) as its last two items."""
    n0 = len(recs)
    kept = [r for r in recs if ratio_max(r[-2], r[-1]) <= RATIO_CEILING]
    if verbose:
        print(f"data-integrity filter: dropped {n0 - len(kept)} corrupted tracks (ratio>{RATIO_CEILING}x); kept {len(kept)}")
    return kept
