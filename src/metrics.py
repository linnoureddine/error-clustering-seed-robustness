# Compare seeds of the original and mutated sequence.
# Metric definitions follow Sahlin 2021 and Edgar 2021.
from __future__ import annotations

import numpy as np
from seeding import SeedSet


def _rows_as_void(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a.astype(np.uint64))
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def matched_mask(orig: SeedSet, mut: SeedSet) -> np.ndarray:
    # a seed matches if the mutated sequence has it at the same positions with the same hash
    if len(orig.starts) == 0 or len(mut.starts) == 0:
        return np.zeros(len(orig.starts), bool)
    return np.isin(_rows_as_void(orig.keys()), _rows_as_void(mut.keys()))


def _union_cover(L: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    diff = np.zeros(L + 1, dtype=np.int64)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    return np.cumsum(diff[:L]) > 0


def _islands(covered: np.ndarray) -> np.ndarray:
    # lengths of the runs of uncovered positions
    x = np.r_[1, covered.astype(np.int8), 1]
    d = np.diff(x)
    starts = np.flatnonzero(d == -1)
    ends = np.flatnonzero(d == 1)
    return ends - starts


def compare(orig: SeedSet, mut: SeedSet, L: int, read_len: int = 150) -> dict:
    mm = matched_mask(orig, mut)
    comps = orig.comps[mm]
    cl = orig.comp_len
    n = len(orig.starts)
    if comps.size:
        seq_cov = _union_cover(L, comps.ravel(), comps.ravel() + cl)
        span_s, span_e = comps.min(axis=1), comps.max(axis=1) + cl
        match_cov = _union_cover(L, span_s, span_e)
        # window starting at x contains the seed if span_e - R <= x <= span_s
        lo = np.maximum(span_e - read_len, 0)
        hi = span_s + 1
        ok = lo < hi
        nwin = L - read_len + 1
        win = _union_cover(nwin, np.minimum(lo[ok], nwin), np.minimum(hi[ok], nwin))
        win_hit = win.mean()
    else:
        seq_cov = match_cov = np.zeros(L, bool)
        win_hit = 0.0
    isl = _islands(match_cov)
    return {
        "n_seeds_orig": n,
        "n_seeds_mut": len(mut.starts),
        "n_matched": int(mm.sum()),
        "seed_retention": mm.mean() if n else float("nan"),
        "seq_coverage": seq_cov.mean(),
        "match_coverage": match_cov.mean(),
        "island_esize": float((isl.astype(float) ** 2).sum() / L),  # Sahlin's E-size
        "max_island": int(isl.max()) if len(isl) else 0,
        f"window_hit_{read_len}": float(win_hit),  # secondary metric
    }
