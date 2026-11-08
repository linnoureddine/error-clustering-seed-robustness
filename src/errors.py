# Substitution errors with a fixed total count.
# m = errors per cluster (m=1 is uniform random), rho = error density inside a cluster.
from __future__ import annotations

import math
import numpy as np


def cluster_starts(L: int, C: int, W: int, rng: np.random.Generator) -> np.ndarray:
    # C windows of length W placed at random without overlapping:
    # pick C sorted points in the leftover space, then shift the i-th one by i*W
    free = L - C * W
    if free < 0:
        raise ValueError("clusters don't fit; lower the rate or raise rho")
    return np.sort(rng.integers(0, free + 1, size=C)) + np.arange(C) * W


def cluster_positions(L: int, E: int, m: int, rho: float, rng: np.random.Generator,
                      return_windows: bool = False):
    if E < 0 or E > L:
        raise ValueError("need 0 <= E <= L")
    if m < 1:
        raise ValueError("m >= 1")
    if E == 0 or m == 1:
        pos = np.sort(rng.choice(L, size=E, replace=False))
        return (pos, []) if return_windows else pos
    W = max(m, math.ceil(m / rho))  # cluster window length
    C = math.ceil(E / m)
    sizes = np.full(C, m)
    sizes[-1] = E - m * (C - 1)  # one cluster takes the remainder
    sizes = rng.permutation(sizes)
    pos, windows = [], []
    for a, size in zip(cluster_starts(L, C, W, rng), sizes):
        pos.append(a + rng.choice(W, size=size, replace=False))
        windows.append((int(a), int(a) + W, int(size)))
    pos = np.sort(np.concatenate(pos))
    assert len(pos) == E and len(np.unique(pos)) == E
    return (pos, windows) if return_windows else pos


def substitute(enc: np.ndarray, positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # replace each base with one of the other three
    out = enc.copy()
    shift = rng.integers(1, 4, size=len(positions)).astype(np.uint8)
    out[positions] = (out[positions] + shift) % 4
    return out


def clustering_stats(positions: np.ndarray, L: int, bin_size: int = 100) -> dict:
    # to check how clustered the errors actually are
    pos = np.sort(np.asarray(positions))
    nb = L // bin_size
    counts = np.bincount(pos[pos < nb * bin_size] // bin_size, minlength=nb)
    fano = counts.var() / counts.mean() if counts.mean() > 0 else float("nan")
    gaps = np.diff(pos)
    cv = gaps.std() / gaps.mean() if len(gaps) > 1 and gaps.mean() > 0 else float("nan")
    if len(pos) > 1:
        d = np.full(len(pos), np.inf)
        d[1:] = np.minimum(d[1:], gaps)
        d[:-1] = np.minimum(d[:-1], gaps)
        nn = float(np.median(d))
    else:
        nn = float("nan")
    return {"n_errors": len(pos), "fano_100": float(fano), "gap_cv": float(cv),
            "median_nn_dist": nn,
            "max_error_free_run": int(np.diff(np.r_[-1, pos, L]).max() - 1)}
