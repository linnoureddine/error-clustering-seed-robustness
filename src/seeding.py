# Seeding methods: all k-mers, minimizers, open syncmers, randstrobes,
# and syncmer-based randstrobes (like strobealign).
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

_ENC = np.full(256, 255, dtype=np.uint8)
for _c, _v in zip(b"ACGT", range(4)):
    _ENC[_c] = _v
    _ENC[_c + 32] = _v  # lowercase

SALT = np.uint64(0x9E3779B97F4A7C15)
U64 = np.uint64


def encode(seq: str) -> np.ndarray:
    # A,C,G,T -> 0,1,2,3
    arr = _ENC[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    if (arr == 255).any():
        raise ValueError("sequence contains non-ACGT characters")
    return arr


def splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (x ^ SALT).astype(np.uint64)
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
        z = z ^ (z >> U64(31))
    return z


def kmer_codes(enc: np.ndarray, k: int) -> np.ndarray:
    # each k-mer packed into one integer, 2 bits per base (k <= 32)
    if not 1 <= k <= 32:
        raise ValueError("k must be in 1..32")
    n = len(enc) - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    codes = np.zeros(n, dtype=np.uint64)
    e = enc.astype(np.uint64)
    for j in range(k):
        codes = (codes << U64(2)) | e[j:j + n]
    return codes


def kmer_hashes(enc: np.ndarray, k: int) -> np.ndarray:
    return splitmix64(kmer_codes(enc, k))


@dataclass
class SeedSet:
    starts: np.ndarray    # start of each seed
    comps: np.ndarray     # start of each part (1 column for k-mers, 2 for strobemers)
    comp_len: int         # length of each part
    ident: np.ndarray     # hash of the seed's content
    n_eligible: int       # positions where a seed could start

    @property
    def density(self) -> float:
        return len(self.starts) / self.n_eligible if self.n_eligible else float("nan")

    def keys(self) -> np.ndarray:
        # used to match seeds between the original and mutated sequence
        return np.column_stack([self.comps.astype(np.uint64), self.ident])


def all_kmers(enc: np.ndarray, k: int) -> SeedSet:
    h = kmer_hashes(enc, k)
    n = len(h)
    pos = np.arange(n)
    return SeedSet(pos, pos[:, None], k, h, n)


def minimizers(enc: np.ndarray, k: int, w: int) -> SeedSet:
    # smallest hash in each window of w k-mers (argmin picks the leftmost on ties)
    h = kmer_hashes(enc, k)
    n = len(h)
    if n < w:
        return SeedSet(np.zeros(0, int), np.zeros((0, 1), int), k, np.zeros(0, np.uint64), n)
    win = np.lib.stride_tricks.sliding_window_view(h, w)
    sel = np.unique(np.argmin(win, axis=1) + np.arange(n - w + 1))
    return SeedSet(sel, sel[:, None], k, h[sel], n)


def open_syncmer_mask(enc: np.ndarray, k: int, s: int, t: int | None = None) -> np.ndarray:
    # k-mer is kept if its smallest s-mer is at offset t (default: middle)
    if t is None:
        t = (k - s) // 2
    if not (1 <= s <= k and 0 <= t <= k - s):
        raise ValueError("need 1 <= s <= k and 0 <= t <= k-s")
    sh = kmer_hashes(enc, s)
    nk = len(enc) - k + 1
    if nk <= 0:
        return np.zeros(0, bool)
    win = np.lib.stride_tricks.sliding_window_view(sh, k - s + 1)[:nk]
    return np.argmin(win, axis=1) == t


def open_syncmers(enc: np.ndarray, k: int, s: int, t: int | None = None) -> SeedSet:
    mask = open_syncmer_mask(enc, k, s, t)
    h = kmer_hashes(enc, k)
    sel = np.flatnonzero(mask)
    return SeedSet(sel, sel[:, None], k, h[sel], len(h))


def _link_xv(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    # link function from Karami et al. 2024
    return splitmix64(x0 ^ x1)


def _final_hash(h0: np.ndarray, h1: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return (U64(2) * h0 - h1).astype(np.uint64)


def randstrobes(enc: np.ndarray, l: int, wmin: int, wmax: int) -> SeedSet:
    # one seed per position i; second strobe chosen in [i+wmin, i+wmax]
    codes = kmer_codes(enc, l)
    hs = splitmix64(codes)
    n_l = len(codes)
    n_seed = n_l - wmax
    if n_seed <= 0:
        return SeedSet(np.zeros(0, int), np.zeros((0, 2), int), l, np.zeros(0, np.uint64), 0)
    i = np.arange(n_seed)
    W = wmax - wmin + 1
    cand = i[:, None] + wmin + np.arange(W)[None, :]
    link = _link_xv(codes[i][:, None], codes[cand])
    j = cand[np.arange(n_seed), np.argmin(link, axis=1)]
    ident = _final_hash(hs[i], hs[j])
    return SeedSet(i, np.column_stack([i, j]), l, ident, n_seed)


def syncmer_randstrobes(enc: np.ndarray, l: int, s: int, wmin: int, wmax: int,
                        t: int | None = None) -> SeedSet:
    # both strobes are syncmers; the second is picked wmin..wmax syncmers later
    mask = open_syncmer_mask(enc, l, s, t)
    codes = kmer_codes(enc, l)
    hs = splitmix64(codes)
    sp = np.flatnonzero(mask)
    m = len(sp)
    firsts, seconds = [], []
    for q in range(m):
        lo, hi = q + wmin, min(q + wmax, m - 1)
        if lo > hi:
            continue
        cands = sp[lo:hi + 1]
        link = _link_xv(np.full(len(cands), codes[sp[q]], dtype=np.uint64), codes[cands])
        firsts.append(sp[q])
        seconds.append(cands[int(np.argmin(link))])
    f = np.asarray(firsts, dtype=int)
    g = np.asarray(seconds, dtype=int)
    ident = _final_hash(hs[f], hs[g]) if len(f) else np.zeros(0, np.uint64)
    return SeedSet(f, np.column_stack([f, g]) if len(f) else np.zeros((0, 2), int),
                   l, ident, len(codes))


def extract(enc: np.ndarray, method: str, **p) -> SeedSet:
    if method == "kmers":
        return all_kmers(enc, p["k"])
    if method == "minimizers":
        return minimizers(enc, p["k"], p["w"])
    if method == "syncmers":
        return open_syncmers(enc, p["k"], p["s"], p.get("t"))
    if method == "randstrobes":
        return randstrobes(enc, p["l"], p["wmin"], p["wmax"])
    if method == "syncmer_randstrobes":
        return syncmer_randstrobes(enc, p["l"], p["s"], p["wmin"], p["wmax"], p.get("t"))
    raise ValueError(f"unknown method {method}")
