# Tests: compare each method with a slow, straightforward version,
# and check the error model and metrics. Run: python tests.py
import numpy as np
import seeding as S
import errors as ER
import metrics as M

rng = np.random.default_rng(1)


def rand_seq(n):
    return "".join(rng.choice(list("ACGT"), n))


def h_str(x: str) -> int:
    code = 0
    for c in x:
        code = (code << 2) | "ACGT".index(c)
    return int(S.splitmix64(np.array([code], dtype=np.uint64))[0])


def code_str(x):
    c = 0
    for ch in x:
        c = (c << 2) | "ACGT".index(ch)
    return c


def brute_minimizers(seq, k, w):
    n = len(seq) - k + 1
    hs = [h_str(seq[i:i + k]) for i in range(n)]
    sel = set()
    for a in range(n - w + 1):
        win = hs[a:a + w]
        sel.update(a + j for j, v in enumerate(win) if v == min(win))
    return sorted(sel)


def brute_syncmers(seq, k, s, t):
    out = []
    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]
        sh = [h_str(km[j:j + s]) for j in range(k - s + 1)]
        if sh.index(min(sh)) == t:
            out.append(i)
    return out


def brute_randstrobes(seq, l, wmin, wmax):
    out = []
    n_l = len(seq) - l + 1
    for i in range(n_l - wmax):
        x0 = code_str(seq[i:i + l])
        best, bj = None, None
        for j in range(i + wmin, i + wmax + 1):
            v = int(S.splitmix64(np.array([x0 ^ code_str(seq[j:j + l])], dtype=np.uint64))[0])
            if best is None or v < best:
                best, bj = v, j
        out.append((i, bj))
    return out


def test_encoding_hash():
    e = S.encode("ACGTacgt")
    assert list(e) == [0, 1, 2, 3, 0, 1, 2, 3]
    try:
        S.encode("ACGN"); raise AssertionError
    except ValueError:
        pass
    seq = rand_seq(200)
    hs = S.kmer_hashes(S.encode(seq), 21)
    assert all(int(hs[i]) == h_str(seq[i:i + 21]) for i in range(0, 180, 7))


def test_minimizers():
    for _ in range(5):
        seq = rand_seq(400)
        ss = S.minimizers(S.encode(seq), 11, 7)
        assert list(ss.starts) == brute_minimizers(seq, 11, 7)
    # repeated k-mers: every tied minimum is kept
    seq = "A" * 60 + rand_seq(60)
    got = list(S.minimizers(S.encode(seq), 11, 7).starts)
    assert got == brute_minimizers(seq, 11, 7)
    assert all(p in got for p in range(50))  # all-A 11-mers tie in every window


def test_syncmers():
    for t in (0, 3, None):
        seq = rand_seq(500)
        tt = (15 - 6) // 2 if t is None else t
        assert list(S.open_syncmers(S.encode(seq), 15, 6, t).starts) == brute_syncmers(seq, 15, 6, tt)


def test_randstrobes():
    seq = rand_seq(300)
    rs = S.randstrobes(S.encode(seq), 10, 5, 15)
    assert [tuple(r) for r in rs.comps] == brute_randstrobes(seq, 10, 5, 15)


def test_syncmer_randstrobes():
    seq = rand_seq(800)
    enc = S.encode(seq)
    rs = S.syncmer_randstrobes(enc, 15, 8, 2, 5)
    sp = list(S.open_syncmers(enc, 15, 8).starts)
    assert set(rs.comps[:, 0]) <= set(sp) and set(rs.comps[:, 1]) <= set(sp)
    for f, g in rs.comps:
        q = sp.index(f); qg = sp.index(g)
        assert 2 <= qg - q <= 5


def test_context_free_syncmers():
    # a mutation outside a syncmer shouldn't change whether it's selected
    seq = S.encode(rand_seq(2000))
    a = set(S.open_syncmers(seq, 21, 11).starts)
    pos = np.array([1000])
    mut = ER.substitute(seq, pos, rng)
    b = set(S.open_syncmers(mut, 21, 11).starts)
    diff = a ^ b
    assert all(980 <= p <= 1000 for p in diff), diff


def test_error_model():
    L = 5000
    for m in (1, 2, 4, 8, 16):
        for seed in range(20):
            r = np.random.default_rng(seed)
            pos = ER.cluster_positions(L, 250, m, 0.25, r)
            assert len(pos) == 250 and len(np.unique(pos)) == 250
            assert pos.min() >= 0 and pos.max() < L
    # cluster windows never overlap and each one holds exactly its errors
    for m in (2, 4, 8, 16):
        for seed in range(20):
            pos, win = ER.cluster_positions(L, 500, m, 0.25, np.random.default_rng(seed), return_windows=True)
            win = sorted(win)
            assert all(b <= a2 for (_, b, _), (a2, _, _) in zip(win, win[1:]))
            assert win[0][0] >= 0 and win[-1][1] <= L
            for a, b, size in win:
                assert ((pos >= a) & (pos < b)).sum() == size
            assert sorted(s for _, _, s in win)[1:] == [m] * (len(win) - 1)
    # reproducibility
    a = ER.cluster_positions(L, 250, 8, 0.25, np.random.default_rng(7))
    b = ER.cluster_positions(L, 250, 8, 0.25, np.random.default_rng(7))
    assert (a == b).all()
    # substitutions change the base and nothing else
    enc = S.encode(rand_seq(L))
    mut = ER.substitute(enc, a, rng)
    assert (mut[a] != enc[a]).all()
    keep = np.ones(L, bool); keep[a] = False
    assert (mut[keep] == enc[keep]).all()
    # errors should get more clustered as m grows
    f = [np.mean([ER.clustering_stats(ER.cluster_positions(L, 250, m, 0.25, np.random.default_rng(s)), L)["fano_100"]
                  for s in range(20)]) for m in (1, 4, 16)]
    assert f[0] < f[1] < f[2], f


def test_metrics():
    seq = S.encode(rand_seq(3000))
    ss = S.all_kmers(seq, 30)
    r = M.compare(ss, ss, 3000)
    assert r["seed_retention"] == 1.0 and r["seq_coverage"] == 1.0
    # one substitution should remove exactly the 30 k-mers overlapping it
    mut = ER.substitute(seq, np.array([1500]), rng)
    r = M.compare(ss, S.all_kmers(mut, 30), 3000)
    assert r["n_matched"] == len(ss.starts) - 30
    assert r["max_island"] == 1
    assert abs(r["seq_coverage"] - r["match_coverage"]) < 1e-12


def test_strobemer_matching():
    # error in strobe 1 or 2 -> seed lost; error past the window -> seed kept
    l, wmin, wmax = 15, 25, 50
    seq = S.encode(rand_seq(3000))
    orig = S.randstrobes(seq, l, wmin, wmax)
    for _ in range(30):
        q = int(rng.integers(100, len(orig.starts) - 100))
        i, j = orig.comps[q]
        cases = {
            "in strobe 1": (i + int(rng.integers(0, l)), False),
            "in strobe 2": (j + int(rng.integers(0, l)), False),
            "outside window": (i + wmax + l + int(rng.integers(0, 50)), True),
        }
        for name, (pos, expect_match) in cases.items():
            mut = S.randstrobes(ER.substitute(seq, np.array([pos]), rng), l, wmin, wmax)
            got = bool(M.matched_mask(orig, mut)[q])
            assert got == expect_match, (name, q, i, j, pos)
    # error between the strobes: seed is kept only if the same second strobe is chosen
    for _ in range(30):
        q = int(rng.integers(100, len(orig.starts) - 100))
        i, j = orig.comps[q]
        gap = [p for p in range(i + l, j)]
        if not gap:
            continue
        pos = gap[int(rng.integers(0, len(gap)))]
        mut = S.randstrobes(ER.substitute(seq, np.array([pos]), rng), l, wmin, wmax)
        same_choice = tuple(mut.comps[q]) == (i, j)
        assert bool(M.matched_mask(orig, mut)[q]) == same_choice


def test_sparse_strobemer_matching():
    seq = S.encode(rand_seq(5000))
    orig = S.syncmer_randstrobes(seq, 15, 6, 2, 5)
    for _ in range(20):
        q = int(rng.integers(50, len(orig.starts) - 50))
        i, j = orig.comps[q]
        mut = S.syncmer_randstrobes(ER.substitute(seq, np.array([i + 3]), rng), 15, 6, 2, 5)
        assert not M.matched_mask(orig, mut)[q]
        mut = S.syncmer_randstrobes(ER.substitute(seq, np.array([j + 3]), rng), 15, 6, 2, 5)
        assert not M.matched_mask(orig, mut)[q]
    # no errors -> everything matches
    r = M.compare(orig, orig, 5000)
    assert r["seed_retention"] == 1.0
    assert r["seq_coverage"] <= r["match_coverage"] <= 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
