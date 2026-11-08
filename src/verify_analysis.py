# Independent cross-check of analyze.py: recomputes every gap from the raw rows with t-intervals.
# Usage: python verify_analysis.py   (after run_experiment.py and analyze.py; needs scipy)
import json
import numpy as np, pandas as pd
from scipy import stats

from common import RESULTS
R = str(RESULTS)
raw = pd.read_csv(f"{R}/raw/experiment_raw.csv.gz")
rc = pd.read_csv(f"{R}/analysis/ranking_changes.csv")
cfg = json.load(open(f"{R}/frozen_config.json"))
ok = True

# 1. completeness
n_exp = 19 * 100 * 10 * 10
print("rows", len(raw), "expected", n_exp, "OK" if len(raw) == n_exp else "MISMATCH")
print("regions", raw.region_id.nunique(), "| reps per cell", raw.groupby(["region_id", "rate", "m", "rho", "group", "method", "params"]).size().unique())
print("NaNs in metrics:", int(raw[["seed_retention", "seq_coverage", "match_coverage"]].isna().sum().sum()))

# 2. recompute A-B at m=1 and m=16 from raw rows with a paired t-interval over regions
def label(g, m, p):
    p = json.loads(p)
    if g == "dense": return m
    return f"{g}:{m}" if g != "window" else f"window:{p['wmin']}-{p['wmax']}"
raw["lab"] = [label(*t) for t in zip(raw.group, raw.method, raw.params)]
names = {"kmers": "kmers", "randstrobes": "randstrobes", "min10": "sparse10:minimizers", "sync10": "sparse10:syncmers",
         "srs10": "sparse10:syncmer_randstrobes", "min20": "sparse20:minimizers", "sync20": "sparse20:syncmers",
         "srs20": "sparse20:syncmer_randstrobes", "srs10_w13": "window:1-3", "srs10_w48": "window:4-8"}

def per_region(metric, lab, rate, m, rho):
    d = raw[(raw.lab == lab) & (raw.rate == rate) & (raw.m == m) & (raw.rho == rho)]
    return d.groupby("region_id")[metric].mean().sort_index().to_numpy()

def tci(x):
    mu = x.mean(); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return mu, mu - h, mu + h

bad = 0
for _, r in rc.iterrows():
    a, b = names[r.A], names[r.B]
    d1 = per_region(r.metric, a, r.rate, 1, 0.25) - per_region(r.metric, b, r.rate, 1, 0.25)
    d16 = per_region(r.metric, a, r.rate, 16, r.rho_clustered) - per_region(r.metric, b, r.rate, 16, r.rho_clustered)
    for name, x, est, lo, hi in (("m1", d1, r.diff_m1, r.lo_m1, r.hi_m1), ("m16", d16, r.diff_m16, r.lo_m16, r.hi_m16),
                                 ("change", d16 - d1, r.change, r.lo_change, r.hi_change)):
        mu, tlo, thi = tci(x)
        if abs(mu - est) > 1e-9 * max(1, abs(est)):
            bad += 1; print("ESTIMATE MISMATCH", r.A, r.B, r.metric, r.rate, name, mu, est)
        # bootstrap vs t: significance should agree except right at the boundary
        if (lo > 0 or hi < 0) != (tlo > 0 or thi < 0):
            print(f"  borderline significance ({name}): {r.A}-{r.B} {r.metric} rate {r.rate} rho {r.rho_clustered}: "
                  f"boot [{lo:.4g},{hi:.4g}] t [{tlo:.4g},{thi:.4g}]")
print("estimate mismatches:", bad, "of", 3 * len(rc))
