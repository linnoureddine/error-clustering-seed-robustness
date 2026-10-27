# Pick parameters for the sparse methods so they all have ~10% (and ~20%) density.
# Uses unmutated regions only. Every seed samples 30 bases (k=30, or 2 strobes of 15).
import json
import numpy as np
import pandas as pd

import seeding as S
from common import load_genome, sample_regions, write_regions, RESULTS

OUT = RESULTS
OUT.mkdir(exist_ok=True)

L = 10_000
N_CAL = 50
SEED_CAL = 2026
TARGETS = [0.10, 0.20]

acc, genome = load_genome()
cal = sample_regions(len(genome), N_CAL, L, SEED_CAL)
write_regions(OUT / "calibration_regions.csv", cal, acc, "calibration")
encs = [S.encode(genome[a:b]) for a, b in cal]

grid = []
for w in range(5, 31):
    grid.append(("minimizers", dict(k=30, w=w)))
for s in range(10, 30):
    grid.append(("syncmers", dict(k=30, s=s)))
for s in range(2, 15):
    for target in TARGETS:
        # window counted in syncmers; gives a strobe gap of ~25-50 bases like Sahlin's
        wmin = max(1, round(25 * target)); wmax = max(wmin + 1, round(50 * target))
        grid.append(("syncmer_randstrobes", dict(l=15, s=s, wmin=wmin, wmax=wmax, _target=target)))
grid.append(("kmers", dict(k=30)))
grid.append(("randstrobes", dict(l=15, wmin=25, wmax=50)))

rows = []
for method, p in grid:
    params = {k: v for k, v in p.items() if not k.startswith("_")}
    d, span, gaps = [], [], []
    for e in encs:
        ss = S.extract(e, method, **params)
        d.append(ss.density)
        span.append(float(np.mean(ss.comps.max(1) + ss.comp_len - ss.comps.min(1))) if len(ss.starts) else np.nan)
        gaps.append(int(np.diff(ss.starts).max()) if len(ss.starts) > 1 else L)  # largest gap between seeds
    rows.append(dict(method=method, params=json.dumps(params), target=p.get("_target", np.nan),
                     density_mean=np.mean(d), density_sd=np.std(d, ddof=1),
                     density_min=np.min(d), density_max=np.max(d),
                     mean_span_nt=np.nanmean(span), max_start_gap=int(np.max(gaps))))
df = pd.DataFrame(rows)
df.to_csv(OUT / "calibration_sweep.csv", index=False)

frozen = {"region_length": L, "sampled_nt_per_seed": 30, "references": {
    "kmers": {"k": 30}, "randstrobes": {"l": 15, "wmin": 25, "wmax": 50}}, "sparse": {}}
for target in TARGETS:
    pick = {}
    for method in ["minimizers", "syncmers", "syncmer_randstrobes"]:
        sub = df[df.method == method]
        if method == "syncmer_randstrobes":
            sub = sub[sub.target == target]
        best = sub.iloc[(sub.density_mean - target).abs().argmin()]
        pick[method] = {"params": json.loads(best.params),
                        "calibrated_density": round(float(best.density_mean), 5)}
    frozen["sparse"][f"{target:.2f}"] = pick
with open(OUT / "frozen_config.json", "w") as fh:
    json.dump(frozen, fh, indent=2)

pd.set_option("display.width", 200)
print(df[df.method.isin(["kmers", "randstrobes"])].to_string(index=False))
print(json.dumps(frozen, indent=2))
