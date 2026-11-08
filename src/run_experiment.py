# Main experiment: insert errors into each region and measure which seeds survive.
# Usage: python run_experiment.py                  (full run: 100 regions, 10 reps)
#        python run_experiment.py --regions 2 --reps 2   (quick test)
import argparse
import csv
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

import seeding as S
import errors as ER
import metrics as M
from common import load_genome, sample_regions, write_regions, RESULTS

L = 10_000
SEED = 2027

# (rate, m, rho); m = 1 is uniform errors, so it isn't repeated for rho = 0.5
CONDITIONS = [(rate, m, 0.25) for rate in (0.01, 0.05, 0.10) for m in (1, 2, 4, 8, 16)]
CONDITIONS += [(0.05, m, 0.5) for m in (2, 4, 8, 16)]


def method_list(config):
    methods = [("dense", "kmers", config["references"]["kmers"]),
               ("dense", "randstrobes", config["references"]["randstrobes"])]
    for density, group in (("0.10", "sparse10"), ("0.20", "sparse20")):
        for name in ("minimizers", "syncmers", "syncmer_randstrobes"):
            methods.append((group, name, config["sparse"][density][name]["params"]))
    # strobe-window sensitivity, same s as the 10% sparse randstrobes
    s10 = config["sparse"]["0.10"]["syncmer_randstrobes"]["params"]["s"]
    methods.append(("window", "syncmer_randstrobes", {"l": 15, "s": s10, "wmin": 1, "wmax": 3}))
    methods.append(("window", "syncmer_randstrobes", {"l": 15, "s": s10, "wmin": 4, "wmax": 8}))
    return methods


_genome = None
_methods = None


def _init(methods):
    global _genome, _methods
    _genome = load_genome()[1]
    _methods = methods


def run_region(args):
    region_id, start, end, n_reps, out_dir = args
    enc = S.encode(_genome[start:end])
    orig = [S.extract(enc, name, **p) for _, name, p in _methods]

    rows, err_rows = [], []
    for cond_id, (rate, m, rho) in enumerate(CONDITIONS):
        E = round(rate * L)
        for rep in range(n_reps):
            rng = np.random.default_rng([SEED, region_id, cond_id, rep])
            pos = ER.cluster_positions(L, E, m, rho, rng)
            mut = ER.substitute(enc, pos, rng)
            err_rows.append({"region_id": region_id, "rate": rate, "m": m, "rho": rho, "rep": rep,
                             **ER.clustering_stats(pos, L)})
            for (group, name, p), o in zip(_methods, orig):
                res = M.compare(o, S.extract(mut, name, **p), L)
                rows.append({"region_id": region_id, "rate": rate, "m": m, "rho": rho, "rep": rep,
                             "group": group, "method": name, "params": json.dumps(p, sort_keys=True),
                             **res})

    # write the error stats first, then the results file last: the results file is the "done" marker
    pd.DataFrame(err_rows).to_csv(out_dir / f"errors_{region_id:03d}.csv", index=False)
    tmp = out_dir / f"region_{region_id:03d}.csv.tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(out_dir / f"region_{region_id:03d}.csv")
    return region_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", type=int, default=100)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="raw", help="output folder inside results/")
    args = ap.parse_args()

    acc, genome = load_genome()
    with open(RESULTS / "frozen_config.json") as fh:
        config = json.load(fh)
    methods = method_list(config)

    with open(RESULTS / "calibration_regions.csv") as fh:
        cal = [(int(r["start0"]), int(r["end0"])) for r in csv.DictReader(fh)]
    # always draw the full 100 so a test run uses the same first regions as the real run
    regions = sample_regions(len(genome), 100, L, SEED, exclude=cal)[:args.regions]
    write_regions(RESULTS / "experiment_regions.csv",
                  sample_regions(len(genome), 100, L, SEED, exclude=cal), acc, "experiment")

    out_dir = RESULTS / args.out / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(i, a, b, args.reps, out_dir) for i, (a, b) in enumerate(regions)
            if not (out_dir / f"region_{i:03d}.csv").exists()]
    print(f"{len(regions)} regions, {len(CONDITIONS)} conditions, {args.reps} reps, "
          f"{len(methods)} methods; {len(regions) - len(todo)} regions already done")

    if args.workers > 1:
        with Pool(args.workers, initializer=_init, initargs=(methods,)) as pool:
            for k, rid in enumerate(pool.imap_unordered(run_region, todo), 1):
                print(f"  region {rid:3d} done ({k}/{len(todo)})", flush=True)
    else:
        _init(methods)
        for k, t in enumerate(todo, 1):
            print(f"  region {run_region(t):3d} done ({k}/{len(todo)})", flush=True)

    # combine the parts
    ids = range(len(regions))
    res = pd.concat([pd.read_csv(out_dir / f"region_{i:03d}.csv") for i in ids], ignore_index=True)
    err = pd.concat([pd.read_csv(out_dir / f"errors_{i:03d}.csv") for i in ids], ignore_index=True)
    res.to_csv(RESULTS / args.out / "experiment_raw.csv.gz", index=False)
    err.to_csv(RESULTS / args.out / "error_stats.csv.gz", index=False)
    print(f"saved {len(res)} result rows and {len(err)} error-pattern rows to results/{args.out}/")


if __name__ == "__main__":
    main()
