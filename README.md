# Error clustering and seed robustness

Code for a CMPS 355 (Bioinformatics Algorithms, AUB, Fall 2026) research project by Aya Daou and Lin Noureddine.

**Research question:** At a fixed overall substitution rate and comparable seed density, does clustering of errors change the relative performance of k-mers, minimizers, syncmers and strobemers, compared with the uniform errors used in existing benchmarks?

## Contents

| Path                    | What it is                                                                                                                                                                                         |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/seeding.py`        | Seeding methods: all k-mers, minimizers, open syncmers, randstrobes (Sahlin 2021), and syncmer-based randstrobes (following strobealign's construction)                                            |
| `src/errors.py`         | Substitution error model with an exact error count and tunable clustering (`m` errors per cluster, density `rho` inside a cluster; `m = 1` is uniform random)                                      |
| `src/metrics.py`        | Seed matching between an original and a mutated sequence, plus metrics: seed retention, sequence coverage, match coverage, island E-size, longest island, and the secondary 150-bp window hit rate |
| `src/common.py`         | Genome loading and reproducible, non-overlapping region sampling                                                                                                                                   |
| `src/calibrate.py`      | Density calibration on unmutated regions; writes `results/frozen_config.json`                                                                                                                      |
| `src/run_experiment.py` | Main experiment: 100 regions × 19 error conditions × 10 replicates; writes raw results to `results/raw/`                                                                                           |
| `src/analyze.py`        | Averages replicates per region, bootstrap confidence intervals, method gaps, ranking changes, sanity check and figures; writes `results/analysis/`                                                 |
| `src/tests.py`          | Correctness tests. Each method is checked against a literal re-implementation of its published definition, and the error model and metrics are checked too                                         |
| `data/`                 | _E. coli_ K-12 MG1655 reference genome (NCBI RefSeq GCF_000005845.2, NC_000913.3, 4,641,652 bp)                                                                                                    |
| `results/`              | Calibration outputs, experiment regions, and the analysis tables and figures                                                                                                                       |

## How to run

You need Python ≥ 3.9 with `numpy`, `pandas` and `matplotlib`. On Windows, use `python` instead of `python3`.

```bash
cd src
python3 tests.py
python3 calibrate.py
python3 run_experiment.py
python3 analyze.py
```

For a quick test of the whole pipeline: `python3 run_experiment.py --regions 2 --reps 2 --out test`, then `python3 analyze.py --raw test`.

Every random step uses a fixed seed, so reruns give identical numbers.

## Design decisions (fixed before any error experiments)

- **Same sampled content per seed.** Every seed samples 30 nucleotides: k = 30 for k-mer-based methods, and 2 strobes × 15 for strobemers, following Sahlin (2021).
- **Two comparison groups.**
  - _Dense_ (one seed per position): all 30-mers vs randstrobes (2, 15, 25, 50).
  - _Sparse_ (density-matched): minimizers, open syncmers and syncmer-based randstrobes, calibrated to 10% density (primary) and 20% density (secondary).
- **Calibration uses separate data.** Parameters are chosen on 50 unmutated 10 kb regions that do not overlap the experiment regions.
- **Deterministic hashing.** 2-bit encoding plus the splitmix64 mixer; ties go to the leftmost position.
- **Forward strand only.** A sequence is compared with a substituted copy of itself in the same orientation, so the reverse strand is not needed.
- **Randstrobe linking.** Strobes are linked with `h(x0 XOR x1)` and the minimum is chosen, following Karami et al. (2024).
- **Seed identity hash.** The final seed hash is `2·h(x0) − h(x1)`, also following Karami et al. (2024).

## Frozen parameters (10% density)

| Method              | Parameters                           | Calibrated density |
| ------------------- | ------------------------------------ | ------------------ |
| Minimizers          | k = 30, w = 19                       | 9.97%              |
| Open syncmers       | k = 30, s = 21, t = 4 (middle)       | 9.99%              |
| Syncmer randstrobes | l = 15, s = 6, window = 2–5 syncmers | 9.95%              |

The 20% set is in `results/frozen_config.json`.
