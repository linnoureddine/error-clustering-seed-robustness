from __future__ import annotations

import csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENOME = ROOT / "data" / "GCF_000005845.2_ASM584v2_genomic.fna"
RESULTS = ROOT / "results"


def load_genome(path: Path = GENOME) -> tuple[str, str]:
    name, chunks = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    break  # only the first record
                name = line[1:].split()[0]
            else:
                chunks.append(line.strip().upper())
    return name, "".join(chunks)


def sample_regions(genome_len: int, n: int, L: int, seed: int,
                   exclude: list[tuple[int, int]] = ()) -> list[tuple[int, int]]:
    # n non-overlapping regions of length L, avoiding anything in exclude
    rng = np.random.default_rng(seed)
    taken = list(exclude)
    out = []
    while len(out) < n:
        s = int(rng.integers(0, genome_len - L + 1))
        if all(s + L <= a or s >= b for a, b in taken):
            taken.append((s, s + L)); out.append((s, s + L))
    return sorted(out)


def write_regions(path: Path, regions, accession: str, purpose: str):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["region_id", "accession", "start0", "end0", "purpose"])
        for i, (a, b) in enumerate(regions):
            w.writerow([i, accession, a, b, purpose])
