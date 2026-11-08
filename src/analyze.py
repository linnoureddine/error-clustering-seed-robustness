# Turns the raw experiment output into the paper's tables and figures.
# Usage: python analyze.py            (reads results/raw/, writes results/analysis/)
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import errors as ER
from common import RESULTS

METRICS = ["seed_retention", "seq_coverage", "match_coverage", "island_esize", "max_island", "window_hit_150"]
LOWER_IS_BETTER = {"island_esize", "max_island"}
NICE = {"seed_retention": "Seed retention", "seq_coverage": "Sequence coverage",
        "match_coverage": "Match coverage", "island_esize": "Island E-size (nt)",
        "max_island": "Longest island (nt)", "window_hit_150": "150-nt window hit rate"}

# short id -> (group, method, params as stored, display name)
def method_ids(config):
    sp = config["sparse"]
    s10 = sp["0.10"]["syncmer_randstrobes"]["params"]["s"]
    p = lambda d: json.dumps(d, sort_keys=True)
    return {
        "kmers": ("dense", "kmers", p(config["references"]["kmers"]), "All 30-mers"),
        "randstrobes": ("dense", "randstrobes", p(config["references"]["randstrobes"]), "Randstrobes"),
        "min10": ("sparse10", "minimizers", p(sp["0.10"]["minimizers"]["params"]), "Minimizers"),
        "sync10": ("sparse10", "syncmers", p(sp["0.10"]["syncmers"]["params"]), "Syncmers"),
        "srs10": ("sparse10", "syncmer_randstrobes", p(sp["0.10"]["syncmer_randstrobes"]["params"]), "Syncmer randstrobes"),
        "min20": ("sparse20", "minimizers", p(sp["0.20"]["minimizers"]["params"]), "Minimizers"),
        "sync20": ("sparse20", "syncmers", p(sp["0.20"]["syncmers"]["params"]), "Syncmers"),
        "srs20": ("sparse20", "syncmer_randstrobes", p(sp["0.20"]["syncmer_randstrobes"]["params"]), "Syncmer randstrobes"),
        "srs10_w13": ("window", "syncmer_randstrobes", p({"l": 15, "s": s10, "wmin": 1, "wmax": 3}), "Syncmer randstrobes, window 1-3"),
        "srs10_w48": ("window", "syncmer_randstrobes", p({"l": 15, "s": s10, "wmin": 4, "wmax": 8}), "Syncmer randstrobes, window 4-8"),
    }

# (A, B): we look at A - B. Fixed before looking at any results.
PRIMARY_PAIRS = [("randstrobes", "kmers"), ("sync10", "min10"), ("srs10", "sync10"), ("srs10", "min10")]
ROBUST_PAIRS = [("sync20", "min20"), ("srs20", "sync20"), ("srs20", "min20"),
                ("srs10_w13", "sync10"), ("srs10_w13", "min10"), ("srs10_w48", "sync10"), ("srs10_w48", "min10")]

B = 10_000            # bootstrap resamples
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # first three validated categorical slots


class Boot:
    # resample regions with replacement; the same resamples are reused for every statistic
    def __init__(self, n, seed=2028):
        self.idx = np.random.default_rng(seed).integers(0, n, size=(B, n))

    def ci(self, x):
        x = np.asarray(x, float)
        means = x[self.idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return x.mean(), lo, hi


def region_means(raw, ids):
    key = {(g, m, p): k for k, (g, m, p, _) in ids.items()}
    raw = raw.assign(mid=[key.get(t) for t in zip(raw.group, raw.method, raw.params)])
    if raw.mid.isna().any():
        raise ValueError("raw results contain a method not listed in method_ids()")
    # average the replicates within each region: regions are the independent units
    return raw.groupby(["mid", "rate", "m", "rho", "region_id"])[METRICS].mean().reset_index()


def wide(reg, metric):
    w = reg.pivot_table(index=["rate", "m", "rho", "region_id"], columns="mid", values=metric)
    return w.sort_index()


def summarize(reg, boot):
    rows = []
    for (mid, rate, m, rho), d in reg.groupby(["mid", "rate", "m", "rho"]):
        d = d.sort_values("region_id")
        for metric in METRICS:
            mean, lo, hi = boot.ci(d[metric])
            rows.append(dict(mid=mid, rate=rate, m=m, rho=rho, metric=metric, mean=mean, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def contrasts(reg, pairs, boot):
    rows = []
    for metric in METRICS:
        w = wide(reg, metric)
        for a, b in pairs:
            for (rate, m, rho), d in w.groupby(level=["rate", "m", "rho"]):
                diff = (d[a] - d[b]).to_numpy()
                mean, lo, hi = boot.ci(diff)
                rows.append(dict(A=a, B=b, metric=metric, rate=rate, m=m, rho=rho, diff=mean, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def ranking_changes(reg, pairs, boot):
    # compare the A - B gap under uniform errors (m = 1) with the most clustered case (m = 16)
    rows = []
    for metric in METRICS:
        w = wide(reg, metric)
        for a, b in pairs:
            for rate, rho in [(0.01, 0.25), (0.05, 0.25), (0.10, 0.25), (0.05, 0.5)]:
                base = w.xs((rate, 1, 0.25), level=["rate", "m", "rho"])  # m = 1 doesn't depend on rho
                clus = w.xs((rate, 16, rho), level=["rate", "m", "rho"])
                d1 = (base[a] - base[b]).to_numpy()
                d16 = (clus[a] - clus[b]).to_numpy()
                m1, lo1, hi1 = boot.ci(d1)
                m16, lo16, hi16 = boot.ci(d16)
                mi, loi, hii = boot.ci(d16 - d1)
                sign = -1 if metric in LOWER_IS_BETTER else 1   # sign * diff > 0 means A is better
                sig1, sig16, sigc = lo1 > 0 or hi1 < 0, lo16 > 0 or hi16 < 0, loi > 0 or hii < 0
                lead16 = "A" if sign * m16 > 0 else "B"
                if sig1 and sig16 and np.sign(m1) != np.sign(m16):
                    verdict = "ranking flips"
                elif not sigc:
                    verdict = "no significant change"
                elif not sig1:
                    verdict = f"gap appears ({lead16} better)" if sig16 else "no significant change"
                elif not sig16:
                    verdict = "gap closes"
                elif abs(m16) < abs(m1):
                    verdict = "gap narrows"
                else:
                    verdict = "gap widens"
                rows.append(dict(A=a, B=b, metric=metric, rate=rate, rho_clustered=rho,
                                 diff_m1=m1, lo_m1=lo1, hi_m1=hi1, diff_m16=m16, lo_m16=lo16, hi_m16=hi16,
                                 change=mi, lo_change=loi, hi_change=hii, verdict=verdict))
    return pd.DataFrame(rows)


def sanity_check(rc):
    # uniform errors must reproduce Edgar 2021 (syncmers conserve more than minimizers)
    # and Sahlin 2021 (randstrobes give more match coverage than k-mers)
    checks = [("Edgar 2021: syncmers > minimizers, sequence coverage", "sync10", "min10", "seq_coverage"),
              ("Edgar 2021 at 20% density", "sync20", "min20", "seq_coverage"),
              ("Sahlin 2021: randstrobes > k-mers, match coverage", "randstrobes", "kmers", "match_coverage")]
    rows = []
    for name, a, b, metric in checks:
        for rate in (0.01, 0.05, 0.10):
            r = rc[(rc.A == a) & (rc.B == b) & (rc.metric == metric) & (rc.rate == rate) & (rc.rho_clustered == 0.25)].iloc[0]
            rows.append(dict(check=name, rate=rate, diff=r.diff_m1, lo=r.lo_m1, hi=r.hi_m1, passed=bool(r.lo_m1 > 0)))
    return pd.DataFrame(rows)


def rank_table(summary, ids):
    rows = []
    for group in ("dense", "sparse10", "sparse20"):
        mids = [k for k, v in ids.items() if v[0] == group]
        s = summary[summary.mid.isin(mids)]
        for (rate, m, rho, metric), d in s.groupby(["rate", "m", "rho", "metric"]):
            d = d.sort_values("mean", ascending=metric in LOWER_IS_BETTER)
            rows.append(dict(group=group, rate=rate, m=m, rho=rho, metric=metric,
                             ranking=" > ".join(ids[k][3] for k in d.mid),
                             means="; ".join(f"{v:.4g}" for v in d["mean"])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def style(ax):
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9994")
    ax.tick_params(colors="#52514e", labelsize=9)


def fig_error_patterns(err, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    conds = [(r, m, 0.25) for r in (0.01, 0.05, 0.10) for m in (1, 2, 4, 8, 16)] + [(0.05, m, 0.5) for m in (2, 4, 8, 16)]
    ax = axes[0]
    for row, m in enumerate((1, 4, 16)):
        cid = conds.index((0.05, m, 0.25))
        rng = np.random.default_rng([2027, 0, cid, 0])       # same draw as region 0, replicate 0
        pos = ER.cluster_positions(10_000, 500, m, 0.25, rng)
        pos = pos[pos < 3000]
        ax.vlines(pos, row - 0.35, row + 0.35, color=COLORS[row], linewidth=0.8)
    ax.set_yticks([0, 1, 2], ["m = 1\n(uniform)", "m = 4", "m = 16"])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(0, 3000)
    ax.set_xlabel("Position (first 3 kb of a 10-kb region)", fontsize=9, color="#52514e")
    ax.set_title("a  Same 500 errors (5%), different placement", loc="left", fontsize=10)
    style(ax); ax.grid(False)

    e = err[err.rho == 0.25]
    for ax, col, title in ((axes[1], "fano_100", "b  Clustering (Fano factor, 100-nt bins)"),
                           (axes[2], "max_error_free_run", "c  Longest error-free stretch (nt)")):
        for i, rate in enumerate((0.01, 0.05, 0.10)):
            d = e[e.rate == rate].groupby("m")[col].mean()
            ax.plot(d.index, d.values, "-o", color=COLORS[i], lw=2, ms=5, label=f"{rate:.0%} errors")
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("Errors per cluster (m)", fontsize=9, color="#52514e")
        ax.set_title(title, loc="left", fontsize=10)
        style(ax)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_error_patterns.png", dpi=200)
    plt.close(fig)


def fig_metrics_vs_m(summary, ids, rate, out):
    metrics = ["seq_coverage", "match_coverage", "max_island"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for r, (group, gname) in enumerate((("dense", "Dense (one seed per position)"), ("sparse10", "Sparse, 10% density"))):
        mids = [k for k, v in ids.items() if v[0] == group]
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            for i, mid in enumerate(mids):
                d = summary[(summary.mid == mid) & (summary.metric == metric) & (summary.rate == rate)
                            & (summary.rho == 0.25)].sort_values("m")
                ax.fill_between(d.m, d.lo, d.hi, color=COLORS[i], alpha=0.18, lw=0)
                ax.plot(d.m, d["mean"], "-o", color=COLORS[i], lw=2, ms=5, label=ids[mid][3])
            ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
            if r == 1:
                ax.set_xlabel("Errors per cluster (m)", fontsize=9, color="#52514e")
            ax.set_title(NICE[metric] if r == 0 else "", fontsize=10, loc="left")
            style(ax)
        axes[r, 0].set_ylabel(gname, fontsize=10)
        axes[r, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Seed metrics as errors cluster ({rate:.0%} substitutions, same total in every condition)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / f"fig2_metrics_vs_m_rate{int(rate * 100):02d}.png", dpi=200)
    plt.close(fig)


def fig_contrasts(con, ids, pairs, n_regions, out, fname, series=None):
    # series: list of (label, rate, rho); m = 1 always comes from rho = 0.25 (uniform errors)
    series = series or [(f"{r:.0%} errors", r, 0.25) for r in (0.01, 0.05, 0.10)]
    metrics = ["seq_coverage", "match_coverage"]
    fig, axes = plt.subplots(2, len(pairs), figsize=(3.75 * len(pairs), 6.4), sharex=True, squeeze=False)
    for c, (a, b) in enumerate(pairs):
        for r, metric in enumerate(metrics):
            ax = axes[r, c]
            ax.axhline(0, color="#9a9994", lw=1)
            for i, (label, rate, rho) in enumerate(series):
                d = con[(con.A == a) & (con.B == b) & (con.metric == metric) & (con.rate == rate)
                        & ((con.rho == rho) | ((con.m == 1) & (con.rho == 0.25)))].sort_values("m")
                ax.fill_between(d.m, d.lo, d.hi, color=COLORS[i], alpha=0.18, lw=0)
                ax.plot(d.m, d["diff"], "-o", color=COLORS[i], lw=2, ms=5, label=label)
            ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
            if r == 0:
                dens = {"sparse10": "\n(10% density)", "sparse20": "\n(20% density)", "window": "\n(10% density)"}
                ax.set_title(f"{ids[a][3]}\n− {ids[b][3]}" + dens.get(ids[a][0], "").replace("\n", " "),
                             fontsize=10, loc="left")
            if r == 1:
                ax.set_xlabel("Errors per cluster (m)", fontsize=9, color="#52514e")
            if c == 0:
                ax.set_ylabel(f"Difference in\n{NICE[metric].lower()}", fontsize=10)
            style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Gap between methods as errors cluster (above 0 = first method better; "
                 f"bands = 95% CI over {n_regions} regions)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / fname, dpi=200)
    plt.close(fig)


def fig_cluster_width(con, ids, gap_nt, n_regions, out):
    # Is the loss of the strobemer advantage explained by cluster width rather than errors per cluster?
    # 5% errors only (the only rate run with rho = 0.5). Width of a cluster window = max(m, ceil(m / rho)).
    pairs = [("randstrobes", "kmers"), ("srs10", "sync10"), ("srs10", "min10")]
    width = lambda m, rho: 1 if m == 1 else max(m, int(np.ceil(m / rho)))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), squeeze=False)
    for c, (a, b) in enumerate(pairs):
        d0 = con[(con.A == a) & (con.B == b) & (con.metric == "match_coverage") & (con.rate == 0.05)]
        for r, xname in enumerate(("m", "width")):
            ax = axes[r, c]
            ax.axhline(0, color="#9a9994", lw=1)
            if xname == "width":
                g = gap_nt[a]
                ax.axvline(g, color="#52514e", lw=1, ls="--")
                ax.text(g * 1.06, 0.98, f"gap between\nstrobes ≈ {g:.0f} nt", transform=ax.get_xaxis_transform(),
                        va="top", fontsize=8, color="#52514e")
            for i, rho in enumerate((0.25, 0.5)):
                d = d0[(d0.rho == rho) | ((d0.m == 1) & (d0.rho == 0.25))].sort_values("m")
                x = d.m if xname == "m" else [width(m, rho) for m in d.m]
                ax.errorbar(x, d["diff"], yerr=[d["diff"] - d.lo, d.hi - d["diff"]], fmt="-o", color=COLORS[i],
                            lw=2, ms=5, capsize=2, label=f"ρ = {rho} (cluster width = {'4m' if rho == .25 else '2m'} nt)")
            ax.set_xscale("log", base=2)
            ticks = [1, 2, 4, 8, 16] if xname == "m" else [1, 4, 8, 16, 32, 64]
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.set_xlabel("Errors per cluster (m)" if xname == "m" else "Cluster width (nt); 1 = uniform errors",
                          fontsize=9, color="#52514e")
            if r == 0:
                dens = "" if ids[a][0] == "dense" else " (10% density)"
                ax.set_title(f"{ids[a][3]}\n− {ids[b][3]}{dens}", fontsize=10, loc="left")
            if c == 0:
                ax.set_ylabel("Difference in\nmatch coverage", fontsize=10)
            style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Strobemer advantage at 5% errors, plotted against errors per cluster (top) and cluster width (bottom); "
                 f"bars = 95% CI over {n_regions} regions", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig4_cluster_width.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw", help="folder inside results/ with the experiment output")
    args = ap.parse_args()
    raw_dir, out = RESULTS / args.raw, RESULTS / "analysis"
    out.mkdir(exist_ok=True)

    with open(RESULTS / "frozen_config.json") as fh:
        ids = method_ids(json.load(fh))
    raw = pd.read_csv(raw_dir / "experiment_raw.csv.gz")
    err = pd.read_csv(raw_dir / "error_stats.csv.gz")

    reg = region_means(raw, ids)
    boot = Boot(reg.region_id.nunique())

    summary = summarize(reg, boot)
    con = contrasts(reg, PRIMARY_PAIRS + ROBUST_PAIRS, boot)
    rc = ranking_changes(reg, PRIMARY_PAIRS + ROBUST_PAIRS, boot)
    sanity = sanity_check(rc)
    ranks = rank_table(summary, ids)
    errsum = err.groupby(["rate", "m", "rho"])[["fano_100", "gap_cv", "median_nn_dist", "max_error_free_run"]].mean()

    summary.assign(method=summary.mid.map(lambda k: ids[k][3]), group=summary.mid.map(lambda k: ids[k][0])) \
        .to_csv(out / "summary_metrics.csv", index=False)
    con.to_csv(out / "method_gaps.csv", index=False)
    rc.to_csv(out / "ranking_changes.csv", index=False)
    sanity.to_csv(out / "sanity_check.csv", index=False)
    ranks.to_csv(out / "rank_table.csv", index=False)
    errsum.to_csv(out / "error_pattern_stats.csv")

    fig_error_patterns(err, out)
    for rate in (0.01, 0.05, 0.10):
        fig_metrics_vs_m(summary, ids, rate, out)
    n = reg.region_id.nunique()
    fig_contrasts(con, ids, PRIMARY_PAIRS, n, out, "fig3_method_gaps_vs_m.png")
    # supplementary: robustness checks
    fig_contrasts(con, ids, [("sync20", "min20"), ("srs20", "sync20"), ("srs20", "min20")], n, out,
                  "figS1_gaps_20pct_density.png")
    fig_contrasts(con, ids, PRIMARY_PAIRS, n, out, "figS2_gaps_tight_clusters.png",
                  series=[("5% errors, ρ = 0.25", 0.05, 0.25), ("5% errors, ρ = 0.5", 0.05, 0.5)])
    fig_contrasts(con, ids, [("srs10", "sync10"), ("srs10_w13", "sync10"), ("srs10_w48", "sync10")], n, out,
                  "figS3_gaps_strobe_window.png")

    # bases between the two strobes, from calibration: mean span - 2 * strobe length
    cal = pd.read_csv(RESULTS / "calibration_sweep.csv")
    gap_nt = {}
    for mid in ("randstrobes", "srs10"):
        g, name, p, _ = ids[mid]
        row = cal[(cal.method == name) & (cal.params.map(lambda s: json.dumps(json.loads(s), sort_keys=True)) == p)].iloc[0]
        gap_nt[mid] = row.mean_span_nt - 2 * json.loads(p)["l"]
    fig_cluster_width(con, ids, gap_nt, n, out)

    pd.set_option("display.width", 200)
    print("Sanity check (uniform errors, m = 1):")
    print(sanity.to_string(index=False))
    prim = rc[rc.metric.isin(["seq_coverage", "match_coverage", "seed_retention", "max_island"])
              & rc.set_index(["A", "B"]).index.isin(PRIMARY_PAIRS)]
    print("\nPrimary pairs, m = 1 vs m = 16:")
    print(prim[["A", "B", "metric", "rate", "rho_clustered", "diff_m1", "diff_m16", "change", "verdict"]]
          .round(4).to_string(index=False))
    print(f"\nTables and figures written to {out}")


if __name__ == "__main__":
    main()
