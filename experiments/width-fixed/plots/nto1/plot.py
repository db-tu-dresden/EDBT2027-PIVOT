#!/usr/bin/env python3
"""n:1 vs 1:1: speedup of the n:1 translation (`pivot`) over the 1:1 one (`pivot-no-nto1`)
on the kernels whose code the n:1 patterns change, per host and vector width.

Bars are the median of the per-run paired ratios; the tables also hold their min and max.
Writes one table per host to data/<arch>.csv and draws vip_nto1.tex from them. Given the paper's
figures/ directory, it also copies the PDF there.

  plots/nto1/plot.py [paper/figures]   -> plots/nto1/data/*.csv, plots/nto1/vip_nto1.pdf
"""
import sys, os, csv, math, shutil
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parents[1] / "exp"
sys.path.insert(0, str(HERE.parents[2] / "figures"))
from build import build  # noqa: E402

SUBJECT, REF = "pivot", "pivot-no-nto1"
WIDTHS = ["128", "256", "512"]
ARCHES = ["sapphirerapids", "znver4", "a64fx"]
# (driver, label) -> tick text; None as label = geomean over all of the driver's kernels
KERNELS = [
    ("main_expr", "expr_decimal/*", "decimal mul"),
    ("main_expr", "expr_decimal//", "decimal div"),
    ("main_expr", None, "expr (all 12)"),
]


def load():
    runs = defaultdict(dict)  # (arch, width, driver, label) -> {method: {run: bpts}}
    for f in sorted(EXP_DIR.glob("*/results/*/results.csv")):
        for r in csv.DictReader(open(f)):
            if r["method"] in (SUBJECT, REF):
                runs[(r["arch"], r["width"], r["driver"], r["label"])] \
                    .setdefault(r["method"], {})[r["run"]] = float(r["bpts"])
    return runs


def paired_ratios(runs, arch, w, drv, lbl):
    """Per-run ratios n:1 / 1:1; both variants' run i ran back to back."""
    keys = [k for k in runs if k[:3] == (arch, w, drv) and (lbl is None or k[3] == lbl)]
    per_run = defaultdict(list)
    for k in keys:
        m = runs[k]
        if SUBJECT not in m or REF not in m:
            continue
        for run, p in m[SUBJECT].items():
            o = m[REF].get(run)
            if p > 0 and o and o > 0:
                per_run[run].append(p / o)
    return [math.exp(sum(map(math.log, xs)) / len(xs)) for xs in per_run.values()]


def main():
    runs = load()
    os.makedirs(HERE / "data", exist_ok=True)
    for arch in ARCHES:
        rows = []
        for drv, lbl, tick in KERNELS:
            row = [tick]
            for w in WIDTHS:
                rs = paired_ratios(runs, arch, w, drv, lbl)
                if not rs:
                    sys.exit(f"no paired {SUBJECT}/{REF} runs for {arch} {w} {drv} {lbl}")
                row += [f"{v:.6f}" for v in (median(rs), min(rs), max(rs))]
            rows.append(row)
        with open(HERE / "data" / f"{arch}.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(["kernel"] + [f"{s}{w}" for w in WIDTHS for s in ("med", "min", "max")])
            out.writerows(rows)
        print(f"{arch}: " + "  ".join(f"{r[0]}=" + "/".join(r[1::3]) for r in rows))

    pdf = HERE / "vip_nto1.pdf"
    build(HERE / "vip_nto1.tex", pdf)
    print(f"  wrote {pdf}")
    if len(sys.argv) > 1:
        shutil.copy(pdf, sys.argv[1])
        print(f"  copied to {sys.argv[1]}")


if __name__ == "__main__":
    main()
