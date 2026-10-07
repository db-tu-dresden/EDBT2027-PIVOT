#!/usr/bin/env python3
"""Multiply-shift hash: ns per key of the hand-written kernels, of the one SSE source
translated to TSL by PIVOT, and of that source compiled against SIMDe.

A panel per machine (A64FX, the two x86 hosts and the X100 cores of the SpacemiT K3), a group
per vector width, a bar per variant present at that width: the hand-written kernels (NEON on
A64FX, 128-bit only, one per width on x86, none on the X100), the TSL translation, and SIMDe
(its native paths on x86, its portable vector code on A64FX and the X100). A bar is the median
over 15 runs; lower is better. Each panel has its own log axis, from and to powers of two
around its bars. Writes the tables to data/ and draws ms_hash.tex from them. Given the
paper's figures/ directory, it also copies the PDF there.

On the X100 (VLEN 256) the 512 and 1024 builds use two and four registers per vector.

  plots/plot.py [paper/figures]   -> plots/data/*.csv, plots/ms_hash.pdf
"""
import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
sys.path.insert(0, str(HERE.parents[3] / "figures"))
from build import build  # noqa: E402

# machine -> (results CSV, widths, SIMDe build shown)
MACHINES = {
    "a64fx": ("a64fx/results_ca08.csv", ["128", "256", "512"], "simde-portable"),
    "spr": ("db-intel/results_bolvar.csv", ["128", "256", "512"], "simde"),
    "zen4": ("db-amd/results_nerzhul.csv", ["128", "256", "512"], "simde"),
    "x100": ("db-riscv/results_bin_x100_milkv.csv", ["256", "512", "1024"], "simde-portable"),
}
# The bars of a group, left to right.
ORDER = ["native", "tsl_vla", "simde"]


def load(path):
    """(variant, width) -> ns per key of the ok rows."""
    vals = defaultdict(list)
    with open(RESULTS / path) as f:
        for r in csv.DictReader(f):
            if r["ok"] == "1":
                vals[(r["variant"], r["width"])].append(float(r["ns_per_elem"]))
    return vals


def axis_range(values):
    """Powers of two around the values, with room below the shortest bar and above the tallest,
    at least two doublings apart so that a grid line falls inside."""
    lo = 2.0 ** math.floor(math.log2(min(values)))
    hi = 2.0 ** math.ceil(math.log2(max(values)))
    lo, hi = (lo / 2 if min(values) / lo < 1.2 else lo), (hi * 2 if hi / max(values) < 1.1 else hi)
    while hi / lo < 4:
        lo /= 2
    return lo, hi


def write(name, header, rows):
    with open(HERE / "data" / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    (HERE / "data").mkdir(exist_ok=True)
    for m, (path, widths, simde) in MACHINES.items():
        data = load(path)
        write(f"{m}-widths.csv", ["width"], [[w] for w in widths])
        bars = defaultdict(list)  # variant -> [(group, position in group, ns)]
        for g, w in enumerate(widths):
            cells = [(v, data.get((simde if v == "simde" else v, w))) for v in ORDER]
            cells = [(v, median(runs)) for v, runs in cells if runs]
            for i, (v, ns) in enumerate(cells):
                bars[v].append((g, i - (len(cells) - 1) / 2, ns))
            print(f"{m:6s} {w:5s} " + "  ".join(f"{v}={ns:.3f}" for v, ns in cells))
        for v, rows in bars.items():
            write(f"{m}-{v}.csv", ["group", "k", "ns"], [(g, f"{k:g}", f"{ns:.6f}") for g, k, ns in rows])
        write(f"{m}-range.csv", ["min", "max"],
              [[f"{x:g}" for x in axis_range([ns for rows in bars.values() for *_, ns in rows])]])

    pdf = HERE / "ms_hash.pdf"
    build(HERE / "ms_hash.tex", pdf)
    print(f"  wrote {pdf}")
    if len(sys.argv) > 1:
        shutil.copy(pdf, sys.argv[1])
        print(f"  copied to {sys.argv[1]}")


if __name__ == "__main__":
    main()
