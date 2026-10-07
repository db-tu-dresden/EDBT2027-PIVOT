#!/usr/bin/env python3
"""SVE quicksort translated to TSL (fixed<N>) by PIVOT: speedup over scalar std::sort.

A panel per dtype (int32, double) on one shared axis, in it a group per machine and variant
(the SVE original on A64FX, the TSL translation on A64FX, the two x86 hosts and the X100 cores
of the SpacemiT K3), a bar per vector width. A bar is the median std::sort ns/element of its
machine over the median of the variant (15 runs each); std::sort is width-invariant, the line
at 1. Writes the tables to data/ and draws sort.tex from them. Given the paper's figures/
directory, it also copies the PDF there.

The A64FX numbers are job 2587587 (ca04); SPR and Zen 4 run the
same binaries with TSL's AVX2/SSSE3 compress left-pack (db-intel/, db-amd/). On the X100 (VLEN 256)
the 512 and 1024 builds use two and four registers per vector, the 128 build half of one
(LMUL 1/2, no 64-bit lanes, so int32 only; its own run, against that run's std::sort).

  plots/plot.py [paper/figures]   -> plots/data/*.csv, plots/sort.pdf
"""
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
sys.path.insert(0, str(HERE.parents[2] / "figures"))
from build import build  # noqa: E402

DTYPES = ["int32", "double"]
WIDTHS = ["128", "256", "512", "1024"]
# Groups whose bars get a guide line at the height of the first one's bars.
PAIRS = [(0, 1)]


def load(path, rename=None):
    """(variant, width, dtype) -> list of ns/elem over the sorted_ok rows."""
    rename = rename or {}
    vals = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["sorted_ok"] != "1":
                continue
            v = rename.get(r["variant"], r["variant"])
            vals[(v, r["width"], r["dtype"])].append(float(r["ns_per_elem"]))
    return vals


def groups():
    """(x tick label, result sets, variant) per group, in x order."""
    a64fx = load(RESULTS / "a64fx" / "results_ca04_uzp.csv",
                 {"tsl_fixed": "tsl_fixed_old", "tsl_fixed_uzp": "tsl_fixed"})
    return [
        ("A64FX SVE", [a64fx], "native"),
        ("A64FX TSL", [a64fx], "tsl_fixed"),
        ("SPR TSL", [load(RESULTS / "db-intel" / "results_x86.csv")], "tsl_fixed"),
        ("Zen 4 TSL", [load(RESULTS / "db-amd" / "results_x86.csv")], "tsl_fixed"),
        ("X100 TSL", [load(RESULTS / "db-riscv" / "results_x100.csv"),
                      load(RESULTS / "db-riscv" / "results_x100_int128.csv")], "tsl_fixed"),
    ]


def speedup(sets, var, w, dtype):
    """The median std::sort over the median of one cell, both from the first result set that
    has the cell, or None."""
    for vals in sets:
        runs, std = vals.get((var, w, dtype)), vals.get(("std_sort", "na", dtype))
        if runs and std:
            return median(std) / median(runs)
    return None


def write(name, header, rows):
    with open(HERE / "data" / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    (HERE / "data").mkdir(exist_ok=True)
    gs = groups()
    write("groups.csv", ["label"], [[label] for label, _, _ in gs])
    bars = defaultdict(list)  # (dtype, width) -> [(group, bar distances from its centre, speedup)]
    for dt in DTYPES:
        for g, (label, sets, var) in enumerate(gs):
            cells = [(w, s) for w in WIDTHS if (s := speedup(sets, var, w, dt)) is not None]
            for i, (w, s) in enumerate(cells):
                bars[(dt, w)].append((g, i - (len(cells) - 1) / 2, s))
            print(f"{dt:6s} {label:10s} " + "  ".join(f"{w}={s:.2f}" for w, s in cells))
    for (dt, w), rows in bars.items():
        write(f"{dt}-{w}.csv", ["group", "k", "speedup"],
              [(g, f"{k:g}", f"{s:.6f}") for g, k, s in rows])
    # Per dtype and width, one line per bar of the first group to the same bar of the second;
    # NaN rows split them.
    guides = defaultdict(list)
    for a, b in PAIRS:
        for (dt, w), rows in bars.items():
            ka = {g: (k, s) for g, k, s in rows}
            if a in ka and b in ka:
                (k_a, s), (k_b, _) = ka[a], ka[b]
                guides[(dt, w)] += [(a, f"{k_a:g}", f"{s:.6f}"), (b, f"{k_b:g}", f"{s:.6f}"),
                                    (a, 0, "nan")]
    for (dt, w), rows in guides.items():
        write(f"guides-{dt}-{w}.csv", ["group", "k", "speedup"], rows)
    allv = [s for rows in bars.values() for *_, s in rows]
    write("range.csv", ["min", "max"], [(f"{min(allv):.6f}", f"{max(allv):.6f}")])

    pdf = HERE / "sort.pdf"
    build(HERE / "sort.tex", pdf)
    print(f"  wrote {pdf}")
    if len(sys.argv) > 1:
        shutil.copy(pdf, sys.argv[1])
        print(f"  copied to {sys.argv[1]}")


if __name__ == "__main__":
    main()
