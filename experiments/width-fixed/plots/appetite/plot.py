#!/usr/bin/env python3
"""The join of VIP on Sapphire Rapids and A64FX at each vector width: pivot's speedup over SIMDe
for each of its kernels, the ratios whose geomean is the join bar of vip_all (plot_relative.py).
Its two SIMDe baselines share one bar here; vip_all tells them apart.

Writes one table per machine and kernel to data/<arch>-<kernel>.csv (each bar's width and
position in its group) and the range of all speedups, and draws join_appetite.tex from them.
Given the paper's figures/ directory, it also copies the PDF there.

  plots/appetite/plot.py [paper/figures]   -> data/, join_appetite.pdf
"""
import sys, csv, shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parents[1] / "exp"
sys.path.insert(0, str(HERE.parents[2] / "figures"))
sys.path.insert(0, str(HERE.parent / "width_scaling"))
from build import build  # noqa: E402
from plot_relative import load, SUBJECT, WIDTHS  # noqa: E402

DRIVER = "main_join"
ARCHES = ["sapphirerapids", "a64fx"]
SIMDE = ["simde", "simde-portable"]


def main():
    med = load(EXP_DIR)
    data = HERE / "data"
    shutil.rmtree(data, ignore_errors=True)
    data.mkdir()

    rows = defaultdict(list)  # (arch, kernel) -> [group, k, speedup]
    shown = []
    for arch in ARCHES:
        for g, w in enumerate(WIDTHS):
            bars = {}  # kernel -> speedup, named by its build/probe payload columns
            for lbl in sorted({k[4] for k in med if k[0] == arch and k[2] == w and k[3] == DRIVER}):
                p = med.get((arch, SUBJECT, w, DRIVER, lbl))
                # load() keeps a single SIMDe per cell, portable or not
                o = next((med[k] for b in SIMDE if (k := (arch, b, w, DRIVER, lbl)) in med), None)
                if p and o:
                    bars[lbl.split()[-1]] = p / o
            # k: the bar's distance from the group's centre, in bars
            for i, (kernel, s) in enumerate(bars.items()):
                rows[(arch, kernel)].append([g, f"{i - (len(bars) - 1) / 2:g}", f"{s:.6f}"])
                shown.append(s)
            print(f"{arch} {w}: " + "  ".join(f"{kn} {s:.2f}" for kn, s in bars.items()))
    for (arch, kernel), rs in rows.items():
        with open(data / f"{arch}-{kernel}.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(["group", "k", "speedup"])
            out.writerows(rs)
    with open(data / "range.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["min", "max"])
        out.writerow([f"{min(shown):.6f}", f"{max(shown):.6f}"])

    pdf = HERE / "join_appetite.pdf"
    build(HERE / "join_appetite.tex", pdf)
    print(f"  wrote {pdf}")
    if len(sys.argv) > 1:
        shutil.copy(pdf, sys.argv[1])
        print(f"  copied to {sys.argv[1]}")


if __name__ == "__main__":
    main()
