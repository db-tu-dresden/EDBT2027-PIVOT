#!/usr/bin/env python3
"""pivot's speedup over each baseline, per host (rows) and vector width (columns).

Per driver, the geomean over its kernels of median(pivot) / median(baseline), with whiskers
from the slowest to the fastest kernel, on a log axis shared by all panels except those
against native, which zoom in around 1. Only kernels that run intrinsics count. Writes one
table per panel and baseline to data/<arch>-<width>-<baseline>.csv (whiskers cut at the axis
to clip-<...>.csv), the drivers with their kernel counts, and the range of all bars and
whiskers, and draws vip_all.tex from them.
Given the paper's figures/ directory, it also copies the PDF there.

  plots/width_scaling/plot_relative.py [paper/figures]   -> data/, vip_all.pdf
"""
import sys, os, csv, math, shutil
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parents[1] / "exp"
sys.path.insert(0, str(HERE.parents[2] / "figures"))
from build import build  # noqa: E402

WIDTHS = ["128", "256", "512"]
SUBJECT = "pivot"
BASELINES = ["native", "simde", "simde-portable"]
ARCHES = ["sapphirerapids", "znver4", "a64fx"]
# Kernels without intrinsics time the same scalar code in every build: the drivers' Codegen
# reference (aggr, join) and VIP's flush_* copy loops (shuffle).
SCALAR = ("Codegen", "flush_")
CAP = 8  # whiskers beyond it stop at it, labelled with their value
ZOOM = (0.9, 1.1)  # the axis of the panels against native, where every bar sits at 1


def load(exp_dir):
    keep = {SUBJECT, "native", "simde", "simde-portable"}
    per_key = {}
    files = sorted(exp_dir.glob("*/results/*/results.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"no results.csv under {exp_dir}/*/results/<host>/")
    for f in files:
        this = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if r["method"] not in keep or r["label"].startswith(SCALAR):
                continue
            # btps times the VIP kernel; the scan drivers print their own scalar loop as bpts.
            this[(r["arch"], r["method"], r["width"], r["driver"], r["label"])].append(float(r["btps"]))
        per_key.update(this)
    med = {k: median(v) for k, v in per_key.items()}

    # One SIMDe per cell: the native-path simde on a64fx and at 512, the portable one below.
    for arch, w, drv, lbl in {(k[0], k[2], k[3], k[4]) for k in med}:
        unchosen = "simde-portable" if arch == "a64fx" or w == "512" else "simde"
        med.pop((arch, unchosen, w, drv, lbl), None)
    return med


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def ratios(med, arch, w, drv, base):
    out = []
    labels = {k[4] for k in med if k[0] == arch and k[2] == w and k[3] == drv}
    for lbl in labels:
        p = med.get((arch, SUBJECT, w, drv, lbl))
        o = med.get((arch, base, w, drv, lbl))
        if p and o and p > 0 and o > 0:
            out.append(p / o)
    return out


def speedup(med, arch, w, drv, base):
    return geomean(ratios(med, arch, w, drv, base))


def main():
    med = load(EXP_DIR)
    drivers = sorted({k[3] for k in med})
    kernels = {d: {k[4] for k in med if k[1] == SUBJECT and k[3] == d} for d in drivers}
    data = HERE / "data"
    shutil.rmtree(data, ignore_errors=True)
    data.mkdir()
    with open(data / "drivers.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["driver"])
        for d in drivers:
            name = d.replace("main_", "").replace("_", r"\_")
            out.writerow([f"{name} ({len(kernels[d])})"])

    shown = []
    for arch in ARCHES:
        for w in WIDTHS:
            zoom = any(ratios(med, arch, w, drv, "native") for drv in drivers)
            bottom, top = ZOOM if zoom else (0, CAP)
            present = [b for b in BASELINES if any(ratios(med, arch, w, d, b) for d in drivers)]
            for b in present:
                rs = [ratios(med, arch, w, drv, b) for drv in drivers]
                k = present.index(b) - (len(present) - 1) / 2  # the bar's place beside the others
                for drv, r in zip(drivers, rs):
                    if r and len(r) != len(kernels[drv]):
                        print(f"  !! {arch} {w} vs {b}: {drv} has {len(r)} of {len(kernels[drv])} kernels")
                bars = [(geomean(r), min(r), max(r)) if r else (float("nan"),) * 3 for r in rs]
                if any(g == g and (lo < bottom or g > top) for g, lo, hi in bars):
                    sys.exit(f"!! {arch} {w} vs {b}: a bar or whisker leaves the axis {bottom}..{top}")
                if not zoom:
                    shown += [x for bar in bars if bar[0] == bar[0] for x in (bar[1], min(bar[2], CAP))]
                with open(data / f"{arch}-{w}-{b}.csv", "w", newline="") as f:
                    out = csv.writer(f)
                    out.writerow(["speedup", "lo", "hi"])
                    out.writerows([f"{g:.6f}", f"{lo:.6f}", f"{min(hi, top):.6f}"] if g == g
                                  else ["nan"] * 3 for g, lo, hi in bars)
                clipped = [[i, k, top, f"{hi:.2f}" if zoom else f"{hi:.0f}"]
                           for i, (g, lo, hi) in enumerate(bars) if hi > top]
                if clipped:
                    with open(data / f"clip-{arch}-{w}-{b}.csv", "w", newline="") as f:
                        out = csv.writer(f)
                        out.writerow(["x", "k", "y", "max"])
                        out.writerows(clipped)
                print(f"{arch} {w} vs {b}: " + " ".join(f"{g:.2f} [{lo:.2f}, {hi:.2f}]" for g, lo, hi in bars))
    with open(data / "range.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["min", "max", "zmin", "zmax"])
        out.writerow([f"{min(shown):.6f}", f"{max(shown):.6f}", *ZOOM])

    pdf = HERE / "vip_all.pdf"
    build(HERE / "vip_all.tex", pdf)
    print(f"  wrote {pdf}")
    if len(sys.argv) > 1:
        shutil.copy(pdf, sys.argv[1])
        print(f"  copied to {sys.argv[1]}")


if __name__ == "__main__":
    main()
