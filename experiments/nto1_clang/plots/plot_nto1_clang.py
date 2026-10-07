#!/usr/bin/env python3
"""plot_nto1_clang.py — speedup of clang_nto1 over clang_1to1, one panel per machine.

Speedup = ns/op(clang_1to1) / ns/op(clang_nto1) per kernel and size regime (cache-resident,
DRAM). Writes one table per machine to data/<machine>.csv, in the kernel order and with the
labels of the x axis, and draws clang_nto1.tex from them. Given the paper's figures/
directory, it also copies the PDF there.

  plots/plot_nto1_clang.py [paper/figures]   -> plots/data/*.csv, plots/clang_nto1.pdf
"""
import csv, glob, os, shutil, sys

PLOTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PLOTS)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "figures"))
from build import build  # noqa: E402

# The panels of the figure: the SSE kernels run on the ARM machines, the NEON kernels on x86.
MACHINES = {"a64fx": "sse", "hpi-altra": "sse", "db-intel": "neon", "db-amd": "neon"}

# The kernel set and its order on the x axis, with the short labels.
KERNELS = {
    "sse": [("hadd_s32_128", "hadd"), ("hmax_s32_128", "hmax"), ("hmin_f32_128", "hmin"),
            ("popcount_u32_128", "popcount"), ("rbit_u8_128", "rbit"),
            ("satadd_s32_128", "satadd"), ("cvt_u32_f32_128", "cvt")],
    "neon": [("neon_popcount_u32_128", "popcount"), ("neon_mul_u64_128", "mul"),
             ("neon_clz_u64_128", "clz")],
}
REGIMES = ["cache", "dram"]


def load():
    ns = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "results-*.csv"))):
        with open(p) as f:
            for r in csv.DictReader(f):
                ns[(r["machine"], r["example"], r["regime"], r["variant"])] = float(r["ns_per_op"])
    return ns


def speedup(ns, m, e, rg):
    try:
        return ns[(m, e, rg, "clang_1to1")] / ns[(m, e, rg, "clang_nto1")]
    except KeyError:
        sys.exit(f"no clang_1to1/clang_nto1 pair for {m} {e} {rg} in results/")


def main():
    ns = load()
    os.makedirs(os.path.join(PLOTS, "data"), exist_ok=True)
    for m, isa in MACHINES.items():
        rows = [(label, *(speedup(ns, m, e, rg) for rg in REGIMES)) for e, label in KERNELS[isa]]
        with open(os.path.join(PLOTS, "data", f"{m}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", *REGIMES])
            w.writerows((label, *(f"{v:.6f}" for v in vs)) for label, *vs in rows)
        print(f"{m}: " + "  ".join(f"{label}={c:.2f}/{d:.2f}" for label, c, d in rows))

    pdf = os.path.join(PLOTS, "clang_nto1.pdf")
    build(os.path.join(PLOTS, "clang_nto1.tex"), pdf)
    print(f"  wrote {pdf}")
    if len(sys.argv) > 1:
        shutil.copy(pdf, sys.argv[1])
        print(f"  copied to {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
