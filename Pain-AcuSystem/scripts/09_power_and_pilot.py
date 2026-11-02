#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study import analysis
from tcm_study import stats as ss
from tcm_study.config import load_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--n", type=int, nargs="*", default=[55, 35, 20])
    ap.add_argument("--sd", type=float, nargs="*", default=[1.0, 1.5, 2.0, 2.5], help="SD of the paired difference (10-point scale)")
    ap.add_argument("--delta", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.5], help="true mean differences to evaluate")
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--from-cv", default=None)
    a = ap.parse_args()
    pl = analysis.plan(load_cfg(a.config))
    m, alpha = len(pl["contrasts"]), pl["alpha_two_sided"]
    sds = list(a.sd)
    if a.from_cv:
        d = pd.read_csv(a.from_cv, encoding="utf-8-sig")
        w = d.pivot_table(index="case_id", columns="arm", values="score").dropna()
        est = {f"{y}-{x}": float((w[y] - w[x]).std(ddof=1)) for x, y in itertools.combinations(w.columns, 2)}
        print("[CV] SD of paired differences by contrast:", {k: round(v, 2) for k, v in est.items()})
        sds = sorted({round(min(est.values()), 2), round(sum(est.values()) / len(est), 2), round(max(est.values()), 2)})
    print(f"design: paired, two-sided alpha={alpha}; {m} pairwise contrasts with Holm adjustment (worst case alpha/{m} = {alpha / m:.4f})")
    print(f"\nminimal detectable mean difference at {int(a.power * 100)}% power (points on the 10-point scale)")
    print(f"{'n':>4} {'SD diff':>8} {'unadjusted':>11} {'Holm worst case':>16}")
    for n in a.n:
        for sd in sds:
            print(f"{n:>4} {sd:>8.2f} {ss.mdd_paired(n, sd, alpha, a.power):>11.2f} {ss.mdd_paired(n, sd, alpha / m, a.power):>16.2f}")
    print("\npower for given true differences (Holm worst case)")
    print(f"{'n':>4} {'SD diff':>8}" + "".join(f"{'Δ=' + str(x):>9}" for x in a.delta))
    for n in a.n:
        for sd in sds:
            print(f"{n:>4} {sd:>8.2f}" + "".join(f"{ss.power_paired(n, x, sd, alpha / m):>9.2f}" for x in a.delta))
    print("\ncases needed for 80% power (Holm worst case):")
    for sd in sds:
        print(f"  SD {sd:.2f}: " + ", ".join(f"Δ={x} → n={ss.n_paired(x, sd, alpha / m, a.power)}" for x in a.delta))
    print("\n[NOTE] report these as the sensitivity of the fixed 55-case design; do not present them as an a-priori sample-size calculation.")


if __name__ == "__main__":
    main()
