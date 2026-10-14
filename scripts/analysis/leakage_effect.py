"""
How much did patient leakage help the fold-0 scores?
Splits the 260 fold-0 validation cases into
  SEEN   : the same patient has another time point in the fold-0 TRAINING set
  UNSEEN : the patient never appears in training
and compares Dice per region (from nnU-Net's summary.json) and RC detection (from the v2 sweep lesion table).

Only reads small files - no images, no GPU.

usage:
  python leakage_effect.py --splits splits_final.json \
      --resencl resencl_summary.json --primus primus_summary.json \
      [--lesions lesions_ensemble.csv]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

REGIONS = {"(1, 2, 3, 4)": "WT", "(1, 3, 4)": "TC", "(3,)": "ET", "(4,)": "RC"}
N_BOOT = 2000


def pid(case):
    return case.split("-")[2]


def case_of(path):
    return Path(path).name.replace(".nii.gz", "")


def boot_ci(a, b, rng):
    """95% CI of mean(a) - mean(b), resampling cases inside each group."""
    d = [np.nanmean(rng.choice(a, len(a))) - np.nanmean(rng.choice(b, len(b))) for _ in range(N_BOOT)]
    return np.nanpercentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--resencl", required=True)
    ap.add_argument("--primus", required=True)
    ap.add_argument("--lesions", default=None)
    args = ap.parse_args()

    fold0 = json.load(open(args.splits))[0]
    train_patients = {pid(c) for c in fold0["train"]}
    seen = {c for c in fold0["val"] if pid(c) in train_patients}
    print(f"fold-0 val: {len(fold0['val'])} cases | SEEN {len(seen)} | UNSEEN {len(fold0['val']) - len(seen)}\n")

    rng = np.random.default_rng(0)
    for name, f in (("ResEnc-L", args.resencl), ("Primus", args.primus)):
        per_case = json.load(open(f))["metric_per_case"]
        print(f"=== {name}: mean Dice (nnU-Net default evaluation) ===")
        print("  region |  SEEN (n)        |  UNSEEN (n)      | UNSEEN - SEEN [95% CI]")
        for key, reg in REGIONS.items():
            s, u = [], []
            for m in per_case:
                d = m["metrics"][key]["Dice"]
                if d is None or (isinstance(d, float) and np.isnan(d)):
                    continue                      # region absent in both GT and prediction
                (s if case_of(m["reference_file"]) in seen else u).append(d)
            s, u = np.array(s), np.array(u)
            lo, hi = boot_ci(u, s, rng)
            print(f"  {reg:6s} | {s.mean():.3f} ({len(s):3d})     | {u.mean():.3f} ({len(u):3d})     |"
                  f" {u.mean() - s.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print()

    if args.lesions:
        rows = list(csv.DictReader(open(args.lesions)))
        print("=== Ensemble RC detection (cavities >= 27 mm3, setting from the v2 sweep table) ===")
        for grp, test in (("SEEN", lambda c: c in seen), ("UNSEEN", lambda c: c not in seen)):
            r = [x for x in rows if test(x["case"])]
            found = sum(x["found"] == "True" for x in r)
            print(f"  {grp:6s}: {found} of {len(r)} cavities found"
                  + (f" ({found / len(r):.0%})" if r else ""))
    print("\nIf UNSEEN is clearly lower (CI below 0), leakage inflated the fold-0 scores; report the UNSEEN numbers too.")


if __name__ == "__main__":
    main()
