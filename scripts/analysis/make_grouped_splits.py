"""
Make leakage-free 5-fold splits for nnU-Net: all time points of a patient go to the SAME fold.
Folds are balanced for (a) number of cases and (b) number of patients with a resection cavity (RC).

Input : longitudinal_check.csv (from longitudinal_check.py; columns case, patient, rc_mm3)
Output: splits_final_grouped.json in nnU-Net format  [{"train": [...], "val": [...]}, x5]

usage:
  python make_grouped_splits.py --csv /workspace/data/longitudinal_check.csv \
                                --out /workspace/data/splits_final_grouped.json
"""
import argparse
import csv
import json
import random
from collections import defaultdict

K = 5
SEED = 12345


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases_of = defaultdict(list)
    rc_patient = defaultdict(bool)
    for r in csv.DictReader(open(args.csv)):
        cases_of[r["patient"]].append(r["case"])
        rc_patient[r["patient"]] |= float(r["rc_mm3"]) > 0

    # assign RC patients first, then the rest; big patients (many time points) first in each group
    rng = random.Random(SEED)
    folds = [{"cases": [], "rc_patients": 0} for _ in range(K)]
    for has_rc in (True, False):
        pats = [p for p in cases_of if rc_patient[p] == has_rc]
        rng.shuffle(pats)
        pats.sort(key=lambda p: -len(cases_of[p]))
        for p in pats:
            key = (lambda f: (f["rc_patients"], len(f["cases"]))) if has_rc else (lambda f: len(f["cases"]))
            f = min(folds, key=key)
            f["cases"] += cases_of[p]
            f["rc_patients"] += has_rc

    all_cases = sorted(c for cs in cases_of.values() for c in cs)
    splits = []
    for f in folds:
        val = sorted(f["cases"])
        splits.append({"train": sorted(set(all_cases) - set(val)), "val": val})

    # sanity checks
    pid = lambda c: c.split("-")[2]
    for i, s in enumerate(splits):
        leak = {pid(c) for c in s["train"]} & {pid(c) for c in s["val"]}
        print(f"fold {i}: val {len(s['val'])} cases | RC patients {folds[i]['rc_patients']} | leaked patients {len(leak)}")
    assert sorted(c for s in splits for c in s["val"]) == all_cases, "every case must be in exactly one val fold"
    json.dump(splits, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
