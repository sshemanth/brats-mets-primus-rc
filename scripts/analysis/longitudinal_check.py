"""
Feasibility check for the "longitudinal cavity segmentation" idea. CPU only, about 5-15 minutes.

Answers:
  1. Patient leakage: are time points of the same patient split between train and val in splits_final.json?
  2. How many patients have more than one time point?
  3. How many resection-cavity (RC) cases have an EARLIER scan of the same patient,
     and how many of those earlier scans have NO cavity (a real "before surgery" scan)?
  4. Do earlier and later scans share the same image grid? (if not, registration is needed)

Case IDs look like BraTS-MET-<patient 5 digits>-<time point 3 digits>.

usage (on the GCP VM, where the raw labels already are):
  python longitudinal_check.py \
     --labels /data/nnUNet_raw/Dataset101_BraTSMET/labelsTr \
     --splits /data/nnUNet_preprocessed/Dataset101_BraTSMET/splits_final.json \
     --out /data/longitudinal_check.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RC_LABEL = 4


def parse(case):
    parts = case.split("-")          # ['BraTS', 'MET', '01124', '001']
    return parts[2], parts[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # ---------- 1. leakage ----------
    splits = json.load(open(args.splits))
    print("=== 1. PATIENT LEAKAGE (patients with time points in both train and val) ===")
    for i, f in enumerate(splits):
        tr = {parse(c)[0] for c in f["train"]}
        va = {parse(c)[0] for c in f["val"]}
        print(f"  fold {i}: {len(tr & va):4d} shared patients | val has {len(va)} patients, {len(f['val'])} cases")

    # ---------- read every label once ----------
    files = sorted(args.labels.glob("*.nii.gz"))
    print(f"\nreading {len(files)} label files ...", flush=True)
    info = {}
    for k, f in enumerate(files, 1):
        case = f.name[:-7]
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayViewFromImage(img)
        vox = float(np.prod(img.GetSpacing()))
        info[case] = {
            "rc_mm3": float((arr == RC_LABEL).sum()) * vox,
            "size": img.GetSize(), "spacing": tuple(round(s, 3) for s in img.GetSpacing()),
        }
        if k % 200 == 0:
            print(f"  {k}/{len(files)}", flush=True)

    by_patient = defaultdict(list)
    for case in info:
        p, t = parse(case)
        by_patient[p].append((t, case))
    for p in by_patient:
        by_patient[p].sort()

    # ---------- 2-4 ----------
    multi = {p: v for p, v in by_patient.items() if len(v) > 1}
    rows, rc_cases, with_prior, prior_no_rc, same_grid = [], 0, 0, 0, 0
    for p, tps in by_patient.items():
        for i, (t, case) in enumerate(tps):
            has_rc = info[case]["rc_mm3"] > 0
            prior = tps[i - 1][1] if i > 0 else ""
            prior_rc = info[prior]["rc_mm3"] > 0 if prior else None
            grid = prior and info[prior]["size"] == info[case]["size"] and info[prior]["spacing"] == info[case]["spacing"]
            rows.append({"case": case, "patient": p, "timepoint": t, "rc_mm3": round(info[case]["rc_mm3"], 1),
                         "prior_case": prior, "prior_has_rc": prior_rc, "same_grid_as_prior": bool(grid)})
            if has_rc:
                rc_cases += 1
                if prior:
                    with_prior += 1
                    prior_no_rc += (not prior_rc)
                    same_grid += bool(grid)

    print(f"\n=== 2. TIME POINTS ===")
    print(f"  patients: {len(by_patient)} | with 2+ time points: {len(multi)} "
          f"({sum(len(v) for v in multi.values())} cases)")
    print(f"\n=== 3. RC CASES WITH AN EARLIER SCAN ===")
    print(f"  RC cases: {rc_cases}")
    print(f"  ... with an earlier scan of the same patient: {with_prior}")
    print(f"  ... earlier scan has NO cavity (true before-surgery scan): {prior_no_rc}")
    print(f"\n=== 4. IMAGE GRID ===")
    print(f"  RC cases whose earlier scan has the same size and spacing: {same_grid} of {with_prior}"
          " (others need registration)")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-case table: {args.out}")


if __name__ == "__main__":
    main()
