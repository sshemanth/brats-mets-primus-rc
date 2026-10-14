"""
For RC cases whose earlier scan also has a cavity:
  - are the two time points aligned (same origin and direction)?
  - how much does the earlier cavity overlap the current one (Dice)?
Also lists the earlier scans of the fold-0 missed cavities.

usage: python3.13 prior_overlap.py   (expects the files from longitudinal_check.py in /workspace/data)
"""
import csv

import numpy as np
import SimpleITK as s

D = "/workspace/data/gt_segmentations/"
CSV = "/workspace/data/longitudinal_check.csv"

rows = [r for r in csv.DictReader(open(CSV)) if r["prior_has_rc"] == "True"]
aligned, dices = 0, []
for r in rows:
    a = s.ReadImage(D + r["case"] + ".nii.gz")
    b = s.ReadImage(D + r["prior_case"] + ".nii.gz")
    same = (np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-2)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-3))
    aligned += same
    if same:
        x = s.GetArrayViewFromImage(a) == 4
        y = s.GetArrayViewFromImage(b) == 4
        dices.append(2 * (x & y).sum() / (x.sum() + y.sum()))

d = np.array(dices)
print(len(rows), "RC cases whose earlier scan has a cavity | same origin+direction:", aligned)
if len(d):
    print("Dice(earlier cavity, current cavity): median %.2f | >0.5: %d | >0: %d of %d"
          % (np.median(d), (d > 0.5).sum(), (d > 0).sum(), len(d)))

miss = ["01092-003", "01124-001", "01124-002", "01172-002", "01177-002", "01205-003", "01220-001",
        "01222-002", "01222-003", "01275-003", "01321-002", "01353-001", "01353-002"]
print("fold-0 missed cases -> earlier scan:")
for r in csv.DictReader(open(CSV)):
    if any(r["case"].endswith(m) for m in miss):
        print("  ", r["case"], "| prior:", r["prior_case"] or "none", "| prior has RC:", r["prior_has_rc"])
