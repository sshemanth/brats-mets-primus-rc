"""
RC detection study on fold 0 (no GPU needed) - version 2.
Changes vs v1: GT cavities < 27 mm3 are ignored for detection (like the official rule),
the "RC must be inside WT" rule is tested on AND off, and missed/found cavities are split by case ID group.

Part 1 - sweep: for each model (ResEnc-L, Primus, Ensemble) try every RC threshold x minimum component size
         and report RC Dice, RC lesion F1, TP/FN/FP and false-alarm cases.
         WT/TC/ET are not affected by these two settings, so they are not recomputed.
Part 2 - missed cavities: at the submission setting (t=0.45, min 100 voxels) on the Ensemble, list every
         ground-truth cavity with its size (voxels and mm3), whether it was found, and the highest
         predicted RC probability inside it. This tells us WHY cavities are missed:
            max prob < 0.10           -> the model does not see it at all
            0.10 <= max prob < 0.45   -> a lower threshold could recover it
            max prob >= 0.45 but missed -> removed by the min-size filter or the "inside WT" rule

Same npz/GT alignment and rules as compare_rc_fold0.py (RC kept only inside WT, WT threshold 0.5).

usage:
  python rc_sweep_fold0.py --resenc /data/resenc_fold0_validation --primus $FOLD0/validation \
      --gt /data/nnUNet_preprocessed/Dataset101_BraTSMET/gt_segmentations --out_dir ~/logs/rc_sweep
"""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
WT_RULES = [True, False]          # True: RC kept only inside WT (current). False: RC kept anywhere.
MIN_GT_MM3 = 27.0                 # GT cavities smaller than this are ignored for detection (like the official rule)
MIN_SIZES = [0, 25, 50, 100, 200]
T_WT = 0.50
BASE_T, BASE_MIN = 0.45, 100


def load_probs(f):
    with np.load(f) as z:
        key = "probabilities" if "probabilities" in z else ("softmax" if "softmax" in z else z.files[0])
        return z[key].astype(np.float32)


def new_stats():
    return {(r, t, m): {"dice": [], "tp": 0, "fn": 0, "fp": 0, "fa": 0}
            for r in WT_RULES for t in THRESHOLDS for m in MIN_SIZES}


def sweep_case(p_rc, wt, gt_rc, gl, big, stats):
    """Label once per (rule, threshold), then derive every min-size setting from component sizes.
    big: ids of GT cavities >= MIN_GT_MM3 (only these count as TP/FN)."""
    gt_vox = int(gt_rc.sum())
    ng = len(big)
    for r in WT_RULES:
      for t in THRESHOLDS:
        pl, n = cc_label((p_rc > t) & wt if r else (p_rc > t))
        sizes = np.bincount(pl.ravel(), minlength=n + 1)
        inter = np.bincount(pl[gt_rc], minlength=n + 1)
        both = (pl > 0) & gt_rc
        pairs = np.unique(np.stack([pl[both], gl[both]], 1), axis=0) if both.any() else np.zeros((0, 2), int)
        for m in MIN_SIZES:
            keep = sizes >= m
            keep[0] = False
            pred_vox, inter_vox = int(sizes[keep].sum()), int(inter[keep].sum())
            s = stats[(r, t, m)]
            if pred_vox + gt_vox > 0:
                s["dice"].append(2 * inter_vox / (pred_vox + gt_vox))
            hit = np.unique(pairs[keep[pairs[:, 0]], 1]) if len(pairs) else np.array([], int)
            tp = int(np.isin(hit, big).sum())
            s["tp"] += tp
            s["fn"] += ng - tp
            s["fp"] += int((keep & (inter == 0)).sum())
            s["fa"] += int(gt_vox == 0 and pred_vox > 0)


def summarise(stats):
    rows = []
    for (r, t, m), s in stats.items():
        denom = 2 * s["tp"] + s["fn"] + s["fp"]
        rows.append({"wt_rule": "on" if r else "off", "thr": t, "min_size": m, "rc_dice": float(np.mean(s["dice"])) if s["dice"] else float("nan"),
                     "lesion_f1": 2 * s["tp"] / denom if denom else float("nan"),
                     "tp": s["tp"], "fn": s["fn"], "fp": s["fp"], "false_alarm_cases": s["fa"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resenc", required=True, type=Path)
    ap.add_argument("--primus", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted({f.stem for f in args.resenc.glob("*.npz")} & {f.stem for f in args.primus.glob("*.npz")})
    print(f"{len(cases)} common cases", flush=True)
    stats = {name: new_stats() for name in ("ResEncL", "Primus", "Ensemble")}
    lesions = []

    for i, case in enumerate(cases, 1):
        pa, pb = load_probs(args.resenc / f"{case}.npz"), load_probs(args.primus / f"{case}.npz")
        img = nib.load(str(args.gt / f"{case}.nii.gz"))
        gt = np.asarray(img.dataobj).astype(np.uint8)
        if gt.shape != pa.shape[1:]:
            gt = np.transpose(gt, (2, 1, 0))
        assert gt.shape == pa.shape[1:] == pb.shape[1:], f"{case}: shape mismatch"
        vox_mm3 = float(np.prod(img.header.get_zooms()[:3]))
        gt_rc = gt == 4
        gl, ng = cc_label(gt_rc)
        gsz = np.bincount(gl.ravel(), minlength=ng + 1) * vox_mm3
        big = np.nonzero(gsz >= MIN_GT_MM3)[0]
        big = big[big > 0]

        pe = (pa + pb) / 2
        for name, p in (("ResEncL", pa), ("Primus", pb), ("Ensemble", pe)):
            sweep_case(p[3], p[0] > T_WT, gt_rc, gl, big, stats[name])

        # missed-cavity analysis on the Ensemble at the submission setting
        if ng:
            wt = pe[0] > T_WT
            pl, n = cc_label((pe[3] > BASE_T) & wt)
            sizes = np.bincount(pl.ravel(), minlength=n + 1)
            kept = np.isin(pl, np.nonzero(sizes >= BASE_MIN)[0][1:])
            for k in big:
                les = gl == k
                lesions.append({
                    "case": case, "lesion": int(k), "voxels": int(les.sum()),
                    "mm3": round(float(les.sum()) * vox_mm3, 1),
                    "found": bool((kept & les).any()),
                    "max_prob": round(float(pe[3][les].max()), 3),
                    "max_prob_in_wt": round(float(np.where(wt, pe[3], 0)[les].max()), 3),
                })
        if i % 25 == 0:
            print(f"processed {i}/{len(cases)}", flush=True)

    # ---------- Part 1: sweep ----------
    print(f"\n=== PART 1: sweep; detection counts only GT cavities >= {MIN_GT_MM3:g} mm3 "
          "(* = current setting, + = current setting without the WT rule) ===")
    for name, s in stats.items():
        rows = summarise(s)
        with open(args.out_dir / f"sweep_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        base = next(r for r in rows if r["wt_rule"] == "on" and r["thr"] == BASE_T and r["min_size"] == BASE_MIN)
        nowt = next(r for r in rows if r["wt_rule"] == "off" and r["thr"] == BASE_T and r["min_size"] == BASE_MIN)
        print(f"\n{name}")
        print("   WT  thr  min | RC Dice | lesion F1 | TP  FN  FP | false-alarm cases")
        for r in sorted(rows, key=lambda r: -r["lesion_f1"])[:8] + [base, nowt]:
            star = "*" if r is base else ("+" if r is nowt else " ")
            print(f" {star}{r['wt_rule']:>3} {r['thr']:.2f} {r['min_size']:4d} |  {r['rc_dice']:.3f}  |   {r['lesion_f1']:.3f}   |"
                  f" {r['tp']:3d} {r['fn']:3d} {r['fp']:3d} | {r['false_alarm_cases']}")

    # ---------- Part 2: missed cavities ----------
    with open(args.out_dir / "lesions_ensemble.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lesions[0].keys())
        w.writeheader()
        w.writerows(lesions)
    missed = [l for l in lesions if not l["found"]]
    found = [l for l in lesions if l["found"]]
    print(f"\n=== PART 2: Ensemble at t={BASE_T}, min={BASE_MIN}: {len(found)} found, {len(missed)} missed ===")
    for grp, test in (("IDs < 01000", lambda c: int(c.split("-")[2]) < 1000),
                      ("IDs >= 01000", lambda c: int(c.split("-")[2]) >= 1000)):
        nf = sum(test(l["case"]) for l in found)
        nm = sum(test(l["case"]) for l in missed)
        print(f"  {grp:13s}: found {nf:3d}, missed {nm:3d}")
    bins = [(27, 100), (100, 1000), (1000, 10000), (10000, 1e12)]
    print("  size (mm3)        found  missed")
    for lo, hi in bins:
        nf = sum(lo <= l["mm3"] < hi for l in found)
        nm = sum(lo <= l["mm3"] < hi for l in missed)
        print(f"  {lo:>6.0f} - {hi if hi < 1e12 else 'inf':>6} {nf:6d} {nm:7d}")
    print("  why missed (highest RC probability inside the cavity):")
    print(f"    < 0.10  (model blind)               : {sum(l['max_prob'] < 0.10 for l in missed)}")
    print(f"    0.10-0.45 (lower threshold may help): {sum(0.10 <= l['max_prob'] < 0.45 for l in missed)}")
    print(f"    >= 0.45 but outside WT              : {sum(l['max_prob'] >= 0.45 and l['max_prob_in_wt'] < 0.45 for l in missed)}")
    print(f"    >= 0.45 inside WT (min-size filter) : {sum(l['max_prob_in_wt'] >= 0.45 for l in missed)}")
    print(f"\nsaved CSVs in {args.out_dir}")


if __name__ == "__main__":
    main()
