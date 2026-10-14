"""
Is the Primus experiment worth it?  Fold-0 comparison: ResEnc-L vs Primus vs their probability-average ensemble.

Uses the same post-processing as the submission (t=0.50 for WT/TC/ET, t_RC=0.45, RC components >= 100 voxels)
and the same npz/GT alignment as tune_resencL_v3_folds0123_rc.py.

Per model it reports:
  * WT/TC/ET/RC subject-wise Dice (RC averaged only over cases with RC in GT or prediction, as before)
  * RC lesion detection: TP / FN / FP components and lesion-level F1  (closer to the BraTS Instance F1)
  * RC false-alarm cases: cases with no RC in GT but RC predicted   (invisible in the old Dice-only metric)
and a paired bootstrap (resampling cases) for the difference in RC Dice and RC lesion F1 vs ResEnc-L.

usage:
  python compare_rc_fold0.py --resenc <ResEncL fold_0/validation> --primus <Primus fold_0/validation> \
                             --gt /data/nnUNet_preprocessed/Dataset101_BraTSMET/gt_segmentations
"""
import argparse
import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label

T_WT = T_TC = T_ET = 0.50
T_RC = 0.45
MIN_RC = 100
N_BOOT = 2000
warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty bootstrap slices


def load_probs(f):
    with np.load(f) as z:
        key = "probabilities" if "probabilities" in z else ("softmax" if "softmax" in z else z.files[0])
        return z[key].astype(np.float32)


def remove_small(mask, min_size):
    lab, n = cc_label(mask)
    if n == 0 or min_size <= 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_size
    keep[0] = False
    return keep[lab]


def make_prediction(p):
    wt, tc, et, rc = p[0] > T_WT, p[1] > T_TC, p[2] > T_ET, p[3] > T_RC
    et &= wt
    rc = remove_small(rc & wt, MIN_RC)
    tc = (tc | et | rc) & wt
    pred = np.zeros(wt.shape, np.uint8)
    pred[wt] = 2
    pred[tc] = 1
    pred[et] = 3
    pred[rc] = 4
    return pred


def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else float(2 * np.logical_and(a, b).sum() / s)


def rc_lesions(pred_rc, gt_rc):
    """Component-level matching: a GT lesion is detected if any predicted RC voxel overlaps it."""
    gl, ng = cc_label(gt_rc)
    pl, npred = cc_label(pred_rc)
    tp = len(np.setdiff1d(np.unique(gl[pred_rc]), [0])) if ng else 0
    matched_pred = len(np.setdiff1d(np.unique(pl[gt_rc]), [0])) if npred else 0
    return tp, ng - tp, npred - matched_pred


def score(pred, gt):
    r = {
        "WT": dice(np.isin(pred, [1, 2, 3, 4]), np.isin(gt, [1, 2, 3, 4])),
        "TC": dice(np.isin(pred, [1, 3, 4]), np.isin(gt, [1, 3, 4])),
        "ET": dice(pred == 3, gt == 3),
        "RC": dice(pred == 4, gt == 4),
    }
    r["tp"], r["fn"], r["fp"] = rc_lesions(pred == 4, gt == 4)
    r["gt_has_rc"] = bool((gt == 4).any())
    r["pred_has_rc"] = bool((pred == 4).any())
    return r


def f1(rows):
    tp, fn, fp = (sum(r[k] for r in rows) for k in ("tp", "fn", "fp"))
    return 2 * tp / (2 * tp + fn + fp) if (2 * tp + fn + fp) else np.nan


def summary(rows):
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in ("WT", "TC", "ET", "RC")}
    out["RC_lesion_F1"] = f1(rows)
    out["RC_TP/FN/FP"] = "/".join(str(sum(r[k] for r in rows)) for k in ("tp", "fn", "fp"))
    out["RC_false_alarm_cases"] = sum(r["pred_has_rc"] and not r["gt_has_rc"] for r in rows)
    return out


def paired_bootstrap(a_rows, b_rows, rng):
    n = len(a_rows)
    d_dice, d_f1 = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a = [a_rows[i] for i in idx]
        b = [b_rows[i] for i in idx]
        d_dice.append(np.nanmean([x["RC"] for x in b]) - np.nanmean([x["RC"] for x in a]))
        d_f1.append(f1(b) - f1(a))
    ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    return ci(d_dice), ci(d_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resenc", required=True, type=Path)
    ap.add_argument("--primus", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--out", default="compare_rc_fold0.json")
    args = ap.parse_args()

    cases = sorted({f.stem for f in args.resenc.glob("*.npz")} & {f.stem for f in args.primus.glob("*.npz")})
    print(f"{len(cases)} common fold-0 validation cases", flush=True)
    if not cases:
        raise RuntimeError("No common npz files - did both runs use --npz?")

    results = {"ResEncL": [], "Primus": [], "Ensemble": []}
    for i, case in enumerate(cases, 1):
        pa, pb = load_probs(args.resenc / f"{case}.npz"), load_probs(args.primus / f"{case}.npz")
        gt = nib.load(str(args.gt / f"{case}.nii.gz")).get_fdata().astype(np.uint8)
        if gt.shape != pa.shape[1:]:
            gt = np.transpose(gt, (2, 1, 0))
        assert gt.shape == pa.shape[1:] == pb.shape[1:], f"{case}: shape mismatch"
        results["ResEncL"].append(score(make_prediction(pa), gt))
        results["Primus"].append(score(make_prediction(pb), gt))
        results["Ensemble"].append(score(make_prediction((pa + pb) / 2), gt))
        if i % 25 == 0:
            print(f"processed {i}/{len(cases)}", flush=True)

    report = {name: summary(rows) for name, rows in results.items()}
    rng = np.random.default_rng(0)
    for name in ("Primus", "Ensemble"):
        (dl, dh), (fl, fh) = paired_bootstrap(results["ResEncL"], results[name], rng)
        report[name]["RC_Dice_diff_vs_ResEncL_95CI"] = [round(dl, 4), round(dh, 4)]
        report[name]["RC_F1_diff_vs_ResEncL_95CI"] = [round(fl, 4), round(fh, 4)]

    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
