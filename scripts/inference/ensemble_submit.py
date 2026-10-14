"""
Make BraTS-METS submission files from two nnU-Net probability folders (ResEnc-L fold 0 + Primus fold 0).

Final setting chosen on fold 0:
  - average the region probabilities of both models   (order WT, TC, ET, RC)
  - WT, TC, ET threshold 0.5 ; ET kept inside WT ; TC = (TC or ET) inside WT
  - RC threshold 0.20, NOT restricted to WT, RC components smaller than 200 voxels removed
  - labels written in the order 2 (ED), 1 (NETC), 3 (ET), 4 (RC); later labels overwrite earlier ones

Geometry is copied from the case's first input image with SimpleITK, so no axis juggling is needed.

usage:
  python ensemble_submit.py --resenc /data/val_pred/resenc --primus /data/val_pred/primus \
      --images /data/val_images --out /data/submission --zip /data/submission.zip
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label

T_WT = T_TC = T_ET = 0.50
T_RC = 0.20
MIN_RC = 200
RC_INSIDE_WT = False


def load_probs(f):
    with np.load(f) as z:
        key = "probabilities" if "probabilities" in z else z.files[0]
        return z[key].astype(np.float32)


def remove_small(mask, min_size):
    lab, n = cc_label(mask)
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_size
    keep[0] = False
    return keep[lab]


def make_labels(p):
    wt, tc, et, rc = p[0] > T_WT, p[1] > T_TC, p[2] > T_ET, p[3] > T_RC
    et &= wt
    if RC_INSIDE_WT:
        rc &= wt
    rc = remove_small(rc, MIN_RC)
    tc = (tc | et) & wt
    out = np.zeros(wt.shape, np.uint8)
    out[wt] = 2
    out[tc] = 1
    out[et] = 3
    out[rc] = 4
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resenc", required=True, type=Path)
    ap.add_argument("--primus", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path, help="nnU-Net style input folder (<case>_0000.nii.gz ...)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--zip", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = sorted({f.stem for f in args.resenc.glob("*.npz")} & {f.stem for f in args.primus.glob("*.npz")})
    only_one = sorted({f.stem for f in args.resenc.glob("*.npz")} ^ {f.stem for f in args.primus.glob("*.npz")})
    print(f"{len(cases)} cases with both predictions", flush=True)
    if only_one:
        print(f"WARNING: {len(only_one)} cases have only one model's prediction: {only_one[:5]} ...", flush=True)

    for i, case in enumerate(cases, 1):
        pa, pb = load_probs(args.resenc / f"{case}.npz"), load_probs(args.primus / f"{case}.npz")
        assert pa.shape == pb.shape, f"{case}: shape mismatch {pa.shape} vs {pb.shape}"
        ref = sitk.ReadImage(str(args.images / f"{case}_0000.nii.gz"))
        labels = make_labels((pa + pb) / 2)
        assert labels.shape == sitk.GetArrayFromImage(ref).shape, f"{case}: label/image shape mismatch"
        img = sitk.GetImageFromArray(labels)
        img.CopyInformation(ref)
        sitk.WriteImage(img, str(args.out / f"{case}.nii.gz"), useCompression=True)
        if i % 20 == 0:
            print(f"written {i}/{len(cases)}", flush=True)

    print(f"done: {len(cases)} files in {args.out}", flush=True)
    if args.zip:
        shutil.make_archive(str(args.zip.with_suffix("")), "zip", args.out)
        print(f"zip: {args.zip}", flush=True)


if __name__ == "__main__":
    main()
