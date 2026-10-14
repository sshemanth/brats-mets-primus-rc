"""
BraTS-METS 2026 inference container entrypoint.
Model: ensemble of two nnU-Net v2 models trained on fold 0
         - ResEnc-L  (trainer nnUNetTrainerCoreCavityV3ResEncL)
         - Primus V3-S + RC-focused sampling (trainer nnUNetTrainerPrimusV3S_RegionAwareRC)
Output: one label map per case, labels 1 = NETC, 2 = SNFH/edema, 3 = ET, 4 = RC,
        saved with exactly the same size/spacing/origin/direction as the input T1c image.

Accepted input layouts (both are found automatically):
    /input/BraTS-MET-00833-000-t1c.nii.gz              (flat folder)
    /input/BraTS-MET-00833-000/BraTS-MET-00833-000-t1c.nii.gz   (one folder per case)
Output file name: /output/<case>.nii.gz   e.g. /output/BraTS-MET-00833-000.nii.gz

Run:
    docker run --rm --gpus all --network none \
        -v /path/to/input:/input:ro -v /path/to/output:/output brats-mets-ensemble:fold0
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label

# ----------------------------------------------------------------------------------------------
# Settings. Region order of the network outputs is WT, TC, ET, RC (from dataset.json "regions").
# These post-processing values were chosen on the fold-0 held-out set (see project README).
# ----------------------------------------------------------------------------------------------
MODELS = {  # name -> folder inside the image (contains plans.json, dataset.json, fold_0/checkpoint_final.pth)
    "resencl": "/opt/models/resencl",
    "primus": "/opt/models/primus",
}
T_WT = T_TC = T_ET = 0.50   # thresholds on the averaged probabilities
T_RC = 0.20                 # low RC threshold: keeps cavities that only ONE of the two models sees
MIN_RC_VOXELS = 200         # RC connected components smaller than this are removed
RC_INSIDE_WT = False        # False: RC is not forced to lie inside WT (this recovered missed cavities)

# nnU-Net input channel order (dataset.json channel_names: 0 T1, 1 T1c, 2 T2, 3 FLAIR)
CHANNELS = {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"}


def find_cases(input_dir: Path) -> dict:
    """Return {case_id: {modality: path}} for every case that has a T1c image."""
    cases = {}
    for t1c in sorted(input_dir.rglob("*-t1c.nii.gz")):
        case = t1c.name[: -len("-t1c.nii.gz")]
        cases[case] = {m: t1c.parent / f"{case}-{m}.nii.gz" for m in CHANNELS.values()}
    return cases


def stage_nnunet_input(cases: dict, stage_dir: Path):
    """Create nnU-Net style file names (<case>_0000.nii.gz ...) as symlinks to the original files.
    Nothing is copied or resampled, so the native geometry is untouched."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    for case, files in cases.items():
        for idx, mod in CHANNELS.items():
            src = files[mod]
            if not src.exists():
                # BraTS-METS made T2W optional. The models were trained with all four inputs; as a crude
                # fallback we feed FLAIR in the T2 slot so the case still gets a prediction.
                print(f"WARNING {case}: {mod} missing, using t2f instead", flush=True)
                src = files["t2f"]
            os.symlink(src.resolve(), stage_dir / f"{case}_{idx}.nii.gz")


def run_model(model_dir: str, in_dir: Path, out_dir: Path, device_name: str):
    """Predict all staged cases with one model and save region probabilities (.npz).
    The sequential predictor uses no worker processes, so it also works with Docker's small default /dev/shm."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    out_dir.mkdir(parents=True, exist_ok=True)   # the sequential predictor does not create its output folder
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                device=torch.device(device_name), verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    predictor.predict_from_files_sequential(str(in_dir), str(out_dir), save_probabilities=True, overwrite=True)
    del predictor
    if device_name == "cuda":
        torch.cuda.empty_cache()


def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    lab, n = cc_label(mask)
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_size
    keep[0] = False
    return keep[lab]


def make_labels(p: np.ndarray) -> np.ndarray:
    """Averaged region probabilities (4, z, y, x) -> BraTS label map (z, y, x)."""
    wt, tc, et, rc = p[0] > T_WT, p[1] > T_TC, p[2] > T_ET, p[3] > T_RC
    et &= wt                                  # enhancing tumour must be inside the whole tumour
    if RC_INSIDE_WT:
        rc &= wt
    rc = remove_small(rc, MIN_RC_VOXELS)
    tc = (tc | et) & wt                       # tumour core contains ET and lies inside WT
    out = np.zeros(wt.shape, np.uint8)
    out[wt] = 2                               # written in the order 2 -> 1 -> 3 -> 4,
    out[tc] = 1                               # later labels overwrite earlier ones
    out[et] = 3
    out[rc] = 4
    return out


def load_probs(f: Path) -> np.ndarray:
    with np.load(f) as z:
        return z["probabilities"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input", type=Path)
    ap.add_argument("--output", default="/output", type=Path)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    cases = find_cases(args.input)
    print(f"found {len(cases)} cases in {args.input}, device={device}", flush=True)
    if not cases:
        raise SystemExit("no *-t1c.nii.gz files found")

    work = Path(tempfile.mkdtemp(prefix="brats_"))
    try:
        stage = work / "nnunet_input"
        stage_nnunet_input(cases, stage)

        # 1) run each model, keeping its probability maps
        for name, model_dir in MODELS.items():
            print(f"running {name}", flush=True)
            run_model(model_dir, stage, work / name, device)

        # 2) average, post-process, write in the exact geometry of the original T1c
        for i, case in enumerate(sorted(cases), 1):
            probs = [load_probs(work / name / f"{case}.npz") for name in MODELS]
            labels = make_labels(np.mean(probs, axis=0))
            ref = sitk.ReadImage(str(cases[case]["t1c"]))
            if labels.shape != sitk.GetArrayFromImage(ref).shape:
                raise RuntimeError(f"{case}: prediction shape {labels.shape} does not match the input image")
            img = sitk.GetImageFromArray(labels)
            img.CopyInformation(ref)              # same size, spacing, origin and direction as the input
            sitk.WriteImage(img, str(args.output / f"{case}.nii.gz"), useCompression=True)
            print(f"[{i}/{len(cases)}] wrote {case}.nii.gz", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)  # remove temporary probability files

    print("done", flush=True)


if __name__ == "__main__":
    main()
