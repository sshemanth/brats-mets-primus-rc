# BraTS-METS 2026 – resection cavity (RC) project

UTS 32933 project. Brain-metastasis MRI segmentation of WT, TC, ET and RC with nnU-Net v2.
Main model: ensemble of **ResEnc-L** and **Primus V3-S + RC-focused sampling**, both trained on fold 0.

## Folder layout
| Folder | What is inside |
|---|---|
| `01_code/training`  | Custom trainers (Primus + region-aware loss + RC sampling; ResEnc-L), launcher, watchdog, memory monitor, VM setup |
| `01_code/analysis`  | Fold-0 comparison (`compare_rc_fold0.py`) and RC threshold sweeps (v1, v2) |
| `01_code/inference` | Ensemble + post-processing (`ensemble_submit.py`) and validation-set prediction (`predict_val.sh`) |
| `01_code/docker`    | `Dockerfile`, `infer.py` (container entrypoint), `build_docker.sh` |
| `02_models/*_fold0` | `plans.json`, `dataset.json`, `fold_0/checkpoint_final.pth`, training logs, `progress.png` |
| `03_results`        | Fold-0 validation predictions (`.npz` + `summary.json`) for both models, comparison JSON, sweep CSVs |
| `04_validation_submission` | `submission.zip` for the official validation set (platform was closed) and raw predictions |
| `05_logs`           | All training / analysis logs |
| `06_docker_image`   | `brats_mets_ensemble_fold0.tar.gz` (load with `gunzip -c ... \| docker load`) |

Data are **not** copied here: preprocessed data and raw images are in the existing bucket folders.

## Final setting (chosen on fold 0)
Average the region probabilities of both models → WT/TC/ET threshold 0.5 → **RC threshold 0.20**,
**RC not restricted to WT**, **RC components < 200 voxels removed** → labels written 2 → 1 → 3 → 4.

## Fold-0 results (260 held-out cases)
nnU-Net default evaluation (threshold 0.5, no post-processing):

| Model | WT | TC | ET | RC | Mean |
|---|---|---|---|---|---|
| ResEnc-L | 0.760 | 0.768 | 0.740 | 0.428 | 0.674 |
| Primus + RC sampling | 0.760 | 0.767 | 0.734 | 0.495 | 0.689 |

RC detection (cavities ≥ 27 mm³, our own lesion matching):

| Setting | RC Dice | Lesion F1 | Found / missed / false |
|---|---|---|---|
| ResEnc-L, old setting (0.45, min 100, RC inside WT) | 0.464 | 0.741 | 30 / 15 / 6 |
| Ensemble, final setting | 0.505 | 0.800 | 34 / 11 / 6 |

Ensemble vs ResEnc-L, RC Dice difference: +0.032 (bootstrap 95% CI 0.005 – 0.065).
Caveat: post-processing was tuned and tested on the same fold, so tuned numbers are slightly optimistic.

## Reproduce
- Train: `nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerPrimusV3S_RegionAwareRC -p nnUNetResEncUNetLPlans --npz`
  (copy the trainer into `nnunetv2/training/nnUNetTrainer/`; nnU-Net and dynamic-network-architectures from GitHub).
- Docker: `docker run --rm --gpus all --network none -v <input>:/input:ro -v <output>:/output brats-mets-ensemble:fold0`
