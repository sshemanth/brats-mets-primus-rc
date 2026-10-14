# Primus + RC-focused sampling for resection cavity segmentation (BraTS-METS 2026)

Brain-metastasis MRI segmentation of four regions (WT, TC, ET and **RC, the resection cavity**) with nnU-Net v2.
This project tries to improve **RC**, the weakest region, with a transformer model (**Primus V3-S**),
**RC-focused patch sampling**, and an **ensemble** with a ResEnc-L CNN.

> Builds on our BraTS-METS 2026 challenge code: [sshemanth/brats2026](https://github.com/sshemanth/brats2026).
> Done as a UTS 32933 project. All results are on **one cross-validation fold (fold 0, 260 cases)**.

## Main results (fold 0)

**Dice per region** (nnU-Net default evaluation: threshold 0.5, no post-processing)

| Model | WT | TC | ET | RC | Mean |
|---|---|---|---|---|---|
| ResEnc-L (baseline) | 0.760 | 0.768 | 0.740 | 0.428 | 0.674 |
| Primus V3-S + RC sampling | 0.760 | 0.767 | 0.734 | **0.495** | **0.689** |

**RC detection** (real cavities ≥ 27 mm³, lesion-level matching)

| Setting | RC Dice | Lesion F1 | Found / missed / false |
|---|---|---|---|
| ResEnc-L, old post-processing | 0.464 | 0.741 | 30 / 15 / 6 |
| **Ensemble, final post-processing** | **0.505** | **0.800** | **34 / 11 / 6** |

Ensemble vs ResEnc-L, RC Dice: **+0.032** (bootstrap 95% CI 0.005 to 0.065).

**Final post-processing:** average both models' region probabilities → WT/TC/ET threshold 0.5 →
RC threshold **0.20**, RC **not** forced inside WT, RC components **< 200 voxels removed** →
labels written in the order 2 → 1 → 3 → 4.

## What I learned

- **All of the gain is in RC.** WT, TC and ET stay the same.
- **Cavities are mostly missed, not invented.** False cavities are rare (about 6); missed ones are the problem.
- **Size matters.** Large cavities (> 1,000 mm³) are almost always found. About 10 medium cavities
  (100–1,000 mm³) get a probability below 0.1, so no threshold can recover them.
- **Earlier scans can help (future work).** 107 of 168 cavity cases have an earlier scan of the same patient.
  The scans are already aligned, and the earlier cavity overlaps the current one (median Dice 0.52).
  9 of 13 missed fold-0 cavities have an earlier scan with a cavity.

## Limitations

- One fold only, and post-processing was tuned and tested on the **same** fold, so tuned numbers are slightly optimistic.
- The original splits were **not grouped by patient**: 147 of 260 fold-0 validation cases come from patients with
  another time point in training. These cases did not score higher than unseen patients, so we found no evidence
  that this inflated the results. Patient-grouped splits are provided for future work (`results/fold0/splits_final_grouped.json`).
- No ablation separating the effect of Primus from the effect of RC sampling.
- The official validation platform was closed, so there is no leaderboard score for this model.

## Repository layout

| Folder | What is inside |
|---|---|
| `nnunet_trainers/` | Custom nnU-Net trainers: Primus V3-S + region-aware Tversky–BCE loss + RC sampling, and the ResEnc-L baseline |
| `scripts/training/` | Launcher, watchdog (auto-restart from checkpoint), memory monitor, VM setup |
| `scripts/analysis/` | Fold-0 comparison with bootstrap CIs, RC threshold sweeps, patient-leakage and longitudinal checks, grouped-split maker |
| `scripts/inference/` | Ensemble + post-processing, prediction of the validation set |
| `docker/` | `Dockerfile`, `infer.py` (container entry point), `build_docker.sh` |
| `results/fold0/` | Per-region summaries, comparison JSON, sweep and per-cavity CSVs, longitudinal table, grouped splits |

MRI data, labels, model checkpoints and predictions are **not** included (data licence and file size).
Some shell scripts contain paths from our training machine; change them before use.

## How to use

**Train** (nnU-Net v2 and dynamic-network-architectures installed from GitHub):

```bash
cp nnunet_trainers/*.py <nnUNet>/nnunetv2/training/nnUNetTrainer/
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerPrimusV3S_RegionAwareRC -p nnUNetResEncUNetLPlans --npz
```

**Run the ensemble with Docker** (inputs named like `BraTS-MET-00833-000-t1c.nii.gz`):

```bash
docker run --rm --gpus all --network none \
  -v <input>:/input:ro -v <output>:/output brats-mets-ensemble:fold0
```

Outputs keep exactly the size, spacing, origin and direction of the input images.

## Acknowledgements

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [Primus](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md) by MIC-DKFZ.
- Data from the BraTS-METS 2026 challenge.
