# Primus + RC-Focused Sampling for Resection Cavity Segmentation in Brain Metastasis MRI

[![BraTS 2026](https://img.shields.io/badge/BraTS-2026-blue)](https://www.synapse.org/brats)
[![nnU-Net v2](https://img.shields.io/badge/nnU--Net-v2-orange)](https://github.com/MIC-DKFZ/nnUNet)
[![Primus](https://img.shields.io/badge/Model-Primus%20V3--S-purple)](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

A follow-up to our BraTS 2026 challenge framework
**LAResSeg** ([**sshemanth/brain-tumor-segmentation**](https://github.com/sshemanth/brain-tumor-segmentation)), focused on the weakest region:
the **Resection Cavity (RC)**. We train a transformer-centric **Primus V3-S** model with the same
region-aware loss plus **RC-focused patch sampling**, and **ensemble** it with the Residual Encoder U-Net.

> All results are on **one cross-validation fold (fold 0, 260 held-out training cases)**.
> They are **not** comparable to the official validation-platform numbers in the challenge repository
> (different cases and a different evaluation).

---

## ✨ Highlights

- 🤖 **Primus V3-S** (transformer-centric 3D segmentation) inside nnU-Net v2, with the region-aware Tversky–BCE loss
- 🎯 **RC-focused sampling**: cavity cases drawn 2× more often; half of forced-foreground patches centred on a cavity
- 🔁 **CNN + transformer ensemble** (ResEnc-L + Primus) in probability space
- 🔍 **Detection-oriented analysis**: lesion-level RC F1, missed-cavity analysis by size and cause
- 🧪 **Patient-leakage and longitudinal analysis** with patient-grouped 5-fold splits for future work
- 🐳 **Tested Docker container** that keeps the exact input geometry (native space)

---

## 🏗️ Method

| Stage | Component |
|---|---|
| **Input** | 4-channel MRI: T1W, T1C, T2W, FLAIR |
| **Model A** | Residual Encoder U-Net (Large), region-aware loss *(from the challenge repo)* |
| **Model B** | Primus V3-S, same loss, AdamW (lr 3e-4, wd 5e-2), 50-epoch warm-up, no deep supervision |
| **RC sampling** | Case weight 2.0 for cavity cases; P(cavity-centred patch \| forced foreground) = 0.5 |
| **Ensemble** | Average region probabilities of A and B |
| **Post-processing** | WT/TC/ET @ 0.50 · **RC @ 0.20** · RC **not** forced inside WT · RC components **< 200 voxels removed** · labels written `[2, 1, 3, 4]` |

Why a low RC threshold? The ensemble averages two models. A cavity seen by only one model gets an
averaged probability below 0.45 and is lost; a lower threshold keeps it.

---

## 📈 Results (fold 0)

**Dice per region** — nnU-Net default evaluation (threshold 0.5, no post-processing)

| Model | WT | TC | ET | RC | Mean |
|---|---|---|---|---|---|
| Standard nnU-Net (baseline) | 0.747 | 0.752 | 0.721 | 0.184 | 0.601 |
| ResEnc-L + region-aware loss | 0.760 | 0.768 | 0.740 | 0.428 | 0.674 |
| **Primus V3-S + RC sampling** | 0.760 | 0.767 | 0.734 | **0.495** | **0.689** |

**RC detection** — real cavities ≥ 27 mm³, lesion-level matching

| Setting | RC Dice | Lesion F1 | Found / Missed / False |
|---|---|---|---|
| ResEnc-L, challenge post-processing | 0.464 | 0.741 | 30 / 15 / 6 |
| **Ensemble, new post-processing** | **0.505** | **0.800** | **34 / 11 / 6** |

🏆 Ensemble vs ResEnc-L, RC Dice **+0.032** (bootstrap 95% CI **0.005 – 0.065**).
Primus alone: +0.031, CI includes 0 on one fold.

<details>
<summary><b>Why are cavities still missed?</b></summary>

| Cavity size | Found | Missed |
|---|---|---|
| 27 – 100 mm³ | 3 | 3 |
| 100 – 1,000 mm³ | 8 | 10 |
| 1,000 – 10,000 mm³ | 13 | 1 |
| > 10,000 mm³ | 7 | 0 |

- Large cavities are almost always found; **medium cavities** are the problem.
- 10 of 14 missed cavities have a predicted probability **below 0.1** — no threshold can recover them.
- 3 were found by the model but removed by the "RC inside WT" rule (now dropped).

</details>

<details>
<summary><b>Longitudinal finding (future work)</b></summary>

- **107 of 168** cavity cases have an **earlier scan** of the same patient.
- Earlier and current scans are **already aligned** (same size, spacing, origin and direction).
- The earlier cavity overlaps the current one: **median Dice 0.52**.
- **9 of 13** missed fold-0 cavities have an earlier scan that contains a cavity.

This suggests using the patient's earlier scan as a guide for the current one.
</details>

---

## ⚠️ Limitations

- One fold only; post-processing was tuned and tested on the **same** fold, so tuned numbers are slightly optimistic.
- The original 5-fold splits were **not grouped by patient**: 147 of 260 fold-0 validation cases come from patients
  with another time point in training. Those cases did **not** score higher than unseen patients, so we found no
  evidence that this inflated the results. Patient-grouped splits are in `results/fold0/splits_final_grouped.json`.
- No ablation separating the effect of Primus from the effect of RC sampling.
- No official validation score (the platform was closed).

---

## 📁 Repository Layout

| Folder | Contents |
|---|---|
| `nnunet_trainers/` | Primus V3-S + region-aware loss + RC sampling trainer, and the ResEnc-L trainer |
| `scripts/training/` | Launcher, watchdog (auto-resume from checkpoint), memory monitor, VM setup |
| `scripts/analysis/` | Model comparison with bootstrap CIs, RC threshold sweeps, leakage and longitudinal checks, grouped-split maker |
| `scripts/inference/` | Ensemble + post-processing, validation-set prediction |
| `docker/` | `Dockerfile`, `infer.py` (entry point), `build_docker.sh` |
| `results/fold0/` | Per-region summaries, comparison JSON, sweep and per-cavity CSVs, longitudinal table, grouped splits |

MRI data, labels, checkpoints and predictions are **not** included (data licence and file size).
Some shell scripts contain paths from our training machine; adjust them before use.

---

## 🚀 Getting Started

Install nnU-Net v2 and dynamic-network-architectures **from GitHub** (PyPI releases may not include Primus V3):

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git && pip install -e nnUNet
pip install git+https://github.com/MIC-DKFZ/dynamic-network-architectures.git
cp nnunet_trainers/*.py nnUNet/nnunetv2/training/nnUNetTrainer/
```

**Train** (reuses the ResEnc-L plans and preprocessed data):

```bash
nnUNetv2_train <DATASET_ID> 3d_fullres 0 \
    -tr nnUNetTrainerPrimusV3S_RegionAwareRC -p nnUNetResEncUNetLPlans --npz
```

**Ensemble + post-processing** from two nnU-Net probability folders:

```bash
python scripts/inference/ensemble_submit.py --resenc <RESENC_PRED> --primus <PRIMUS_PRED> \
    --images <INPUT_IMAGES> --out <OUTPUT_FOLDER>
```

**Docker** (inputs named like `BraTS-MET-00833-000-t1c.nii.gz`):

```bash
docker run --rm --gpus all --network none \
    -v <INPUT>:/input:ro -v <OUTPUT>:/output brats-mets-ensemble:fold0
```

---

## ⚙️ Implementation Details

| Setting | Value |
|---|---|
| Primus model | Primus V3-S (`nnUNet_PrimusV3S_Trainer`) |
| Optimizer | AdamW, lr 3 × 10⁻⁴, weight decay 5 × 10⁻², 50-epoch linear warm-up, polynomial decay |
| Epochs | 1000 |
| Batch size / patch size | 2 / 160 × 224 × 192 |
| Foreground oversampling | 0.60 |
| Training time | ≈ 450 s per epoch, ≈ 5 days on 1× NVIDIA L4 (24 GB) |

---

## 🙏 Acknowledgements

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) (Isensee et al., *Nature Methods*, 2021) and
[Primus](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md) (Wald et al., 2025).
Data from the [BraTS 2026 Brain Metastases Challenge](https://www.synapse.org/brats).
