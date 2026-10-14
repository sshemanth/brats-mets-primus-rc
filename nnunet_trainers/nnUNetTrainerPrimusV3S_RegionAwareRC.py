"""
Primus V3-S trainer for BraTS-METS 2026 (Dataset101_BraTSMET) with
  1) the same asymmetric Tversky-BCE region loss as nnUNetTrainerCoreCavityV3ResEncL, and
  2) resection-cavity (RC) biased patch sampling.

Region order (from dataset.json): WT, TC, ET, RC  ->  RC is foreground region index 3.

Place this file anywhere under  nnunetv2/training/nnUNetTrainer/  and train with e.g.
    nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerPrimusV3S_RegionAwareRC -p nnUNetResEncUNetLPlans

Requires a recent nnU-Net master (Primus V3 trainers + foreground_locations API) and
dynamic-network-architectures installed from GitHub (PyPI releases may lack PrimusV3S).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _base_trainer_module
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.foreground_locations import normalize_class_key
from nnunetv2.training.nnUNetTrainer.primus.primus_trainers import nnUNet_PrimusV3S_Trainer


# --------------------------------------------------------------------------------------
# Loss: identical coefficients to the ResEnc-L submission (WT, TC, ET, RC)
# --------------------------------------------------------------------------------------
class AsymmetricRegionTverskyBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        self.register_buffer("region_weights", torch.tensor([1.0, 2.5, 1.2, 3.0]))
        self.register_buffer("alpha_fp", torch.tensor([0.45, 0.90, 0.55, 0.55]))
        self.register_buffer("beta_fn", torch.tensor([0.55, 0.10, 0.45, 0.45]))
        self.register_buffer("bce_pos_weight", torch.tensor([1.0, 0.75, 1.0, 3.0]))
        self.register_buffer("bce_neg_weight", torch.tensor([1.0, 2.50, 1.25, 0.75]))

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Primus has no deep supervision, so both arguments are plain tensors.
        # Compute in fp32: the Primus train_step runs the loss inside autocast.
        net_output = net_output.float()
        target = target.float()
        if net_output.shape != target.shape:
            raise RuntimeError(f"Shape mismatch: prediction {net_output.shape}, target {target.shape}")

        c = net_output.shape[1]
        dev = net_output.device
        reduce_dims = (0,) + tuple(range(2, net_output.ndim))
        view_shape = (1, c) + (1,) * (net_output.ndim - 2)

        w = self.region_weights[:c].to(dev)
        alpha = self.alpha_fp[:c].to(dev)
        beta = self.beta_fn[:c].to(dev)
        u_pos = self.bce_pos_weight[:c].to(dev).view(view_shape)
        u_neg = self.bce_neg_weight[:c].to(dev).view(view_shape)

        bce = F.binary_cross_entropy_with_logits(net_output, target, reduction="none")
        bce_per_region = (bce * (target * u_pos + (1.0 - target) * u_neg)).mean(dim=reduce_dims)

        probs = torch.sigmoid(net_output)
        tp = (probs * target).sum(dim=reduce_dims)
        fp = (probs * (1.0 - target)).sum(dim=reduce_dims)
        fn = ((1.0 - probs) * target).sum(dim=reduce_dims)
        tversky_loss = 1.0 - (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth)

        loss_per_region = 0.75 * tversky_loss + 0.25 * bce_per_region
        return (loss_per_region * w).sum() / w.sum()


# --------------------------------------------------------------------------------------
# RC-biased data loader
# --------------------------------------------------------------------------------------
class RCBiasedDataLoader(nnUNetDataLoader):
    """
    Two levers on top of the standard nnU-Net loader:
      * case level  : cases that contain RC are drawn `rc_case_weight` times as often
      * patch level : when a patch is forced to contain foreground, it is centred on an RC
                      voxel with probability `p_rc_patch` (if the case has RC)
    """

    def __init__(self, *args, rc_key=None, rc_case_weight: float = 2.0, p_rc_patch: float = 0.5,
                 log_fn=print, **kwargs):
        super().__init__(*args, **kwargs)
        self.rc_key = rc_key
        self.p_rc_patch = p_rc_patch

        fg = self._data.foreground_locations
        has_rc = np.array([fg.count(i, rc_key) > 0 for i in self.indices], dtype=bool)
        probs = np.where(has_rc, rc_case_weight, 1.0).astype(np.float64)
        self.sampling_probabilities = probs / probs.sum()
        log_fn(f"[RC sampler] {has_rc.sum()}/{len(has_rc)} training cases contain RC; "
               f"RC case weight {rc_case_weight}, P(RC-centred | forced fg) {p_rc_patch}, "
               f"share of draws from RC cases {self.sampling_probabilities[has_rc].sum():.2f}")

    def get_bbox(self, identifier, data_shape, force_fg, overwrite_class=None, verbose=False):
        if (force_fg and overwrite_class is None and np.random.uniform() < self.p_rc_patch
                and self._data.foreground_locations.count(identifier, self.rc_key) > 0):
            overwrite_class = self.rc_key
        return super().get_bbox(identifier, data_shape, force_fg, overwrite_class, verbose)


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
class nnUNetTrainerPrimusV3S_RegionAwareRC(nnUNet_PrimusV3S_Trainer):
    RC_REGION_INDEX = 3      # WT, TC, ET, RC
    RC_CASE_WEIGHT = 2.0
    P_RC_PATCH = 0.5

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Primus defaults are kept: AdamW, lr 3e-4, wd 5e-2, 50-epoch warmup, no deep supervision
        self.oversample_foreground_percent = 0.60
        self.save_every = 10

    def _build_loss(self):
        assert self.label_manager.has_regions, "This trainer expects region-based training (dataset.json regions)."
        return AsymmetricRegionTverskyBCELoss()

    def get_dataloaders(self):
        rc_key = normalize_class_key(self.label_manager.foreground_regions[self.RC_REGION_INDEX])
        trainer = self
        n_built = [0]

        def _factory(*args, **kwargs):
            # The base get_dataloaders builds the training loader first, then the validation loader.
            # Only the training loader gets RC biasing; validation stays unbiased.
            n_built[0] += 1
            if n_built[0] == 1:
                return RCBiasedDataLoader(*args, rc_key=rc_key, rc_case_weight=trainer.RC_CASE_WEIGHT,
                                          p_rc_patch=trainer.P_RC_PATCH, log_fn=trainer.print_to_log_file,
                                          **kwargs)
            return nnUNetDataLoader(*args, **kwargs)

        # temporarily swap the loader class the base trainer instantiates
        original = _base_trainer_module.nnUNetDataLoader
        _base_trainer_module.nnUNetDataLoader = _factory
        try:
            return super().get_dataloaders()
        finally:
            _base_trainer_module.nnUNetDataLoader = original


class nnUNetTrainerPrimusV3S_RegionAwareRC_P128(nnUNetTrainerPrimusV3S_RegionAwareRC):
    """Fallback if 160x224x192 runs out of memory on the L4 (all dims stay divisible by 8)."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        plans["configurations"][configuration]["patch_size"] = (128, 192, 160)
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainerPrimusV3S_RegionAware_NoRCBias(nnUNetTrainerPrimusV3S_RegionAwareRC):
    """Ablation: same loss and architecture, standard nnU-Net sampling."""
    RC_CASE_WEIGHT = 1.0
    P_RC_PATCH = 0.0
