import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class AsymmetricRegionTverskyBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.register_buffer("region_weights", torch.tensor([1.0, 2.5, 1.2, 3.0], dtype=torch.float32))
        self.register_buffer("alpha_fp", torch.tensor([0.45, 0.90, 0.55, 0.55], dtype=torch.float32))
        self.register_buffer("beta_fn", torch.tensor([0.55, 0.10, 0.45, 0.45], dtype=torch.float32))
        self.register_buffer("bce_pos_weight", torch.tensor([1.0, 0.75, 1.0, 3.0], dtype=torch.float32))
        self.register_buffer("bce_neg_weight", torch.tensor([1.0, 2.50, 1.25, 0.75], dtype=torch.float32))

    def forward(self, net_output, target):
        target = target.float()
        if net_output.shape != target.shape:
            raise RuntimeError(f"Shape mismatch: prediction {net_output.shape}, target {target.shape}")

        c = net_output.shape[1]
        reduce_dims = (0,) + tuple(range(2, net_output.ndim))

        region_weights = self.region_weights[:c].to(net_output.device)
        alpha_fp = self.alpha_fp[:c].to(net_output.device)
        beta_fn = self.beta_fn[:c].to(net_output.device)
        bce_pos_weight = self.bce_pos_weight[:c].to(net_output.device)
        bce_neg_weight = self.bce_neg_weight[:c].to(net_output.device)

        view_shape = (1, c) + (1,) * (net_output.ndim - 2)

        bce = F.binary_cross_entropy_with_logits(net_output, target, reduction="none")
        voxel_weights = target * bce_pos_weight.view(view_shape) + (1.0 - target) * bce_neg_weight.view(view_shape)
        bce_per_region = (bce * voxel_weights).mean(dim=reduce_dims)

        probs = torch.sigmoid(net_output)
        tp = (probs * target).sum(dim=reduce_dims)
        fp = (probs * (1.0 - target)).sum(dim=reduce_dims)
        fn = ((1.0 - probs) * target).sum(dim=reduce_dims)

        tversky = (tp + self.smooth) / (tp + alpha_fp * fp + beta_fn * fn + self.smooth)
        tversky_loss = 1.0 - tversky

        loss_per_region = 0.75 * tversky_loss + 0.25 * bce_per_region
        return (loss_per_region * region_weights).sum() / region_weights.sum()


class nnUNetTrainerCoreCavityV3ResEncL(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.60

    def _build_loss(self):
        loss = AsymmetricRegionTverskyBCELoss()

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float32)
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
