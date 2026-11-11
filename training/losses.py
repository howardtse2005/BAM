"""Loss for boundary-band prediction.

Three terms:
  * BCE with positive weighting -- handles the ~4-6% positive rate.
  * Soft Dice -- stabilises early training when the prediction is near zero.
  * L1 -- sharpens the peak; without it the model predicts a wide, timid band.

All terms are masked by ``weight``, which carries the per-image quality weight
and zeroes padded pixels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BAMLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 6.0,
        w_bce: float = 0.5,
        w_dice: float = 0.5,
        w_l1: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)
        self.w_l1 = float(w_l1)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if logits.shape != target.shape:
            raise ValueError(
                f"shape mismatch: logits {tuple(logits.shape)} vs target {tuple(target.shape)}"
            )
        if weight is None:
            weight = torch.ones_like(target)

        w = weight.to(dtype=logits.dtype)
        denom = w.sum().clamp_min(self.eps)

        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        bce = (bce * w).sum() / denom

        probs = torch.sigmoid(logits)
        intersection = (probs * target * w).sum()
        cardinality = ((probs + target) * w).sum()
        dice = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)

        l1 = (torch.abs(probs - target) * w).sum() / denom

        total = self.w_bce * bce + self.w_dice * dice + self.w_l1 * l1
        parts = {
            "loss": float(total.detach()),
            "bce": float(bce.detach()),
            "dice": float(dice.detach()),
            "l1": float(l1.detach()),
        }
        return total, parts


__all__ = ["BAMLoss"]
