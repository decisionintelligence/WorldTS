"""Variance/covariance regularizer migrated from DAG commit 0731aa8."""

from typing import Dict

import torch
from torch import nn


class SIGRegLoss(nn.Module):
    """Simple variance/covariance regularizer on projected latent tokens."""

    def __init__(
        self,
        sigreg_std_weight: float = 1.0,
        sigreg_cov_weight: float = 0.04,
        sigreg_eps: float = 1e-4,
    ):
        super().__init__()
        self.sigreg_std_weight = float(sigreg_std_weight)
        self.sigreg_cov_weight = float(sigreg_cov_weight)
        self.sigreg_eps = float(sigreg_eps)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z is None:
            raise ValueError("z must not be None")
        if z.dim() != 3:
            raise ValueError(f"z must be [B, N, D], got {tuple(z.shape)}")

        flat = z.flatten(start_dim=0, end_dim=1)
        if flat.shape[0] == 0:
            raise ValueError("z must contain at least one token")

        centered = flat - flat.mean(dim=0, keepdim=True)
        std = torch.sqrt(
            centered.var(dim=0, unbiased=False) + self.sigreg_eps
        )
        std_loss = torch.relu(1.0 - std).mean()

        if flat.shape[0] <= 1:
            cov_loss = z.new_tensor(0.0)
        else:
            cov = (
                centered.transpose(0, 1) @ centered
            ) / float(flat.shape[0] - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_loss = (off_diag ** 2).sum() / float(z.shape[-1])

        loss = (
            self.sigreg_std_weight * std_loss
            + self.sigreg_cov_weight * cov_loss
        )
        if not torch.isfinite(loss).all():
            raise ValueError("SIGRegLoss produced NaN or Inf")

        return {
            "loss": loss,
            "std_loss": std_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "z_std_mean": std.mean().detach(),
        }
