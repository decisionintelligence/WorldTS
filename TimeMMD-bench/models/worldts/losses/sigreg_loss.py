"""Variance/covariance regularizer migrated from DAG commit 0731aa8."""

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat = z.flatten(start_dim=0, end_dim=1)
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

        return (
            self.sigreg_std_weight * std_loss
            + self.sigreg_cov_weight * cov_loss
        )
