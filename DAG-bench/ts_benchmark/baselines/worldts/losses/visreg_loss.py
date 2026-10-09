# -*- coding: utf-8 -*-
"""Variance-Invariance-Sketching regularization for latent sequences."""

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class VISRegLoss(nn.Module):
    """Regularize projected encoder states with center, scale, and shape losses.

    The input follows the local latent convention ``[B, N, D]``. Each token
    position is treated like a VISReg view, so distribution statistics are
    computed over the batch independently for every position. The stochastic
    sliced-Wasserstein calculation is kept in float32 for AMP stability.
    """

    def __init__(
        self,
        num_projections: int = 256,
        scale_weight: float = 1.0,
        shape_weight: float = 1.0,
        center_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_projections = int(num_projections)
        self.scale_weight = float(scale_weight)
        self.shape_weight = float(shape_weight)
        self.center_weight = float(center_weight)
        self.eps = float(eps)
        if self.num_projections <= 0:
            raise ValueError("num_projections must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    @staticmethod
    def _normal_quantiles(
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        probabilities = torch.arange(
            1,
            batch_size + 1,
            device=device,
            dtype=torch.float32,
        ) / float(batch_size + 1)
        return torch.erfinv(2.0 * probabilities - 1.0) * math.sqrt(2.0)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z is None:
            raise ValueError("z must not be None")
        if z.dim() != 3:
            raise ValueError(f"z must be [B, N, D], got {tuple(z.shape)}")
        batch_size, num_tokens, dim = z.shape
        if batch_size <= 0 or num_tokens <= 0 or dim <= 0:
            raise ValueError("z dimensions must all be positive")
        if not torch.isfinite(z).all():
            raise ValueError("z contains NaN or Inf")

        # Official VISReg uses [V, B, D] and computes each view's distribution
        # over B. Our N temporal positions play the role of V here.
        views = z.transpose(0, 1).float()
        mean = views.mean(dim=1, keepdim=True)
        center_loss = mean.square().mean()

        centered = views - mean
        std = centered.norm(dim=1) / math.sqrt(float(batch_size))
        std = std.clamp_min(self.eps)
        scale_loss = (std - 1.0).square().mean()

        normalized = centered / std.detach().unsqueeze(1)
        directions = F.normalize(
            torch.randn(
                dim,
                self.num_projections,
                device=z.device,
                dtype=torch.float32,
            ),
            dim=0,
        )
        sorted_projections = (normalized @ directions).sort(dim=1).values
        target = self._normal_quantiles(batch_size, z.device).view(
            1,
            batch_size,
            1,
        )
        shape_loss = (sorted_projections - target).square().mean()

        loss = (
            self.scale_weight * scale_loss
            + self.shape_weight * shape_loss
            + self.center_weight * center_loss
        )
        if not torch.isfinite(loss).all():
            raise ValueError("VISRegLoss produced NaN or Inf")

        return {
            "loss": loss,
            "scale_loss": scale_loss.detach(),
            "shape_loss": shape_loss.detach(),
            "center_loss": center_loss.detach(),
            "z_std_mean": std.mean().detach(),
        }
