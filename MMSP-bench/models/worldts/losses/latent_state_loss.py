"""Latent-state objective migrated from DAG commit 0731aa8."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .sigreg_loss import SIGRegLoss


class LatentStateLoss(nn.Module):
    def __init__(
        self,
        latent_mse_weight: float = 1.0,
        latent_alignment_loss: str = "mae",
        latent_cosine_weight: float = 0.0,
        latent_sigreg_weight: float = 0.01,
        latent_detach_target: bool = True,
        sigreg_std_weight: float = 1.0,
        sigreg_cov_weight: float = 0.04,
        sigreg_eps: float = 1e-4,
    ):
        super().__init__()
        self.latent_mse_weight = float(latent_mse_weight)
        self.latent_alignment_loss = str(latent_alignment_loss)
        self.latent_cosine_weight = float(latent_cosine_weight)
        self.latent_sigreg_weight = float(latent_sigreg_weight)
        self.latent_detach_target = bool(latent_detach_target)
        self.sigreg_loss = SIGRegLoss(
            sigreg_std_weight=sigreg_std_weight,
            sigreg_cov_weight=sigreg_cov_weight,
            sigreg_eps=sigreg_eps,
        )

    def forward(
        self,
        h_pred: torch.Tensor,
        h_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if h_target is None:
            raise ValueError("h_target must not be None")
        if h_pred.shape != h_target.shape:
            raise ValueError(
                f"h_pred shape {tuple(h_pred.shape)} must match "
                f"h_target shape {tuple(h_target.shape)}"
            )
        target = (
            h_target.detach() if self.latent_detach_target else h_target
        )
        if self.latent_alignment_loss == "mse":
            alignment_loss = F.mse_loss(h_pred, target)
        elif self.latent_alignment_loss == "mae":
            alignment_loss = F.l1_loss(h_pred, target)
        elif self.latent_alignment_loss == "smooth_l1":
            alignment_loss = F.smooth_l1_loss(h_pred, target)
        elif self.latent_alignment_loss == "normalized_mse":
            centered_pred = h_pred - h_pred.mean(dim=1, keepdim=True)
            centered_target = target - target.mean(dim=1, keepdim=True)
            alignment_loss = F.mse_loss(centered_pred, centered_target)
        else:
            raise ValueError(
                f"Unknown latent_alignment_loss: "
                f"{self.latent_alignment_loss}"
            )

        cosine_loss = h_pred.new_tensor(0.0)
        if self.latent_cosine_weight != 0.0:
            cosine_loss = 1.0 - F.cosine_similarity(
                h_pred.flatten(start_dim=1),
                target.flatten(start_dim=1),
                dim=-1,
            ).mean()

        sigreg_loss = h_pred.new_tensor(0.0)
        sigreg_pred_std_loss = h_pred.new_tensor(0.0)
        sigreg_target_std_loss = h_pred.new_tensor(0.0)
        sigreg_pred_cov_loss = h_pred.new_tensor(0.0)
        sigreg_target_cov_loss = h_pred.new_tensor(0.0)
        h_pred_std_mean = h_pred.new_tensor(0.0)
        h_target_std_mean = h_pred.new_tensor(0.0)
        if self.latent_sigreg_weight != 0.0:
            sigreg_pred = self.sigreg_loss(h_pred)
            sigreg_target = self.sigreg_loss(h_target)
            sigreg_loss = 0.5 * (
                sigreg_pred["loss"] + sigreg_target["loss"]
            )
            sigreg_pred_std_loss = sigreg_pred["std_loss"]
            sigreg_target_std_loss = sigreg_target["std_loss"]
            sigreg_pred_cov_loss = sigreg_pred["cov_loss"]
            sigreg_target_cov_loss = sigreg_target["cov_loss"]
            h_pred_std_mean = sigreg_pred["z_std_mean"]
            h_target_std_mean = sigreg_target["z_std_mean"]

        total_loss = (
            self.latent_mse_weight * alignment_loss
            + self.latent_cosine_weight * cosine_loss
            + self.latent_sigreg_weight * sigreg_loss
        )
        if not torch.isfinite(total_loss).all():
            raise ValueError("LatentStateLoss produced NaN or Inf")

        return {
            "loss": total_loss,
            "mse_loss": alignment_loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "sigreg_pred_std_loss": sigreg_pred_std_loss.detach(),
            "sigreg_target_std_loss": sigreg_target_std_loss.detach(),
            "sigreg_pred_cov_loss": sigreg_pred_cov_loss.detach(),
            "sigreg_target_cov_loss": sigreg_target_cov_loss.detach(),
            "h_pred_std_mean": h_pred_std_mean.detach(),
            "h_target_std_mean": h_target_std_mean.detach(),
        }
