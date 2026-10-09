# -*- coding: utf-8 -*-
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ts_benchmark.baselines.worldts.losses.sigreg_loss import SIGRegLoss
from ts_benchmark.baselines.worldts.losses.visreg_loss import VISRegLoss


class LatentStateLoss(nn.Module):
    def __init__(
        self,
        latent_mse_weight: float = 1.0,
        latent_alignment_loss: str = 'mae',
        latent_cosine_weight: float = 0.0,
        latent_sigreg_weight: float = 0.01,
        latent_detach_target: bool = True,
        sigreg_std_weight: float = 1.0,
        sigreg_cov_weight: float = 0.04,
        sigreg_eps: float = 1e-4,
        latent_regularizer_type: str = 'legacy_vc',
        visreg_scope: str = 'encoder_all',
        visreg_num_projections: int = 256,
        visreg_scale_weight: float = 1.0,
        visreg_shape_weight: float = 1.0,
        visreg_center_weight: float = 1.0,
        visreg_eps: float = 1e-6,
    ):
        super().__init__()
        self.latent_mse_weight = float(latent_mse_weight)
        self.latent_alignment_loss = str(latent_alignment_loss)
        self.latent_cosine_weight = float(latent_cosine_weight)
        self.latent_sigreg_weight = float(latent_sigreg_weight)
        self.latent_detach_target = bool(latent_detach_target)
        self.latent_regularizer_type = self._normalize_regularizer_type(
            latent_regularizer_type
        )
        self.visreg_scope = str(visreg_scope).lower()
        if self.visreg_scope not in {'encoder_all', 'target_only', 'pred_target'}:
            raise ValueError(
                "visreg_scope must be 'encoder_all', 'target_only', or "
                f"'pred_target', got {visreg_scope!r}"
            )

        self.sigreg_loss = SIGRegLoss(
            sigreg_std_weight=sigreg_std_weight,
            sigreg_cov_weight=sigreg_cov_weight,
            sigreg_eps=sigreg_eps,
        )
        self.visreg_loss = (
            VISRegLoss(
                num_projections=visreg_num_projections,
                scale_weight=visreg_scale_weight,
                shape_weight=visreg_shape_weight,
                center_weight=visreg_center_weight,
                eps=visreg_eps,
            )
            if self.latent_regularizer_type == 'visreg'
            else None
        )

    @staticmethod
    def _normalize_regularizer_type(value: str) -> str:
        normalized = str(value).lower()
        aliases = {
            'legacy': 'legacy_vc',
            'sigreg': 'legacy_vc',
            'vc': 'legacy_vc',
            'vicreg': 'legacy_vc',
            'legacy_vc': 'legacy_vc',
            'none': 'none',
            'visreg': 'visreg',
        }
        if normalized not in aliases:
            raise ValueError(
                "latent_regularizer_type must be 'legacy_vc', 'visreg', "
                f"or 'none', got {value!r}"
            )
        return aliases[normalized]

    def _selected_visreg_inputs(
        self,
        z_visreg_hist: Optional[torch.Tensor],
        z_visreg_target: Optional[torch.Tensor],
        z_visreg_pred: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        candidates = {
            'hist': z_visreg_hist,
            'target': z_visreg_target,
            'pred': z_visreg_pred,
        }
        if self.visreg_scope == 'encoder_all':
            names = ('hist', 'target')
        elif self.visreg_scope == 'target_only':
            names = ('target',)
        else:
            names = ('pred', 'target')
        missing = [name for name in names if candidates[name] is None]
        if missing:
            raise ValueError(
                f"VISReg scope {self.visreg_scope!r} requires projected "
                f"states: {', '.join(missing)}"
            )
        return {name: candidates[name] for name in names}

    def forward(
        self,
        h_pred: torch.Tensor,
        h_target: torch.Tensor,
        z_visreg_hist: Optional[torch.Tensor] = None,
        z_visreg_target: Optional[torch.Tensor] = None,
        z_visreg_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if h_target is None:
            raise ValueError('h_target must not be None')
        if h_pred.shape != h_target.shape:
            raise ValueError(
                f'h_pred shape {tuple(h_pred.shape)} must match h_target shape {tuple(h_target.shape)}'
            )
        # Alignment and trajectory cosine use a stop-gradient target.
        target = h_target.detach() if self.latent_detach_target else h_target
        if self.latent_alignment_loss == 'mse':
            mse_loss = F.mse_loss(h_pred, target)
        elif self.latent_alignment_loss == 'mae':
            mse_loss = F.l1_loss(h_pred, target)
        elif self.latent_alignment_loss == 'smooth_l1':
            mse_loss = F.smooth_l1_loss(h_pred, target)
        elif self.latent_alignment_loss == 'normalized_mse':
            centered_pred = h_pred - h_pred.mean(dim=1, keepdim=True)
            centered_target = target - target.mean(dim=1, keepdim=True)
            mse_loss = F.mse_loss(centered_pred, centered_target)
        else:
            raise ValueError(f'Unknown latent_alignment_loss: {self.latent_alignment_loss}')

        # Compute one cosine distance per sample after flattening [H, D] to [HD].
        cosine_loss = h_pred.new_tensor(0.0)
        if self.latent_cosine_weight != 0.0:
            flat_pred = h_pred.flatten(start_dim=1)
            flat_target = target.flatten(start_dim=1)
            cosine_loss = 1.0 - F.cosine_similarity(flat_pred, flat_target, dim=-1).mean()

        zero = h_pred.new_tensor(0.0)
        regularizer_loss = zero
        sigreg_pred_loss = zero
        sigreg_target_loss = zero
        sigreg_pred_std_loss = zero
        sigreg_target_std_loss = zero
        sigreg_pred_cov_loss = zero
        sigreg_target_cov_loss = zero
        h_pred_std_mean = zero
        h_target_std_mean = zero
        visreg_scale_loss = zero
        visreg_shape_loss = zero
        visreg_center_loss = zero
        visreg_hist_loss = zero
        visreg_target_loss = zero
        visreg_pred_loss = zero

        if self.latent_sigreg_weight != 0.0:
            if self.latent_regularizer_type == 'legacy_vc':
                sigreg_pred = self.sigreg_loss(h_pred)
                sigreg_target = self.sigreg_loss(h_target)
                sigreg_pred_loss = sigreg_pred['loss']
                sigreg_target_loss = sigreg_target['loss']
                regularizer_loss = 0.5 * (
                    sigreg_pred_loss + sigreg_target_loss
                )
                sigreg_pred_std_loss = sigreg_pred['std_loss']
                sigreg_target_std_loss = sigreg_target['std_loss']
                sigreg_pred_cov_loss = sigreg_pred['cov_loss']
                sigreg_target_cov_loss = sigreg_target['cov_loss']
                h_pred_std_mean = sigreg_pred['z_std_mean']
                h_target_std_mean = sigreg_target['z_std_mean']
            elif self.latent_regularizer_type == 'visreg':
                selected = self._selected_visreg_inputs(
                    z_visreg_hist=z_visreg_hist,
                    z_visreg_target=z_visreg_target,
                    z_visreg_pred=z_visreg_pred,
                )
                results = {
                    name: self.visreg_loss(states)
                    for name, states in selected.items()
                }
                regularizer_loss = torch.stack(
                    [result['loss'] for result in results.values()]
                ).mean()
                visreg_scale_loss = torch.stack(
                    [result['scale_loss'] for result in results.values()]
                ).mean()
                visreg_shape_loss = torch.stack(
                    [result['shape_loss'] for result in results.values()]
                ).mean()
                visreg_center_loss = torch.stack(
                    [result['center_loss'] for result in results.values()]
                ).mean()
                if 'hist' in results:
                    visreg_hist_loss = results['hist']['loss'].detach()
                if 'target' in results:
                    visreg_target_loss = results['target']['loss'].detach()
                    h_target_std_mean = results['target']['z_std_mean']
                if 'pred' in results:
                    visreg_pred_loss = results['pred']['loss'].detach()
                    h_pred_std_mean = results['pred']['z_std_mean']

        total_loss = (
            self.latent_mse_weight * mse_loss
            + self.latent_cosine_weight * cosine_loss
            + self.latent_sigreg_weight * regularizer_loss
        )
        if not torch.isfinite(total_loss).all():
            raise ValueError('LatentStateLoss produced NaN or Inf')

        return {
            'loss': total_loss,
            'mse_loss': mse_loss.detach(),
            'cosine_loss': cosine_loss.detach(),
            # Keep the historical key as an alias for logging compatibility.
            'sigreg_loss': regularizer_loss.detach(),
            'regularizer_loss': regularizer_loss.detach(),
            'sigreg_pred_std_loss': sigreg_pred_std_loss.detach(),
            'sigreg_target_std_loss': sigreg_target_std_loss.detach(),
            'sigreg_pred_cov_loss': sigreg_pred_cov_loss.detach(),
            'sigreg_target_cov_loss': sigreg_target_cov_loss.detach(),
            'h_pred_std_mean': h_pred_std_mean.detach(),
            'h_target_std_mean': h_target_std_mean.detach(),
            'visreg_scale_loss': visreg_scale_loss.detach(),
            'visreg_shape_loss': visreg_shape_loss.detach(),
            'visreg_center_loss': visreg_center_loss.detach(),
            'visreg_hist_loss': visreg_hist_loss.detach(),
            'visreg_target_loss': visreg_target_loss.detach(),
            'visreg_pred_loss': visreg_pred_loss.detach(),
        }
