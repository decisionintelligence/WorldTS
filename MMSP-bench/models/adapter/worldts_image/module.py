"""GluonTS-facing module for image-conditioned WorldTS."""

from typing import List, Optional

import torch
import torch.nn.functional as F
from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from torch import nn

from models.worldts.losses import LatentStateLoss

from .model import ImageConditionedWorldTS


class WorldTSImageModule(nn.Module):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_feat_dynamic_real: Optional[List[int]] = None,
        d_multi_modal: int = 1,
        homogenizer_type: str = "linear",
        image_hidden_dim: int = 768,
        modal_hidden_dim: int = 256,
        modal_feature_dim: int = 12,
        latent_dim: int = 64,
        condition_dim: int = 64,
        with_future: bool = True,
        use_revin: bool = True,
        revin_eps: float = 1e-5,
        state_encoder: str = "causal_patch",
        state_patch_len: int = 3,
        state_hidden_multiplier: int = 2,
        decoder_type: str = "pointwise_mlp",
        decoder_patch_len: int = 3,
        decoder_hidden_multiplier: int = 2,
        predictor_patch_len: int = 4,
        predictor_patch_stride: int = 2,
        predictor_d_model: int = 64,
        predictor_n_heads: int = 4,
        predictor_e_layers: int = 2,
        predictor_d_ff: int = 128,
        predictor_dropout: float = 0.0,
        predictor_activation: str = "gelu",
        predictor_factor: int = 1,
        modal_gate_init_logit: float = -4.0,
        cfa_reduction_factor: int = 8,
        cfa_dropout: float = 0.0,
        cfa_activation: str = "gelu",
        latent_alignment_loss: str = "mae",
        latent_mse_weight: float = 10.0,
        latent_cosine_weight: float = 15.0,
        latent_sigreg_weight: float = 0.1,
        latent_detach_target: bool = True,
        sigreg_std_weight: float = 1.0,
        sigreg_cov_weight: float = 0.04,
        sigreg_eps: float = 1e-4,
        decoder_warmup_weight: float = 0.3,
        loss: str = "mse",
        **kwargs,
    ) -> None:
        super().__init__()
        obsolete = sorted(
            key for key in kwargs
            if key.startswith("cody_") or key.startswith("patch_")
            or key == "forecast_loss"
        )
        if obsolete:
            raise ValueError(
                f"Obsolete WorldTS parameters: {obsolete}; use predictor_* and loss."
            )
        del kwargs
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.covariate_dim = sum(d_feat_dynamic_real or [])
        self.decoder_warmup_weight = float(decoder_warmup_weight)
        self.loss = str(loss).lower()
        if self.loss not in {"mse", "mae", "huber"}:
            raise ValueError("loss must be mse, mae, or huber")

        self.model = ImageConditionedWorldTS(
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            covariate_dim=self.covariate_dim,
            image_hidden_dim=image_hidden_dim,
            image_token_dim=d_multi_modal,
            homogenizer_type=homogenizer_type,
            modal_hidden_dim=modal_hidden_dim,
            modal_feature_dim=modal_feature_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            with_future=with_future,
            use_revin=use_revin,
            revin_eps=revin_eps,
            state_encoder=state_encoder,
            state_patch_len=state_patch_len,
            state_hidden_multiplier=state_hidden_multiplier,
            decoder_type=decoder_type,
            decoder_patch_len=decoder_patch_len,
            decoder_hidden_multiplier=decoder_hidden_multiplier,
            predictor_patch_len=predictor_patch_len,
            predictor_patch_stride=predictor_patch_stride,
            predictor_d_model=predictor_d_model,
            predictor_n_heads=predictor_n_heads,
            predictor_e_layers=predictor_e_layers,
            predictor_d_ff=predictor_d_ff,
            predictor_dropout=predictor_dropout,
            predictor_activation=predictor_activation,
            predictor_factor=predictor_factor,
            modal_gate_init_logit=modal_gate_init_logit,
            cfa_reduction_factor=cfa_reduction_factor,
            cfa_dropout=cfa_dropout,
            cfa_activation=cfa_activation,
        )
        self.latent_state_loss = LatentStateLoss(
            latent_mse_weight=latent_mse_weight,
            latent_alignment_loss=latent_alignment_loss,
            latent_cosine_weight=latent_cosine_weight,
            latent_sigreg_weight=latent_sigreg_weight,
            latent_detach_target=latent_detach_target,
            sigreg_std_weight=sigreg_std_weight,
            sigreg_cov_weight=sigreg_cov_weight,
            sigreg_eps=sigreg_eps,
        )

    def describe_inputs(self, batch_size: int = 1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length),
                    dtype=torch.float,
                ),
                "feat_dynamic_real": Input(
                    shape=(
                        batch_size,
                        self.context_length + self.prediction_length,
                        self.covariate_dim,
                    ),
                    dtype=torch.float,
                ),
                "satellite_data": Input(
                    shape=(batch_size, self.context_length, 4, 64, 64),
                    dtype=torch.float,
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        feat_dynamic_real: Optional[torch.Tensor],
        satellite_data: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        prediction = self.model.forecast(
            past_target=past_target,
            past_observed_values=past_observed_values,
            feat_dynamic_real=feat_dynamic_real,
            satellite_data=satellite_data,
        )
        return prediction.squeeze(-1).unsqueeze(1)

    def _forecast_criterion(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        observed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        if observed is None:
            observed = torch.ones_like(target)
        elif observed.dim() == 2:
            observed = observed.unsqueeze(-1)
        if self.loss == "mse":
            point_loss = (prediction - target) ** 2
        elif self.loss == "mae":
            point_loss = torch.abs(prediction - target)
        else:
            point_loss = F.huber_loss(
                prediction,
                target,
                reduction="none",
                delta=0.5,
            )
        denominator = observed.sum().clamp_min(1.0)
        return (point_loss * observed).sum() / denominator

    def stage1_loss(
        self,
        future_target: torch.Tensor,
        future_observed_values: Optional[torch.Tensor] = None,
        **inputs,
    ):
        del future_observed_values
        output = self.model.forward_states(
            future_target=future_target,
            **inputs,
        )
        losses = self.latent_state_loss(
            h_pred=output["h_pred"],
            h_target=output["h_target"],
        )
        return losses["loss"], losses

    def stage2_loss(
        self,
        future_target: torch.Tensor,
        future_observed_values: Optional[torch.Tensor] = None,
        **inputs,
    ):
        with torch.no_grad():
            output = self.model.forward_states(
                future_target=future_target,
                **inputs,
            )
        prediction = self.model.core.decode(
            output["h_pred"].detach(),
            target_revin_stats=output.get("target_revin_stats"),
            history_tokens=output["h_hist"].detach(),
        )
        reconstruction = self.model.core.decode(
            output["h_target"].detach(),
            target_revin_stats=output.get("target_revin_stats"),
            history_tokens=output["h_hist"].detach(),
        )

        prediction_loss = self._forecast_criterion(
            prediction,
            future_target,
            future_observed_values,
        )
        reconstruction_loss = self._forecast_criterion(
            reconstruction,
            future_target,
            future_observed_values,
        )
        total = (
            prediction_loss
            + self.decoder_warmup_weight * reconstruction_loss
        )
        return total, {
            "prediction_loss": prediction_loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
        }

    def validation_forecast_loss(
        self,
        future_target: torch.Tensor,
        future_observed_values: Optional[torch.Tensor] = None,
        **inputs,
    ) -> torch.Tensor:
        prediction = self.model.forecast(**inputs)
        return self._forecast_criterion(
            prediction,
            future_target,
            future_observed_values,
        )
