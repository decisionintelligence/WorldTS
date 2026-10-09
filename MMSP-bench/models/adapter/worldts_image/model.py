"""WorldTS with the UniCA satellite frontend and the existing modal MLP."""

from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

from models.worldts import WorldTSModel

from .image_homogenizer import UniCAImageHomogenizer


class ImageConditionedWorldTS(nn.Module):
    """Compose UniCA image homogenization with WorldTS's modal branch.

    The shared comparison boundary is ``image_tokens`` with shape ``[B,T,K]``.
    WorldTS's existing ``HistoricalMLPEncoder(K -> M)`` then produces
    ``modal_state``, which is passed unchanged as ``m_hist`` to CFA.
    """

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        covariate_dim: int,
        image_hidden_dim: int = 768,
        image_token_dim: int = 1,
        homogenizer_type: str = "linear",
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
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.covariate_dim = int(covariate_dim)
        self.image_token_dim = int(image_token_dim)
        self.with_future = bool(with_future)
        if self.context_length <= 0 or self.prediction_length <= 0:
            raise ValueError("context_length and prediction_length must be positive")
        if self.covariate_dim < 0:
            raise ValueError("covariate_dim must be non-negative")

        self.image_homogenizer = UniCAImageHomogenizer(
            hidden_dim=image_hidden_dim,
            output_dim=self.image_token_dim,
            homogenizer_type=homogenizer_type,
        )
        config = SimpleNamespace(
            seq_len=self.context_length,
            pred_len=self.prediction_length,
            output_dim=1,
            series_dim=1,
            latent_dim=int(latent_dim),
            future_x_dim=self.covariate_dim,
            condition_dim=(int(condition_dim) if self.covariate_dim > 0 else 0),
            use_future_x_condition=(
                self.with_future and self.covariate_dim > 0
            ),
            allow_missing_future_covariate=False,
            use_modal_condition=True,
            use_time_condition=False,
            llm_dim=self.image_token_dim,
            modal_hidden_dim=int(modal_hidden_dim),
            modal_feature_dim=int(modal_feature_dim),
            use_revin=bool(use_revin),
            revin_eps=float(revin_eps),
            state_encoder=state_encoder,
            state_patch_len=int(state_patch_len),
            state_hidden_multiplier=int(state_hidden_multiplier),
            decoder_type=decoder_type,
            decoder_patch_len=int(decoder_patch_len),
            decoder_hidden_multiplier=int(decoder_hidden_multiplier),
            predictor_patch_len=int(predictor_patch_len),
            predictor_patch_stride=int(predictor_patch_stride),
            predictor_d_model=int(predictor_d_model),
            predictor_n_heads=int(predictor_n_heads),
            predictor_e_layers=int(predictor_e_layers),
            predictor_d_ff=int(predictor_d_ff),
            predictor_dropout=float(predictor_dropout),
            predictor_activation=predictor_activation,
            predictor_factor=int(predictor_factor),
            modal_gate_init_logit=float(modal_gate_init_logit),
            cfa_reduction_factor=int(cfa_reduction_factor),
            cfa_dropout=float(cfa_dropout),
            cfa_activation=cfa_activation,
        )
        self.core = WorldTSModel(config)

    @property
    def modal_encoder(self) -> nn.Module:
        return self.core.modal_encoder

    @property
    def predictor(self) -> nn.Module:
        return self.core.predictor

    @property
    def decoder(self) -> nn.Module:
        return self.core.decoder

    def _prepare_inputs(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        feat_dynamic_real: Optional[torch.Tensor],
        satellite_data: torch.Tensor,
    ):
        if tuple(past_target.shape[1:]) != (self.context_length,):
            raise ValueError(
                f"past_target must be [B, {self.context_length}], "
                f"got {tuple(past_target.shape)}"
            )
        if past_observed_values.shape != past_target.shape:
            raise ValueError("past_observed_values must match past_target")
        if satellite_data.shape[:2] != past_target.shape:
            raise ValueError(
                "satellite_data must align with the historical target; "
                f"got target={tuple(past_target.shape)} "
                f"satellite={tuple(satellite_data.shape)}"
            )

        image_tokens = self.image_homogenizer(satellite_data)
        image_tokens = image_tokens * past_observed_values.unsqueeze(-1)
        y_hist = past_target.unsqueeze(-1)

        x_hist = None
        x_future = None
        if self.covariate_dim > 0:
            expected = (
                past_target.shape[0],
                self.context_length + self.prediction_length,
                self.covariate_dim,
            )
            if feat_dynamic_real is None or tuple(feat_dynamic_real.shape) != expected:
                actual = None if feat_dynamic_real is None else tuple(feat_dynamic_real.shape)
                raise ValueError(
                    f"feat_dynamic_real must be {expected}, got {actual}"
                )
            x_hist = feat_dynamic_real[:, : self.context_length]
            if self.with_future:
                x_future = feat_dynamic_real[:, self.context_length :]
            model_input = torch.cat([y_hist, x_hist], dim=-1)
        else:
            model_input = y_hist
        return model_input, x_future, image_tokens

    def forward_states(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        feat_dynamic_real: Optional[torch.Tensor],
        satellite_data: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> dict:
        model_input, x_future, image_tokens = self._prepare_inputs(
            past_target,
            past_observed_values,
            feat_dynamic_real,
            satellite_data,
        )
        target = None
        if future_target is not None:
            target = (
                future_target.unsqueeze(-1)
                if future_target.dim() == 2
                else future_target
            )
        output = self.core(
            input=model_input,
            future_target=target,
            x_future=x_future,
            modal_tokens=image_tokens,
        )
        output["image_tokens"] = image_tokens
        return output

    def forecast(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        feat_dynamic_real: Optional[torch.Tensor],
        satellite_data: torch.Tensor,
    ) -> torch.Tensor:
        output = self.forward_states(
            past_target=past_target,
            past_observed_values=past_observed_values,
            feat_dynamic_real=feat_dynamic_real,
            satellite_data=satellite_data,
        )
        return self.core.decode(
            output["h_pred"],
            target_revin_stats=output.get("target_revin_stats"),
            history_tokens=output["h_hist"],
        )

    def configure_for_stage1(self) -> None:
        self.core.configure_for_stage1()
        self.core._set_requires_grad(self.image_homogenizer, True)

    def configure_for_stage2(self) -> None:
        self.core.configure_for_stage2()
        self.core._set_requires_grad(self.image_homogenizer, False)

    def set_stage1_eval(self) -> None:
        self.core.set_stage1_eval()
        self.image_homogenizer.eval()
