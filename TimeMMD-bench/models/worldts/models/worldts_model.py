"""Self-contained WorldTS model with a latent PatchTST predictor."""

from typing import Optional

import torch
from torch import nn

from ..layers.latent_encoder import (
    CausalPatchLatentEncoder,
    PointwiseLatentEncoder,
)
from ..layers.latent_patchtst import PatchTSTStateTransitionPredictor
from ..layers.multimodal_fusion import HistoricalMLPEncoder
from ..layers.state_to_observation_mlp_decoder import (
    CausalPatchStateToObservationDecoder,
    PointwiseMLPStateToObservationDecoder,
)


class WorldTSModel(nn.Module):
    """WorldTS whose only temporal state predictor is PatchTST."""

    def __init__(self, config):
        super().__init__()
        self.seq_len = int(getattr(config, "seq_len"))
        configured_pred_len = getattr(config, "pred_len", None)
        if configured_pred_len is None:
            configured_pred_len = getattr(config, "horizon")
        self.pred_len = int(configured_pred_len)
        self.target_dim = int(
            getattr(
                config,
                "output_dim",
                getattr(config, "series_dim", 1),
            )
        )
        self.latent_dim = int(getattr(config, "latent_dim", 32))
        self.state_encoder_type = str(
            getattr(config, "state_encoder", "pointwise")
        ).lower()
        self.state_patch_len = int(
            getattr(config, "state_patch_len", 3)
        )
        self.state_hidden_multiplier = int(
            getattr(config, "state_hidden_multiplier", 2)
        )
        configured_condition_dim = int(
            getattr(config, "condition_dim", 0)
        )
        configured_covariate_dim = getattr(config, "future_x_dim", 0)
        self.future_x_dim = int(configured_covariate_dim or 0)
        self.condition_dim = (
            configured_condition_dim
            if configured_condition_dim > 0
            else (self.latent_dim if self.future_x_dim > 0 else 0)
        )

        self.use_revin = bool(getattr(config, "use_revin", True))
        self.revin_eps = float(getattr(config, "revin_eps", 1e-5))
        self.use_future_x_condition = bool(
            getattr(config, "use_future_x_condition", False)
        )
        configured_modal_condition = getattr(
            config,
            "use_modal_condition",
            None,
        )
        if configured_modal_condition is None:
            configured_modal_condition = (
                hasattr(config, "llm_dim")
                and not bool(
                    getattr(config, "numeric_only", False)
                )
            )
        self.use_modal_condition = bool(configured_modal_condition)
        self.use_time_condition = bool(
            getattr(
                config,
                "use_time_condition",
                self.use_modal_condition,
            )
        )
        self.modal_input_dim = int(
            getattr(config, "llm_dim", 768)
        )
        self.time_input_dim = int(
            getattr(config, "time_input_dim", 4)
        )
        self.modal_hidden_dim = int(
            getattr(config, "modal_hidden_dim", 256)
        )
        self.modal_feature_dim = int(
            getattr(config, "modal_feature_dim", 12)
        )
        self.time_hidden_dim = int(
            getattr(config, "time_hidden_dim", 64)
        )
        self.time_feature_dim = int(
            getattr(config, "time_feature_dim", 12)
        )
        self.decoder_type = str(
            getattr(config, "decoder_type", "pointwise_mlp")
        ).lower()
        self.decoder_patch_len = int(
            getattr(config, "decoder_patch_len", self.state_patch_len)
        )
        self.decoder_hidden_multiplier = int(
            getattr(
                config,
                "decoder_hidden_multiplier",
                self.state_hidden_multiplier,
            )
        )

        if self.state_encoder_type == "pointwise":
            self.target_encoder = PointwiseLatentEncoder(
                input_dim=self.target_dim,
                output_dim=self.latent_dim,
            )
        else:
            self.target_encoder = CausalPatchLatentEncoder(
                input_dim=self.target_dim,
                output_dim=self.latent_dim,
                patch_len=self.state_patch_len,
                hidden_dim=(
                    self.state_hidden_multiplier * self.latent_dim
                ),
            )
        self.condition_encoder = (
            CausalPatchLatentEncoder(
                input_dim=self.future_x_dim,
                output_dim=self.condition_dim,
                patch_len=self.state_patch_len,
                hidden_dim=(
                    self.state_hidden_multiplier * self.condition_dim
                ),
            )
            if self.future_x_dim > 0
            else None
        )
        self.modal_encoder = (
            HistoricalMLPEncoder(
                input_dim=self.modal_input_dim,
                hidden_dim=self.modal_hidden_dim,
                output_dim=self.modal_feature_dim,
            )
            if self.use_modal_condition
            else None
        )
        self.time_encoder = (
            HistoricalMLPEncoder(
                input_dim=self.time_input_dim,
                hidden_dim=self.time_hidden_dim,
                output_dim=self.time_feature_dim,
            )
            if self.use_time_condition
            else None
        )
        self.predictor = PatchTSTStateTransitionPredictor(
            history_length=self.seq_len,
            horizon=self.pred_len,
            latent_dim=self.latent_dim,
            condition_dim=(
                self.condition_dim
                if self.condition_encoder is not None
                else 0
            ),
            use_future_x_condition=self.use_future_x_condition,
            modal_condition_dim=(
                self.modal_feature_dim if self.use_modal_condition else 0
            ),
            time_condition_dim=(
                self.time_feature_dim if self.use_time_condition else 0
            ),
            patch_len=int(getattr(config, "predictor_patch_len", 4)),
            stride=int(getattr(config, "predictor_patch_stride", 2)),
            d_model=int(getattr(config, "predictor_d_model", 64)),
            n_heads=int(getattr(config, "predictor_n_heads", 4)),
            e_layers=int(getattr(config, "predictor_e_layers", 2)),
            d_ff=int(getattr(config, "predictor_d_ff", 128)),
            dropout=float(getattr(config, "predictor_dropout", 0.0)),
            activation=str(
                getattr(config, "predictor_activation", "gelu")
            ),
            factor=int(getattr(config, "predictor_factor", 1)),
            modal_gate_init_logit=float(
                getattr(config, "modal_gate_init_logit", -4.0)
            ),
            cfa_reduction_factor=int(
                getattr(config, "cfa_reduction_factor", 8)
            ),
            cfa_dropout=float(
                getattr(config, "cfa_dropout", 0.0)
            ),
            cfa_activation=str(
                getattr(config, "cfa_activation", "gelu")
            ),
        )
        if self.decoder_type == "pointwise_mlp":
            self.decoder = PointwiseMLPStateToObservationDecoder(
                latent_dim=self.latent_dim,
                output_dim=self.target_dim,
            )
        else:
            self.decoder = CausalPatchStateToObservationDecoder(
                latent_dim=self.latent_dim,
                output_dim=self.target_dim,
                patch_len=self.decoder_patch_len,
                hidden_dim=(
                    self.decoder_hidden_multiplier * self.latent_dim
                ),
            )

    def _stage1_modules(self):
        modules = [self.target_encoder, self.predictor]
        if self.condition_encoder is not None:
            modules.insert(1, self.condition_encoder)
        if self.modal_encoder is not None:
            modules.insert(1, self.modal_encoder)
        if self.time_encoder is not None:
            modules.insert(1, self.time_encoder)
        return modules

    def set_stage1_eval(self):
        for module in self._stage1_modules():
            module.eval()

    def encode_history(self, input: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(input)

    def encode_future_target(
        self,
        future_target: Optional[torch.Tensor],
        history_target: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if future_target is None:
            return None
        if self.state_encoder_type == "pointwise":
            return self.target_encoder(future_target)

        prefix_len = self.state_patch_len - 1
        if prefix_len == 0:
            return self.target_encoder(future_target)
        target_with_context = torch.cat(
            [history_target[:, -prefix_len:, :], future_target],
            dim=1,
        )
        encoded = self.target_encoder(target_with_context)
        return encoded[:, -self.pred_len :, :]

    def encode_condition(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.condition_encoder(x)

    def encode_future_condition(
        self,
        x_future: torch.Tensor,
        x_history: torch.Tensor,
    ) -> torch.Tensor:
        """Encode future covariates with a historical patch prefix."""
        prefix_len = self.state_patch_len - 1
        if prefix_len == 0:
            condition_with_context = x_future
        else:
            condition_with_context = torch.cat(
                [x_history[:, -prefix_len:, :], x_future],
                dim=1,
            )
        encoded = self.condition_encoder(condition_with_context)
        return encoded[:, -self.pred_len :, :]

    def _compute_revin_stats(self, input: torch.Tensor) -> dict:
        mean = input.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(input, dim=1, keepdim=True, unbiased=False)
            + self.revin_eps
        ).detach()
        return {"mean": mean, "stdev": stdev}

    @staticmethod
    def _normalize_with_stats(
        x: torch.Tensor,
        mean: torch.Tensor,
        stdev: torch.Tensor,
    ) -> torch.Tensor:
        return (x - mean) / stdev

    def _denormalize_target(
        self,
        y: torch.Tensor,
        target_revin_stats: Optional[dict],
    ) -> torch.Tensor:
        if target_revin_stats is None:
            return y
        mean = target_revin_stats["mean"]
        stdev = target_revin_stats["stdev"]
        return y * stdev + mean

    def decode(
        self,
        h_tokens: torch.Tensor,
        target_revin_stats: Optional[dict] = None,
        history_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.decoder_type == "pointwise_mlp":
            y = self.decoder(h_tokens)
        else:
            y = self.decoder(h_tokens, history_tokens)
        if self.use_revin:
            y = self._denormalize_target(y, target_revin_stats)
        return y

    @staticmethod
    def _set_requires_grad(
        module: nn.Module,
        requires_grad: bool,
    ) -> None:
        for param in module.parameters():
            param.requires_grad = requires_grad

    def configure_for_stage1(self):
        for module in self._stage1_modules():
            self._set_requires_grad(module, True)
        self._set_requires_grad(self.decoder, False)

    def configure_for_stage2(self):
        for module in self._stage1_modules():
            self._set_requires_grad(module, False)
        self._set_requires_grad(self.decoder, True)

    def forward(
        self,
        input: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
        time_hist: Optional[torch.Tensor] = None,
        time_future: Optional[torch.Tensor] = None,
        x_future: Optional[torch.Tensor] = None,
        modal_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        del kwargs, time_future

        y_hist = input[..., : self.target_dim]
        x_hist = (
            input[..., self.target_dim :]
            if input.shape[-1] > self.target_dim
            else None
        )

        target_revin_stats = None
        if self.use_revin:
            all_stats = self._compute_revin_stats(input)
            normalized_input = self._normalize_with_stats(
                input,
                all_stats["mean"],
                all_stats["stdev"],
            )
            y_hist = normalized_input[..., : self.target_dim]
            x_hist = (
                normalized_input[..., self.target_dim :]
                if normalized_input.shape[-1] > self.target_dim
                else None
            )
            target_mean = all_stats["mean"][..., : self.target_dim]
            target_stdev = all_stats["stdev"][..., : self.target_dim]
            target_revin_stats = {
                "mean": target_mean,
                "stdev": target_stdev,
            }
            if future_target is not None:
                future_target = self._normalize_with_stats(
                    future_target,
                    target_mean,
                    target_stdev,
                )
            if x_future is not None:
                x_future = self._normalize_with_stats(
                    x_future,
                    all_stats["mean"][..., self.target_dim :],
                    all_stats["stdev"][..., self.target_dim :],
                )

        h_hist = self.encode_history(y_hist)
        h_target = self.encode_future_target(
            future_target,
            history_target=y_hist,
        )
        modal_state = (
            self.modal_encoder(modal_tokens)
            if self.modal_encoder is not None
            else None
        )
        time_state = (
            self.time_encoder(time_hist)
            if self.time_encoder is not None
            else None
        )
        z_cond_hist = (
            self.encode_condition(x_hist)
            if x_hist is not None
            else None
        )
        z_cond_future = (
            self.encode_future_condition(x_future, x_hist)
            if x_future is not None
            else None
        )
        h_pred = self.predictor(
            h_hist=h_hist,
            c_hist=z_cond_hist,
            m_hist=modal_state,
            t_hist=time_state,
            c_future=z_cond_future,
        )
        return {
            "h_hist": h_hist,
            "h_target": h_target,
            "z_cond_hist": z_cond_hist,
            "z_cond_future": z_cond_future,
            "modal_state": modal_state,
            "time_state": time_state,
            "h_pred": h_pred,
            "target_revin_stats": target_revin_stats,
        }

    def forecast(
        self,
        input: torch.Tensor,
        time_hist: Optional[torch.Tensor] = None,
        time_future: Optional[torch.Tensor] = None,
        x_future: Optional[torch.Tensor] = None,
        modal_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.forward(
            input=input,
            future_target=None,
            time_hist=time_hist,
            time_future=time_future,
            x_future=x_future,
            modal_tokens=modal_tokens,
        )
        return self.decode(
            out["h_pred"],
            target_revin_stats=out.get("target_revin_stats"),
            history_tokens=out["h_hist"],
        )
