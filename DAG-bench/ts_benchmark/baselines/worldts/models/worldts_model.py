"""WorldTS numerical patch-state core with DAG's frozen defaults."""

from typing import Optional

import torch
from torch import nn

from ts_benchmark.baselines.worldts.layers.latent_encoder import (
    CausalPatchLatentEncoder,
)
from ts_benchmark.baselines.worldts.layers.latent_patchtst import (
    PatchTSTStateTransitionPredictor,
)
from ts_benchmark.baselines.worldts.layers.multimodal_fusion import (
    HistoricalMLPEncoder,
)
from ts_benchmark.baselines.worldts.layers.state_to_observation_mlp_decoder import (
    CausalPatchStateToObservationDecoder,
    PointwiseMLPStateToObservationDecoder,
)
from ts_benchmark.baselines.worldts.layers.visreg_projector import (
    VISRegProjector,
)


class WorldTSModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = int(getattr(config, "seq_len"))
        self.pred_len = int(getattr(config, "pred_len", getattr(config, "horizon")))
        self.target_dim = int(
            getattr(config, "output_dim", getattr(config, "series_dim", 1))
        )
        self.latent_dim = int(getattr(config, "latent_dim", 32))
        self.state_patch_len = int(getattr(config, "state_patch_len", 3))
        hidden_multiplier = int(
            getattr(config, "state_hidden_multiplier", 2)
        )
        self.decoder_type = str(
            getattr(config, "decoder_type", "causal_patch")
        ).lower()
        self.decoder_patch_len = int(
            getattr(config, "decoder_patch_len", self.state_patch_len)
        )
        decoder_hidden_multiplier = int(
            getattr(
                config,
                "decoder_hidden_multiplier",
                hidden_multiplier,
            )
        )
        configured_condition_dim = int(getattr(config, "condition_dim", 0))
        self.future_x_dim = int(getattr(config, "future_x_dim", 0) or 0)
        self.condition_dim = (
            configured_condition_dim
            if configured_condition_dim > 0
            else (self.latent_dim if self.future_x_dim > 0 else 0)
        )
        self.use_revin = bool(getattr(config, "use_revin", True))
        self.revin_eps = float(getattr(config, "revin_eps", 1e-5))
        self.use_future_x_condition = bool(
            getattr(config, "use_future_x_condition", True)
        )
        self.allow_missing_future_covariate = bool(
            getattr(config, "allow_missing_future_covariate", False)
        )
        self.use_modal_condition = bool(
            getattr(config, "use_modal_condition", False)
        )
        self.use_time_condition = bool(
            getattr(config, "use_time_condition", False)
        )
        self.modal_input_dim = int(getattr(config, "llm_dim", 768))
        self.modal_hidden_dim = int(getattr(config, "modal_hidden_dim", 256))
        self.modal_feature_dim = int(
            getattr(config, "modal_feature_dim", self.latent_dim)
        )
        self.time_input_dim = int(getattr(config, "time_input_dim", 4))
        self.time_hidden_dim = int(getattr(config, "time_hidden_dim", 64))
        self.time_feature_dim = int(getattr(config, "time_feature_dim", 12))
        regularizer_aliases = {
            "legacy": "legacy_vc",
            "sigreg": "legacy_vc",
            "vc": "legacy_vc",
            "vicreg": "legacy_vc",
            "legacy_vc": "legacy_vc",
            "none": "none",
            "visreg": "visreg",
        }
        configured_regularizer = str(
            getattr(config, "latent_regularizer_type", "legacy_vc")
        ).lower()
        if configured_regularizer not in regularizer_aliases:
            raise ValueError(
                "latent_regularizer_type must be 'legacy_vc', 'visreg', "
                f"or 'none', got {configured_regularizer!r}"
            )
        self.latent_regularizer_type = regularizer_aliases[
            configured_regularizer
        ]
        self.visreg_scope = str(
            getattr(config, "visreg_scope", "encoder_all")
        ).lower()
        if self.visreg_scope not in {
            "encoder_all",
            "target_only",
            "pred_target",
        }:
            raise ValueError(
                "visreg_scope must be 'encoder_all', 'target_only', or "
                f"'pred_target', got {self.visreg_scope!r}"
            )
        self.visreg_use_projector = bool(
            getattr(config, "visreg_use_projector", True)
        )
        self.visreg_pooling = str(
            getattr(config, "visreg_pooling", "mean")
        ).lower()
        if self.visreg_pooling not in {"mean", "tokens"}:
            raise ValueError(
                "visreg_pooling must be 'mean' or 'tokens', "
                f"got {self.visreg_pooling!r}"
            )
        self.autonomous_predictor = "latent_patchtst"
        self.use_raw_time_condition = False

        if str(getattr(config, "state_encoder", "causal_patch")) != "causal_patch":
            raise ValueError("WorldTS requires state_encoder='causal_patch'")
        if self.state_patch_len > self.seq_len:
            raise ValueError("state_patch_len must not exceed seq_len")
        if hidden_multiplier <= 0:
            raise ValueError("state_hidden_multiplier must be positive")
        if self.use_future_x_condition and self.future_x_dim <= 0:
            raise ValueError(
                "future covariate conditioning requires positive future_x_dim"
            )
        if self.decoder_type not in {"pointwise_mlp", "causal_patch"}:
            raise ValueError(
                "decoder_type must be 'pointwise_mlp' or 'causal_patch', "
                f"got {self.decoder_type!r}"
            )
        if self.decoder_type == "causal_patch":
            if self.decoder_patch_len <= 0:
                raise ValueError("decoder_patch_len must be positive")
            if self.decoder_patch_len > self.seq_len:
                raise ValueError(
                    "decoder_patch_len must not exceed seq_len"
                )
            if decoder_hidden_multiplier <= 0:
                raise ValueError(
                    "decoder_hidden_multiplier must be positive"
                )

        self.target_encoder = CausalPatchLatentEncoder(
            input_dim=self.target_dim,
            output_dim=self.latent_dim,
            patch_len=self.state_patch_len,
            hidden_dim=hidden_multiplier * self.latent_dim,
        )
        self.condition_encoder = (
            CausalPatchLatentEncoder(
                input_dim=self.future_x_dim,
                output_dim=self.condition_dim,
                patch_len=self.state_patch_len,
                hidden_dim=hidden_multiplier * self.condition_dim,
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
            if self.use_time_condition and not self.use_raw_time_condition
            else None
        )
        self.predictor = PatchTSTStateTransitionPredictor(
            history_length=self.seq_len,
            horizon=self.pred_len,
            latent_dim=self.latent_dim,
            condition_dim=self.condition_dim,
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
            activation=str(getattr(config, "predictor_activation", "gelu")),
            factor=int(getattr(config, "predictor_factor", 1)),
            modal_gate_init_logit=float(
                getattr(config, "modal_gate_init_logit", -4.0)
            ),
            cfa_reduction_factor=int(
                getattr(config, "cfa_reduction_factor", 8)
            ),
            cfa_dropout=float(getattr(config, "cfa_dropout", 0.0)),
            cfa_activation=str(
                getattr(config, "cfa_activation", "gelu")
            ),
        )
        self.predictor_kind = (
            "latent_patchtst_cfa_text_residual_state_causal_patch"
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
                hidden_dim=decoder_hidden_multiplier * self.latent_dim,
            )
        configured_projection_dim = int(
            getattr(config, "visreg_projector_dim", 0)
        )
        self.visreg_projection_dim = (
            configured_projection_dim
            if configured_projection_dim > 0
            else max(1, self.latent_dim // 3)
        )
        self.visreg_projector = (
            VISRegProjector(
                input_dim=self.latent_dim,
                projection_dim=self.visreg_projection_dim,
                hidden_dim=int(
                    getattr(config, "visreg_projector_hidden_dim", 2048)
                ),
            )
            if self.latent_regularizer_type == "visreg"
            and self.visreg_use_projector
            else None
        )
        self.history_num_tokens = self.seq_len
        self.future_num_tokens = self.pred_len
        self._last_modality_diagnostics = {}

    def _stage1_modules(self):
        modules = [self.target_encoder, self.predictor]
        if self.condition_encoder is not None:
            modules.insert(1, self.condition_encoder)
        if self.modal_encoder is not None:
            modules.insert(1, self.modal_encoder)
        if self.time_encoder is not None:
            modules.insert(1, self.time_encoder)
        if self.visreg_projector is not None:
            modules.append(self.visreg_projector)
        return modules

    def _visreg_states(
        self,
        h_hist: torch.Tensor,
        h_target: Optional[torch.Tensor],
        h_pred: torch.Tensor,
    ) -> tuple:
        if self.latent_regularizer_type != "visreg" or h_target is None:
            return None, None, None

        def project_together(*states):
            if self.visreg_pooling == "mean":
                states = tuple(
                    state.mean(dim=1, keepdim=True) for state in states
                )
            if self.visreg_projector is None:
                return states
            lengths = [state.shape[1] for state in states]
            projected = self.visreg_projector(torch.cat(states, dim=1))
            return projected.split(lengths, dim=1)

        if self.visreg_scope == "encoder_all":
            z_hist, z_target = project_together(h_hist, h_target)
            return z_hist, z_target, None
        if self.visreg_scope == "target_only":
            z_target = project_together(h_target)[0]
            return None, z_target, None
        z_pred, z_target = project_together(h_pred, h_target)
        return None, z_target, z_pred

    def set_stage1_eval(self):
        for module in self._stage1_modules():
            module.eval()

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool):
        for parameter in module.parameters():
            parameter.requires_grad = value

    def configure_for_stage1(self):
        for module in self._stage1_modules():
            self._set_requires_grad(module, True)
        self._set_requires_grad(self.decoder, False)

    def configure_for_stage2(self):
        for module in self._stage1_modules():
            self._set_requires_grad(module, False)
        self._set_requires_grad(self.decoder, True)

    def _compute_revin_stats(self, x: torch.Tensor) -> dict:
        mean = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.revin_eps
        ).detach()
        return {"mean": mean, "stdev": stdev}

    @staticmethod
    def _normalize_with_stats(
        x: torch.Tensor,
        mean: torch.Tensor,
        stdev: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if x.shape[0] != mean.shape[0] or x.shape[-1] != mean.shape[-1]:
            raise ValueError(f"{name} is incompatible with RevIN statistics")
        normalized = (x - mean) / stdev
        if not torch.isfinite(normalized).all():
            raise ValueError(f"{name} contains NaN or Inf after normalization")
        return normalized

    def encode_future_target(
        self,
        future_target: torch.Tensor,
        history_target: torch.Tensor,
    ) -> torch.Tensor:
        prefix_len = self.state_patch_len - 1
        if prefix_len == 0:
            return self.target_encoder(future_target)
        target_with_context = torch.cat(
            [history_target[:, -prefix_len:, :], future_target],
            dim=1,
        )
        return self.target_encoder(target_with_context)[:, -self.pred_len :, :]

    def encode_future_condition(
        self,
        x_future: torch.Tensor,
        x_history: torch.Tensor,
    ) -> torch.Tensor:
        if self.condition_encoder is None:
            raise ValueError("future covariates require a condition encoder")
        prefix_len = self.state_patch_len - 1
        if prefix_len == 0:
            return self.condition_encoder(x_future)
        condition_with_context = torch.cat(
            [x_history[:, -prefix_len:, :], x_future],
            dim=1,
        )
        return self.condition_encoder(condition_with_context)[
            :, -self.pred_len :, :
        ]

    def decode(
        self,
        h_tokens: torch.Tensor,
        target_revin_stats: Optional[dict] = None,
        history_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.decoder_type == "pointwise_mlp":
            output = self.decoder(h_tokens)
        else:
            if history_tokens is None:
                raise ValueError(
                    "history_tokens is required by the causal patch decoder"
                )
            output = self.decoder(h_tokens, history_tokens)
        if self.use_revin and target_revin_stats is not None:
            output = (
                output * target_revin_stats["stdev"]
                + target_revin_stats["mean"]
            )
        if not torch.isfinite(output).all():
            raise ValueError("decoded output contains NaN or Inf")
        return output

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
        if self.use_modal_condition and modal_tokens is None:
            raise ValueError(
                "modal_tokens are required when use_modal_condition=True"
            )
        if not self.use_modal_condition and modal_tokens is not None:
            raise ValueError(
                "modal_tokens must be None when modal conditioning is disabled"
            )
        if self.use_time_condition and time_hist is None:
            raise ValueError(
                "time_hist is required when use_time_condition=True"
            )
        if not self.use_time_condition and time_hist is not None:
            raise ValueError(
                "time_hist must be None when time conditioning is disabled"
            )
        expected_input_dim = self.target_dim + self.future_x_dim
        if input.dim() != 3 or tuple(input.shape[1:]) != (
            self.seq_len,
            expected_input_dim,
        ):
            raise ValueError(
                f"input must be [B, {self.seq_len}, {expected_input_dim}], "
                f"got {tuple(input.shape)}"
            )
        if future_target is not None and tuple(future_target.shape[1:]) != (
            self.pred_len,
            self.target_dim,
        ):
            raise ValueError(
                f"future_target must be [B, {self.pred_len}, {self.target_dim}]"
            )
        if x_future is not None and (
            x_future.dim() != 3
            or tuple(x_future.shape) != (
                input.shape[0],
                self.pred_len,
                self.future_x_dim,
            )
        ):
            raise ValueError(
                f"x_future must be [B, {self.pred_len}, {self.future_x_dim}], "
                f"got {tuple(x_future.shape)}"
            )
        if (
            self.use_future_x_condition
            and x_future is None
            and not self.allow_missing_future_covariate
        ):
            raise ValueError(
                "x_future is required when use_future_x_condition=True"
            )
        if not self.use_future_x_condition and x_future is not None:
            raise ValueError(
                "x_future must be None when future conditioning is disabled"
            )
        if not torch.isfinite(input).all():
            raise ValueError("input contains NaN or Inf")
        for name, value, feature_dim in (
            ("modal_tokens", modal_tokens, self.modal_input_dim),
            ("time_hist", time_hist, self.time_input_dim),
        ):
            if value is None:
                continue
            expected = (input.shape[0], self.seq_len, feature_dim)
            if value.dim() != 3 or tuple(value.shape) != expected:
                raise ValueError(
                    f"{name} must be {expected}, got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")

        normalized_input = input
        target_revin_stats = None
        if self.use_revin:
            all_stats = self._compute_revin_stats(input)
            normalized_input = self._normalize_with_stats(
                input, all_stats["mean"], all_stats["stdev"], "input"
            )
            target_revin_stats = {
                "mean": all_stats["mean"][..., : self.target_dim],
                "stdev": all_stats["stdev"][..., : self.target_dim],
            }
            if future_target is not None:
                future_target = self._normalize_with_stats(
                    future_target,
                    target_revin_stats["mean"],
                    target_revin_stats["stdev"],
                    "future_target",
                )
            if x_future is not None:
                x_future = self._normalize_with_stats(
                    x_future,
                    all_stats["mean"][..., self.target_dim :],
                    all_stats["stdev"][..., self.target_dim :],
                    "x_future",
                )

        y_hist = normalized_input[..., : self.target_dim]
        x_hist = (
            normalized_input[..., self.target_dim :]
            if self.future_x_dim > 0
            else None
        )
        h_hist = self.target_encoder(y_hist)
        h_target = (
            self.encode_future_target(future_target, y_hist)
            if future_target is not None
            else None
        )
        z_cond_hist = (
            self.condition_encoder(x_hist)
            if self.condition_encoder is not None
            else None
        )
        z_cond_future = (
            self.encode_future_condition(x_future, x_hist)
            if x_future is not None
            else None
        )
        modal_state = (
            self.modal_encoder(modal_tokens)
            if self.modal_encoder is not None
            else None
        )
        time_state = (
            self.time_encoder(time_hist)
            if self.time_encoder is not None
            else (time_hist if self.use_raw_time_condition else None)
        )
        h_pred = self.predictor(
            h_hist=h_hist,
            c_hist=z_cond_hist,
            m_hist=modal_state,
            t_hist=time_state,
            c_future=z_cond_future,
        )
        z_visreg_hist, z_visreg_target, z_visreg_pred = self._visreg_states(
            h_hist=h_hist,
            h_target=h_target,
            h_pred=h_pred,
        )
        diagnostics = self.predictor.condition_diagnostics()
        self._last_modality_diagnostics = {}
        if modal_state is not None:
            modality_norm = torch.linalg.vector_norm(modal_state.detach())
            diagnostics["modality_norm"] = modality_norm
            self._last_modality_diagnostics["modality_norm"] = modality_norm
        return {
            "h_hist": h_hist,
            "h_target": h_target,
            "z_cond_hist": z_cond_hist,
            "z_cond_future": z_cond_future,
            "modal_state": modal_state,
            "time_state": time_state,
            "h_pred": h_pred,
            "z_visreg_hist": z_visreg_hist,
            "z_visreg_target": z_visreg_target,
            "z_visreg_pred": z_visreg_pred,
            "condition_diagnostics": diagnostics,
            "target_revin_stats": target_revin_stats,
        }

    @staticmethod
    def _module_gradient_norm(module: Optional[nn.Module]) -> float:
        if module is None:
            return 0.0
        squared_norm = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                grad = parameter.grad.detach()
                squared_norm += float(torch.sum(grad * grad).cpu())
        return squared_norm ** 0.5

    def conditioning_diagnostics(self) -> dict:
        diagnostics = self.predictor.condition_diagnostics()
        diagnostics.update(self._last_modality_diagnostics)
        diagnostics.update(
            {
                "modal_encoder_grad_norm": self._module_gradient_norm(
                    self.modal_encoder
                ),
                "time_encoder_grad_norm": self._module_gradient_norm(
                    self.time_encoder
                ),
                "cfa_adapter_grad_norm": self._module_gradient_norm(
                    self.predictor.text_residual_adapter
                ),
                "predictor_grad_norm": self._module_gradient_norm(
                    self.predictor
                ),
            }
        )
        return diagnostics

    def forecast(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.forward(input=input, future_target=None, **kwargs)
        return self.decode(
            out["h_pred"],
            target_revin_stats=out["target_revin_stats"],
            history_tokens=out["h_hist"],
        )
