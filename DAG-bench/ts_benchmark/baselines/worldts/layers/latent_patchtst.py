"""Latent PatchTST predictor with optional historical conditioning."""

from math import log
from typing import Optional

import torch
from torch import nn

from ts_benchmark.baselines.worldts.layers.deterministic_padding import (
    deterministic_replicate_pad_1d,
)
from ts_benchmark.baselines.worldts.layers.multimodal_fusion import (
    CFAResidualAdapter,
)
from ts_benchmark.baselines.worldts.layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
)
from ts_benchmark.baselines.worldts.layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(log(10000.0) / d_model)
        ).exp()
        pe = torch.zeros(max_len, d_model).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    """The same replicated-end patch embedding used by PatchTST."""

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding = int(padding)
        self.value_embedding = nn.Linear(self.patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        n_vars = x.shape[1]
        x = deterministic_replicate_pad_1d(x, right=self.padding)
        x = x.unfold(-1, self.patch_len, self.stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class LatentPatchTSTCore(nn.Module):
    """Treat latent coordinates as PatchTST channels."""

    def __init__(
        self,
        history_length: int,
        horizon: int,
        latent_dim: int,
        patch_len: int = 4,
        stride: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.0,
        activation: str = "gelu",
        factor: int = 1,
    ):
        super().__init__()
        self.history_length = int(history_length)
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.e_layers = int(e_layers)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.activation = str(activation)
        self.factor = int(factor)

        for name, value in (
            ("history_length", self.history_length),
            ("horizon", self.horizon),
            ("latent_dim", self.latent_dim),
            ("patch_len", self.patch_len),
            ("stride", self.stride),
            ("d_model", self.d_model),
            ("n_heads", self.n_heads),
            ("e_layers", self.e_layers),
            ("d_ff", self.d_ff),
            ("factor", self.factor),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.patch_len > self.history_length:
            raise ValueError("patch_len must not exceed history_length")
        if self.stride > self.patch_len:
            raise ValueError("stride must not exceed patch_len")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")

        self.patch_embedding = PatchEmbedding(
            self.d_model,
            self.patch_len,
            self.stride,
            self.stride,
            self.dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )
        self.patch_num = (
            (self.history_length - self.patch_len) // self.stride + 2
        )
        self.head = nn.Linear(self.d_model * self.patch_num, self.horizon)
        self._last_attentions = None

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        expected = (
            state_history.shape[0],
            self.history_length,
            self.latent_dim,
        )
        if state_history.dim() != 3 or tuple(state_history.shape) != expected:
            raise ValueError(
                f"state_history must be {expected}, got {tuple(state_history.shape)}"
            )
        if not torch.isfinite(state_history).all():
            raise ValueError("state_history contains NaN or Inf")

        # Deliberately no second temporal normalization in latent space.
        patch_tokens, variable_count = self.patch_embedding(
            state_history.transpose(1, 2)
        )
        if variable_count != self.latent_dim:
            raise RuntimeError("latent channel count changed during patching")
        if patch_tokens.shape[1] != self.patch_num:
            raise RuntimeError(
                f"expected {self.patch_num} patches, got {patch_tokens.shape[1]}"
            )
        encoded, attentions = self.encoder(patch_tokens, attn_mask=None)
        self._last_attentions = [
            item.detach() if item is not None else None for item in attentions
        ]
        batch_size = state_history.shape[0]
        encoded = encoded.reshape(
            batch_size,
            self.latent_dim,
            self.patch_num,
            self.d_model,
        )
        encoded = encoded.permute(0, 1, 3, 2).flatten(start_dim=-2)
        future_latent = self.head(encoded).transpose(1, 2)
        expected_output = (batch_size, self.horizon, self.latent_dim)
        if tuple(future_latent.shape) != expected_output:
            raise RuntimeError(
                f"future latent must be {expected_output}, got {tuple(future_latent.shape)}"
            )
        if not torch.isfinite(future_latent).all():
            raise ValueError("latent PatchTST produced NaN or Inf")
        return future_latent


class PatchTSTStateTransitionPredictor(nn.Module):
    """Forecast an autonomous trajectory and correct it with future covariates."""

    def __init__(
        self,
        history_length: int,
        horizon: int,
        latent_dim: int,
        condition_dim: int,
        use_future_x_condition: bool,
        modal_condition_dim: int = 0,
        time_condition_dim: int = 0,
        patch_len: int = 4,
        stride: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.0,
        activation: str = "gelu",
        factor: int = 1,
        modal_gate_init_logit: float = -4.0,
        cfa_reduction_factor: int = 8,
        cfa_dropout: float = 0.0,
        cfa_activation: str = "gelu",
    ):
        super().__init__()
        self.history_length = int(history_length)
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.use_future_x_condition = bool(use_future_x_condition)
        self.modal_condition_dim = int(modal_condition_dim)
        self.time_condition_dim = int(time_condition_dim)
        if self.condition_dim < 0:
            raise ValueError("condition_dim must be non-negative")
        if self.modal_condition_dim < 0:
            raise ValueError("modal_condition_dim must be non-negative")
        if self.time_condition_dim < 0:
            raise ValueError("time_condition_dim must be non-negative")
        if self.use_future_x_condition and self.condition_dim == 0:
            raise ValueError(
                "future covariate conditioning requires positive condition_dim"
            )

        if self.condition_dim > 0:
            joint_dim = self.latent_dim + self.condition_dim
            self.hist_fusion_mlp = nn.Sequential(
                nn.LayerNorm(joint_dim),
                nn.Linear(joint_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.hist_fusion_mlp = None
        self.text_residual_adapter = (
            CFAResidualAdapter(
                input_dim=self.modal_condition_dim,
                latent_dim=self.latent_dim,
                reduction_factor=cfa_reduction_factor,
                dropout=cfa_dropout,
                activation=cfa_activation,
            )
            if self.modal_condition_dim > 0
            else None
        )
        if self.time_condition_dim > 0:
            historical_context_dim = self.condition_dim + self.time_condition_dim
            self.condition_projection = nn.Sequential(
                nn.Linear(historical_context_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            self.historical_context_fusion = nn.Sequential(
                nn.Linear(2 * self.latent_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.condition_projection = None
            self.historical_context_fusion = None
        self.patch_predictor = LatentPatchTSTCore(
            history_length=self.history_length,
            horizon=self.horizon,
            latent_dim=self.latent_dim,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            factor=factor,
        )
        self.time_residual_gate_logit = (
            nn.Parameter(torch.tensor(float(modal_gate_init_logit)))
            if self.time_condition_dim > 0
            else None
        )
        if self.use_future_x_condition:
            joint_dim = self.latent_dim + self.condition_dim
            self.control_mlp = nn.Sequential(
                nn.LayerNorm(joint_dim),
                nn.Linear(joint_dim, self.latent_dim),
                nn.GELU(),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
            # Begin exactly from the history-only PatchTST trajectory.
            nn.init.zeros_(self.control_mlp[-1].weight)
            nn.init.zeros_(self.control_mlp[-1].bias)
        else:
            self.control_mlp = None
        self._last_condition_diagnostics = {}

    @staticmethod
    def _check_shape(name: str, x: torch.Tensor, expected: tuple) -> None:
        if x.dim() != 3 or tuple(x.shape) != expected:
            raise ValueError(f"{name} must be {expected}, got {tuple(x.shape)}")

    def _base_state(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if c_hist is None:
            return h_hist
        if self.condition_dim <= 0 or self.hist_fusion_mlp is None:
            raise ValueError(
                "c_hist was provided, but the predictor has no covariate branch"
            )
        self._check_shape(
            "c_hist",
            c_hist,
            (h_hist.shape[0], self.history_length, self.condition_dim),
        )
        return self.hist_fusion_mlp(torch.cat([h_hist, c_hist], dim=-1))

    def _historical_condition_fusion(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
        t_hist: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.condition_projection is None
            or self.historical_context_fusion is None
        ):
            raise ValueError("historical time fusion was not configured")
        conditions = []
        if self.condition_dim > 0:
            if c_hist is None:
                raise ValueError("c_hist is required by this time configuration")
            self._check_shape(
                "c_hist",
                c_hist,
                (h_hist.shape[0], self.history_length, self.condition_dim),
            )
            conditions.append(c_hist)
        elif c_hist is not None:
            raise ValueError("c_hist was provided, but condition_dim is zero")
        conditions.append(t_hist)
        projected_condition = self.condition_projection(
            torch.cat(conditions, dim=-1)
        )
        time_delta = self.historical_context_fusion(
            torch.cat([h_hist, projected_condition], dim=-1)
        )
        history_norm = torch.linalg.vector_norm(h_hist.detach())
        condition_norm = torch.linalg.vector_norm(projected_condition.detach())
        self._last_condition_diagnostics = {
            "history_norm": history_norm,
            "condition_projection_norm": condition_norm,
            "condition_contribution": (
                condition_norm / history_norm.clamp_min(1e-12)
            ),
        }
        return time_delta

    def _validate_historical_modalities(
        self,
        m_hist: Optional[torch.Tensor],
        t_hist: Optional[torch.Tensor],
        batch_size: int,
    ) -> None:
        if self.modal_condition_dim > 0:
            if m_hist is None:
                raise ValueError(
                    "m_hist is required when modal conditioning is enabled"
                )
            self._check_shape(
                "m_hist",
                m_hist,
                (batch_size, self.history_length, self.modal_condition_dim),
            )
        elif m_hist is not None:
            raise ValueError(
                "m_hist was provided, but modal conditioning is disabled"
            )

        if self.time_condition_dim > 0:
            if t_hist is None:
                raise ValueError(
                    "t_hist is required when time conditioning is enabled"
                )
            self._check_shape(
                "t_hist",
                t_hist,
                (batch_size, self.history_length, self.time_condition_dim),
            )
        elif t_hist is not None:
            raise ValueError(
                "t_hist was provided, but time conditioning is disabled"
            )

    def autonomous_trajectory(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
        m_hist: Optional[torch.Tensor] = None,
        t_hist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        expected_hist = (
            h_hist.shape[0],
            self.history_length,
            self.latent_dim,
        )
        self._check_shape("h_hist", h_hist, expected_hist)
        self._last_condition_diagnostics = {}
        self._validate_historical_modalities(
            m_hist,
            t_hist,
            h_hist.shape[0],
        )

        base_state = self._base_state(h_hist, c_hist)
        if t_hist is None:
            state_history = base_state
        else:
            if self.time_residual_gate_logit is None:
                raise RuntimeError("time residual gate was not initialized")
            time_delta = self._historical_condition_fusion(
                h_hist=h_hist,
                c_hist=c_hist,
                t_hist=t_hist,
            )
            gate = torch.sigmoid(self.time_residual_gate_logit)
            effective_delta = gate * time_delta
            state_history = base_state + effective_delta
            base_norm = torch.linalg.vector_norm(base_state.detach())
            delta_norm = torch.linalg.vector_norm(time_delta.detach())
            effective_norm = torch.linalg.vector_norm(effective_delta.detach())
            self._last_condition_diagnostics.update(
                {
                    "time_residual_gate": gate.detach(),
                    "time_delta_norm": delta_norm,
                    "effective_time_delta_norm": effective_norm,
                    "ungated_time_contribution": (
                        delta_norm / base_norm.clamp_min(1e-12)
                    ),
                    "time_condition_contribution": (
                        effective_norm / base_norm.clamp_min(1e-12)
                    ),
                }
            )

        if self.text_residual_adapter is None:
            if m_hist is not None:
                raise ValueError(
                    "m_hist was provided, but modal conditioning is disabled"
                )
        else:
            if m_hist is None:
                raise ValueError(
                    "m_hist is required when modal conditioning is enabled"
                )
            modal_delta = self.text_residual_adapter(m_hist)
            state_norm = torch.linalg.vector_norm(state_history.detach())
            modal_norm = torch.linalg.vector_norm(m_hist.detach())
            delta_norm = torch.linalg.vector_norm(modal_delta.detach())
            state_history = state_history + modal_delta
            self._last_condition_diagnostics.update(
                {
                    "cfa_text_feature_norm": modal_norm,
                    "cfa_text_delta_norm": delta_norm,
                    "cfa_text_contribution": (
                        delta_norm / state_norm.clamp_min(1e-12)
                    ),
                }
            )
        return self.patch_predictor(state_history)

    def forward(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor] = None,
        m_hist: Optional[torch.Tensor] = None,
        t_hist: Optional[torch.Tensor] = None,
        c_future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        u_base = self.autonomous_trajectory(
            h_hist=h_hist,
            c_hist=c_hist,
            m_hist=m_hist,
            t_hist=t_hist,
        )
        if not self.use_future_x_condition:
            if c_future is not None:
                raise ValueError(
                    "c_future must be None when future conditioning is disabled"
                )
            return u_base

        expected_future = (
            h_hist.shape[0],
            self.horizon,
            self.condition_dim,
        )
        if c_future is None:
            return u_base
        if tuple(c_future.shape) != expected_future:
            raise ValueError(
                f"c_future must be {expected_future}, got {tuple(c_future.shape)}"
            )
        if self.control_mlp is None:
            raise RuntimeError("future control branch was not initialized")
        delta_future = self.control_mlp(
            torch.cat([u_base, c_future], dim=-1)
        )
        h_pred = u_base + delta_future
        if not torch.isfinite(h_pred).all():
            raise ValueError("h_pred contains NaN or Inf")
        return h_pred

    def condition_diagnostics(self) -> dict:
        return dict(self._last_condition_diagnostics)
