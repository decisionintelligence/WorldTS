"""Self-contained latent PatchTST state-transition predictor.

Historical conditioning, constrained text correction, future control, and
the PatchTST temporal projection live here without model inheritance.
"""

from typing import Optional

import torch
from torch import nn

from .multimodal_fusion import CFAResidualAdapter
from .patchtst_primitives import (
    AttentionLayer,
    Encoder,
    EncoderLayer,
    FullAttention,
    PatchEmbedding,
)


class PatchTSTStateTransitionPredictor(nn.Module):
    """Predict a full-resolution future latent trajectory."""

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
        for name, value in (
            ("history_length", self.history_length),
            ("horizon", self.horizon),
            ("latent_dim", self.latent_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.condition_dim < 0:
            raise ValueError(
                f"condition_dim must be non-negative, got {self.condition_dim}"
            )
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
            historical_context_dim = (
                self.condition_dim
                + self.time_condition_dim
            )
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
            nn.init.zeros_(self.control_mlp[-1].weight)
            nn.init.zeros_(self.control_mlp[-1].bias)
        else:
            self.control_mlp = None
    @staticmethod
    def _check_shape(name: str, x: torch.Tensor, expected: tuple) -> None:
        if x.dim() != 3 or tuple(x.shape) != expected:
            raise ValueError(f"{name} must be {expected}, got {tuple(x.shape)}")

    def _gated_historical_condition_fusion(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
        t_hist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.time_residual_gate_logit is None:
            raise RuntimeError(
                "gated historical fusion requires timestamp conditioning"
            )
        base_state = self._base_state(h_hist, c_hist)
        time_delta = self._historical_condition_fusion(
            h_hist=h_hist,
            c_hist=c_hist,
            t_hist=t_hist,
        )
        gate = torch.sigmoid(self.time_residual_gate_logit)
        effective_delta = gate * time_delta
        state_history = base_state + effective_delta
        return state_history

    def _cfa_text_residual_fusion(
        self,
        state_history: torch.Tensor,
        m_hist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.text_residual_adapter is None:
            if m_hist is not None:
                raise ValueError(
                    "m_hist was provided, but text conditioning is disabled"
                )
            return state_history
        if m_hist is None:
            raise ValueError(
                "m_hist is required when text conditioning is enabled"
            )

        text_delta = self.text_residual_adapter(m_hist)
        corrected_state = state_history + text_delta
        return corrected_state

    def autonomous_trajectory(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
        m_hist: Optional[torch.Tensor] = None,
        t_hist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h_hist.dim() != 3:
            raise ValueError(
                f"h_hist must be [B, T, D], got {tuple(h_hist.shape)}"
            )
        batch_size = h_hist.shape[0]
        self._check_shape(
            "h_hist",
            h_hist,
            (batch_size, self.history_length, self.latent_dim),
        )
        self._validate_historical_modalities(
            m_hist,
            t_hist,
            batch_size,
        )

        if t_hist is None:
            state_history = self._base_state(h_hist, c_hist)
        else:
            state_history = self._gated_historical_condition_fusion(
                h_hist=h_hist,
                c_hist=c_hist,
                t_hist=t_hist,
            )

        state_history = self._cfa_text_residual_fusion(
            state_history,
            m_hist,
        )

        expected = (batch_size, self.history_length, self.latent_dim)
        if tuple(state_history.shape) != expected:
            raise RuntimeError(
                "historical fusion changed the latent history shape: "
                f"{tuple(state_history.shape)} != {expected}"
            )
        return self.patch_predictor(state_history)

    def _base_state(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if c_hist is None:
            return h_hist
        if self.condition_dim <= 0 or self.hist_fusion_mlp is None:
            raise ValueError(
                "c_hist was provided, but the predictor has no "
                "covariate branch"
            )
        self._check_shape(
            "c_hist",
            c_hist,
            (
                h_hist.shape[0],
                self.history_length,
                self.condition_dim,
            ),
        )
        return self.hist_fusion_mlp(torch.cat([h_hist, c_hist], dim=-1))

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
                (
                    batch_size,
                    self.history_length,
                    self.modal_condition_dim,
                ),
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
                (
                    batch_size,
                    self.history_length,
                    self.time_condition_dim,
                ),
            )
        elif t_hist is not None:
            raise ValueError(
                "t_hist was provided, but time conditioning is disabled"
            )

    def _historical_condition_fusion(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
        t_hist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.condition_projection is None
            or self.historical_context_fusion is None
        ):
            raise ValueError(
                "historical condition fusion was not configured"
            )
        batch_size = h_hist.shape[0]
        conditions = []
        if self.condition_dim > 0:
            if c_hist is None:
                raise ValueError(
                    "c_hist is required by this multimodal predictor "
                    "configuration"
                )
            self._check_shape(
                "c_hist",
                c_hist,
                (batch_size, self.history_length, self.condition_dim),
            )
            conditions.append(c_hist)
        elif c_hist is not None:
            raise ValueError(
                "c_hist was provided, but condition_dim is zero"
            )
        if t_hist is not None:
            conditions.append(t_hist)

        condition = torch.cat(conditions, dim=-1)
        projected_condition = self.condition_projection(condition)
        state_history = self.historical_context_fusion(
            torch.cat([h_hist, projected_condition], dim=-1)
        )
        return state_history

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
                    "c_future must be None when use_future_x_condition=False"
                )
            h_pred = u_base
        elif c_future is None:
            h_pred = u_base
        else:
            if c_hist is None:
                raise ValueError(
                    "c_future cannot be used without historical covariates"
                )
            self._check_shape(
                "c_future",
                c_future,
                (h_hist.shape[0], self.horizon, self.condition_dim),
            )
            control_input = torch.cat([u_base, c_future], dim=-1)
            if self.control_mlp is None:
                raise RuntimeError("future control branch was not initialized")
            delta = self.control_mlp(control_input)
            h_pred = u_base + delta

        if not torch.isfinite(h_pred).all():
            raise ValueError("h_pred contains NaN or Inf")
        return h_pred

class LatentPatchTSTCore(nn.Module):
    """Forecast every latent coordinate from temporal patches.

    Latent coordinates are treated as PatchTST channels.  Patch embedding,
    Transformer weights, and the forecast head are shared across coordinates.
    Historical modalities must therefore be fused into ``state_history``
    before this module is called.
    """

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
    ) -> None:
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
            raise ValueError(
                "patch_len must not exceed history_length; "
                f"got patch_len={self.patch_len}, "
                f"history_length={self.history_length}"
            )
        if self.stride > self.patch_len:
            raise ValueError(
                "stride must not exceed patch_len; "
                f"got stride={self.stride}, patch_len={self.patch_len}"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads; "
                f"got d_model={self.d_model}, n_heads={self.n_heads}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )
        if self.activation not in {"relu", "gelu"}:
            raise ValueError(
                "activation must be 'relu' or 'gelu'; "
                f"got {self.activation!r}"
            )

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
        self.head = nn.Linear(
            self.d_model * self.patch_num,
            self.horizon,
        )
        self._last_attentions = None

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        if state_history.dim() != 3:
            raise ValueError(
                "state_history must be [B, L, D], "
                f"got {tuple(state_history.shape)}"
            )
        batch_size = state_history.shape[0]
        expected = (
            batch_size,
            self.history_length,
            self.latent_dim,
        )
        if tuple(state_history.shape) != expected:
            raise ValueError(
                f"state_history must be {expected}, "
                f"got {tuple(state_history.shape)}"
            )
        if not torch.isfinite(state_history).all():
            raise ValueError("state_history contains NaN or Inf")

        latent_channels = state_history.transpose(1, 2)
        patch_tokens, variable_count = self.patch_embedding(latent_channels)
        if variable_count != self.latent_dim:
            raise RuntimeError(
                f"expected {self.latent_dim} latent channels, "
                f"got {variable_count}"
            )
        if patch_tokens.shape[1] != self.patch_num:
            raise RuntimeError(
                f"expected {self.patch_num} patches, "
                f"got {patch_tokens.shape[1]}"
            )

        encoded, attentions = self.encoder(
            patch_tokens,
            attn_mask=None,
        )
        self._last_attentions = [
            attention.detach() if attention is not None else None
            for attention in attentions
        ]
        encoded = encoded.reshape(
            batch_size,
            self.latent_dim,
            self.patch_num,
            self.d_model,
        )
        encoded = encoded.permute(0, 1, 3, 2)
        encoded = encoded.flatten(start_dim=-2)
        future_latent = self.head(encoded).transpose(1, 2)

        expected_output = (
            batch_size,
            self.horizon,
            self.latent_dim,
        )
        if tuple(future_latent.shape) != expected_output:
            raise RuntimeError(
                f"latent PatchTST returned {tuple(future_latent.shape)}, "
                f"expected {expected_output}"
            )
        if not torch.isfinite(future_latent).all():
            raise ValueError("future_latent contains NaN or Inf")
        return future_latent
