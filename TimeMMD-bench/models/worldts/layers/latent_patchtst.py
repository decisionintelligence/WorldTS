"""Latent PatchTST state-transition predictor and conditioning."""

from typing import Optional

import torch
from torch import nn

from layers.Embed import PositionalEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer

from .deterministic_padding import deterministic_replicate_pad_1d
from .multimodal_fusion import CFAResidualAdapter


class PatchEmbedding(nn.Module):
    """PatchTST embedding with deterministic replicated-end padding."""

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

    def _gated_historical_condition_fusion(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
        t_hist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        base_state = self._base_state(h_hist, c_hist)
        time_delta = self._historical_condition_fusion(
            h_hist=h_hist,
            c_hist=c_hist,
            t_hist=t_hist,
        )
        gate = torch.sigmoid(self.time_residual_gate_logit)
        effective_delta = gate * time_delta
        return base_state + effective_delta

    def _cfa_text_residual_fusion(
        self,
        state_history: torch.Tensor,
        m_hist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.text_residual_adapter is None:
            return state_history
        return state_history + self.text_residual_adapter(m_hist)

    def autonomous_trajectory(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
        m_hist: Optional[torch.Tensor] = None,
        t_hist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        return self.patch_predictor(state_history)

    def _base_state(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if c_hist is None:
            return h_hist
        return self.hist_fusion_mlp(torch.cat([h_hist, c_hist], dim=-1))

    def _historical_condition_fusion(
        self,
        h_hist: torch.Tensor,
        c_hist: Optional[torch.Tensor],
        t_hist: Optional[torch.Tensor],
    ) -> torch.Tensor:
        conditions = []
        if self.condition_dim > 0:
            conditions.append(c_hist)
        if t_hist is not None:
            conditions.append(t_hist)

        condition = torch.cat(conditions, dim=-1)
        projected_condition = self.condition_projection(condition)
        return self.historical_context_fusion(
            torch.cat([h_hist, projected_condition], dim=-1)
        )

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
            h_pred = u_base
        elif c_future is None:
            h_pred = u_base
        else:
            control_input = torch.cat([u_base, c_future], dim=-1)
            delta = self.control_mlp(control_input)
            h_pred = u_base + delta

        return h_pred


class LatentPatchTSTCore(nn.Module):
    """Forecast latent coordinates from shared PatchTST temporal patches."""

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

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        batch_size = state_history.shape[0]
        latent_channels = state_history.transpose(1, 2)
        patch_tokens, _ = self.patch_embedding(latent_channels)
        encoded, _ = self.encoder(
            patch_tokens,
            attn_mask=None,
        )
        encoded = encoded.reshape(
            batch_size,
            self.latent_dim,
            self.patch_num,
            self.d_model,
        )
        encoded = encoded.permute(0, 1, 3, 2)
        encoded = encoded.flatten(start_dim=-2)
        return self.head(encoded).transpose(1, 2)
