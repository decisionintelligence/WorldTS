"""Historical condition encoders and constrained residual adapters."""

import torch
from torch import nn


class HistoricalMLPEncoder(nn.Module):
    """Project one point-aligned historical condition into a compact space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        for name, value in (
            ("input_dim", self.input_dim),
            ("output_dim", self.output_dim),
            ("hidden_dim", self.hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError(
                "historical condition must be [B, L, C], "
                f"got {tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"historical condition feature dim must be {self.input_dim}, "
                f"got {inputs.shape[-1]}"
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("historical condition contains NaN or Inf")
        encoded = self.projection(inputs)
        if not torch.isfinite(encoded).all():
            raise ValueError("historical condition encoding contains NaN or Inf")
        return encoded


class CFAResidualAdapter(nn.Module):
    """Map text features to a zero-initialized low-rank latent residual.

    The zero-initialized up projection makes the adapter an exact identity
    perturbation at initialization: ``state + adapter(text) == state``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        reduction_factor: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.reduction_factor = int(reduction_factor)
        self.dropout_probability = float(dropout)
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.latent_dim <= 0:
            raise ValueError(
                f"latent_dim must be positive, got {self.latent_dim}"
            )
        if self.reduction_factor <= 0:
            raise ValueError(
                "reduction_factor must be positive, "
                f"got {self.reduction_factor}"
            )
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError(
                "dropout must be in [0, 1), "
                f"got {self.dropout_probability}"
            )

        self.bottleneck_dim = max(
            1,
            self.latent_dim // self.reduction_factor,
        )
        activation_name = str(activation).lower()
        if activation_name == "gelu":
            activation_layer = nn.GELU()
        elif activation_name == "relu":
            activation_layer = nn.ReLU()
        else:
            raise ValueError(
                "activation must be 'gelu' or 'relu', "
                f"got {activation!r}"
            )

        self.down_projection = nn.Linear(
            self.input_dim,
            self.bottleneck_dim,
        )
        self.normalization = nn.LayerNorm(self.bottleneck_dim)
        self.activation = activation_layer
        self.dropout = nn.Dropout(self.dropout_probability)
        self.up_projection = nn.Linear(
            self.bottleneck_dim,
            self.latent_dim,
        )
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError(
                f"text state must be [B, L, C], got {tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"text state feature dim must be {self.input_dim}, "
                f"got {inputs.shape[-1]}"
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("text state contains NaN or Inf")

        hidden = self.down_projection(inputs)
        hidden = self.normalization(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        residual = self.up_projection(hidden)
        if not torch.isfinite(residual).all():
            raise ValueError("CFA text residual contains NaN or Inf")
        return residual
