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
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class CFAResidualAdapter(nn.Module):
    """Map text features to a zero-initialized low-rank latent residual."""

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
        hidden = self.down_projection(inputs)
        hidden = self.normalization(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.up_projection(hidden)
