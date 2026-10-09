"""Projection head used by the WorldTS VISReg path."""

import torch
import torch.nn.functional as F
from torch import nn


class VISRegProjector(nn.Module):
    """Three-layer MLP projector matching the VISReg training architecture."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.projection_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 3:
            raise ValueError(
                f"states must be [B, N, D], got {tuple(states.shape)}"
            )
        if states.shape[-1] != self.input_dim:
            raise ValueError(
                f"states feature dimension must be {self.input_dim}, "
                f"got {states.shape[-1]}"
            )
        batch_size, num_tokens, _ = states.shape
        projected = states.reshape(-1, self.input_dim)
        for layer in self.layers:
            if (
                isinstance(layer, nn.BatchNorm1d)
                and layer.training
                and projected.shape[0] == 1
            ):
                projected = F.batch_norm(
                    projected,
                    layer.running_mean,
                    layer.running_var,
                    layer.weight,
                    layer.bias,
                    training=False,
                    momentum=layer.momentum,
                    eps=layer.eps,
                )
            else:
                projected = layer(projected)
        return projected.reshape(batch_size, num_tokens, self.projection_dim)
