"""Numerical observation-to-state encoders."""

import torch
from torch import nn

from ts_benchmark.baselines.worldts.layers.deterministic_padding import (
    deterministic_replicate_pad_1d,
)


class PointwiseLatentEncoder(nn.Module):
    """Map each covariate time step without temporal mixing."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x must be [B, L, {self.input_dim}], got {tuple(x.shape)}"
            )
        z = self.net(x)
        if not torch.isfinite(z).all():
            raise ValueError("pointwise latent encoder produced NaN or Inf")
        return z


class CausalPatchLatentEncoder(nn.Module):
    """Encode every target state from a stride-one trailing patch."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        patch_len: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.patch_len = int(patch_len)
        self.hidden_dim = int(hidden_dim)
        for name, value in (
            ("input_dim", self.input_dim),
            ("output_dim", self.output_dim),
            ("patch_len", self.patch_len),
            ("hidden_dim", self.hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.patch_projection = nn.Linear(
            self.input_dim * self.patch_len,
            self.output_dim,
        )
        self.state_ffn = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def _causal_patches(self, x: torch.Tensor) -> torch.Tensor:
        channel_first = x.transpose(1, 2)
        channel_first = deterministic_replicate_pad_1d(
            channel_first,
            left=self.patch_len - 1,
        )
        patches = channel_first.unfold(-1, self.patch_len, 1)
        return patches.permute(0, 2, 1, 3).flatten(start_dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"x must be [B, L, {self.input_dim}], got {tuple(x.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("causal patch encoder input contains NaN or Inf")
        state = self.patch_projection(self._causal_patches(x))
        state = state + self.state_ffn(state)
        expected = (x.shape[0], x.shape[1], self.output_dim)
        if tuple(state.shape) != expected:
            raise RuntimeError(
                f"causal patch state must be {expected}, got {tuple(state.shape)}"
            )
        if not torch.isfinite(state).all():
            raise ValueError("causal patch latent encoder produced NaN or Inf")
        return state
