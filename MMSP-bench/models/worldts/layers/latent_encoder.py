"""Observation-to-state encoders for the latent PatchTST model."""

import torch
from torch import nn

from .deterministic_padding import deterministic_replicate_pad_1d


class PointwiseLatentEncoder(nn.Module):
    """Map each time step to a latent vector without temporal mixing."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B, L, C], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input channels {self.input_dim}, got {x.shape[-1]}"
            )

        z = self.net(x)
        if not torch.isfinite(z).all():
            raise ValueError("pointwise latent encoder produced NaN or Inf")
        return z


class CausalPatchLatentEncoder(nn.Module):
    """Construct each latent state from a trailing observation patch.

    A stride-one causal unfold preserves the input sequence length.  The
    patch projection supplies local temporal context, while a residual FFN
    refines every resulting state without adding normalization on top of the
    model's existing RevIN input normalization.
    """

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
        patches = channel_first.unfold(
            dimension=-1,
            size=self.patch_len,
            step=1,
        )
        patches = patches.permute(0, 2, 1, 3).flatten(start_dim=-2)
        expected = (
            x.shape[0],
            x.shape[1],
            self.input_dim * self.patch_len,
        )
        if tuple(patches.shape) != expected:
            raise RuntimeError(
                f"causal patches must be {expected}, got {tuple(patches.shape)}"
            )
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B, L, C], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input channels {self.input_dim}, got {x.shape[-1]}"
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
