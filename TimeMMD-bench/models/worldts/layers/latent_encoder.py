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
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalPatchLatentEncoder(nn.Module):
    """Construct each latent state from a trailing observation patch."""

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
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.patch_projection(self._causal_patches(x))
        state = state + self.state_ffn(state)
        return state
