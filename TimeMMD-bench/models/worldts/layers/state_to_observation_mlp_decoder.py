"""Pointwise observation decoder adapted for the WorldTS model."""

import torch
from torch import nn


class PointwiseMLPStateToObservationDecoder(nn.Module):
    """Decode each future latent state with one shared two-layer MLP."""

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.output_dim),
        )

    def forward(self, h_states: torch.Tensor) -> torch.Tensor:
        return self.net(h_states)


class CausalPatchStateToObservationDecoder(nn.Module):
    """Decode each future point from a stride-one trailing latent patch."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        patch_len: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.patch_len = int(patch_len)
        self.hidden_dim = int(hidden_dim)
        self.patch_projection = nn.Linear(
            self.latent_dim * self.patch_len,
            self.latent_dim,
        )
        self.state_ffn = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.output_projection = nn.Linear(
            self.latent_dim,
            self.output_dim,
        )

    def _future_patches(
        self,
        future_states: torch.Tensor,
        history_states: torch.Tensor,
    ) -> torch.Tensor:
        prefix_len = self.patch_len - 1
        if prefix_len:
            states = torch.cat(
                [history_states[:, -prefix_len:, :], future_states],
                dim=1,
            )
        else:
            states = future_states
        patches = states.transpose(1, 2).unfold(-1, self.patch_len, 1)
        return patches.permute(0, 2, 1, 3).flatten(start_dim=-2)

    def forward(
        self,
        future_states: torch.Tensor,
        history_states: torch.Tensor,
    ) -> torch.Tensor:
        state = self.patch_projection(
            self._future_patches(future_states, history_states)
        )
        state = state + self.state_ffn(state)
        return self.output_projection(state)
