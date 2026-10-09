"""The unchanged pointwise decoder used by the VoT-side model."""

import torch
from torch import nn


class PointwiseMLPStateToObservationDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        if self.latent_dim <= 0 or self.output_dim <= 0:
            raise ValueError("latent_dim and output_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.output_dim),
        )

    def forward(self, h_states: torch.Tensor) -> torch.Tensor:
        if h_states.dim() != 3 or h_states.shape[-1] != self.latent_dim:
            raise ValueError(
                f"h_states must be [B, L, {self.latent_dim}], "
                f"got {tuple(h_states.shape)}"
            )
        output = self.net(h_states)
        if not torch.isfinite(output).all():
            raise ValueError("pointwise MLP decoder produced NaN or Inf")
        return output


class CausalPatchStateToObservationDecoder(nn.Module):
    """Decode every future point from a stride-one trailing latent patch."""

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
        for name, value in (
            ("latent_dim", self.latent_dim),
            ("output_dim", self.output_dim),
            ("patch_len", self.patch_len),
            ("hidden_dim", self.hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

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
            if history_states.shape[1] < prefix_len:
                raise ValueError(
                    "history_states is shorter than the decoder prefix: "
                    f"{history_states.shape[1]} < {prefix_len}"
                )
            states = torch.cat(
                (history_states[:, -prefix_len:, :], future_states),
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
        for name, value in (
            ("future_states", future_states),
            ("history_states", history_states),
        ):
            if value.dim() != 3 or value.shape[-1] != self.latent_dim:
                raise ValueError(
                    f"{name} must be [B, L, {self.latent_dim}], "
                    f"got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if history_states.shape[0] != future_states.shape[0]:
            raise ValueError("history_states and future_states batch sizes differ")

        state = self.patch_projection(
            self._future_patches(future_states, history_states)
        )
        state = state + self.state_ffn(state)
        output = self.output_projection(state)
        expected = (
            future_states.shape[0],
            future_states.shape[1],
            self.output_dim,
        )
        if tuple(output.shape) != expected:
            raise RuntimeError(
                f"causal patch decoder output must be {expected}, "
                f"got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise ValueError("causal patch decoder produced NaN or Inf")
        return output
