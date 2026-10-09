"""The image frontend used by UniCA's MMSP experiments."""

import torch
from torch import nn


class UniCAImageHomogenizer(nn.Module):
    """Map raw four-channel satellite frames to aligned pseudo-series.

    The layer sequence intentionally mirrors ``models.adapter.unica.module``.
    MMSP batches are expected after the instance splitter, in
    ``[batch, time, channel, height, width]`` order.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        homogenizer_type: str = "linear",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.satellite_encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (64 // 4) * (64 // 4), self.hidden_dim),
        )
        if homogenizer_type == "linear":
            self.homogenization_projection = nn.Linear(
                self.hidden_dim,
                self.output_dim,
            )
        elif homogenizer_type == "mlp":
            self.homogenization_projection = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            raise ValueError(
                "homogenizer_type must be 'linear' or 'mlp', "
                f"got {homogenizer_type!r}"
            )

    def forward(self, satellite_data: torch.Tensor) -> torch.Tensor:
        expected_tail = (4, 64, 64)
        if satellite_data.dim() != 5:
            raise ValueError(
                "satellite_data must be [B, T, 4, 64, 64], "
                f"got {tuple(satellite_data.shape)}"
            )
        if tuple(satellite_data.shape[2:]) != expected_tail:
            raise ValueError(
                f"satellite frame shape must be {expected_tail}, "
                f"got {tuple(satellite_data.shape[2:])}"
            )
        if not torch.isfinite(satellite_data).all():
            raise ValueError("satellite_data contains NaN or Inf")

        batch_size, history_length = satellite_data.shape[:2]
        # ``MixedRandomDataset`` uses NumPy ``transpose()`` on the source
        # [H,W,C,T] field, producing [T,C,W,H].  UniCA restores [C,H,W]
        # through its permute/rearrange pair.  Keep that exact axis operation
        # even though MMSP frames are square and their shape alone cannot
        # reveal a swapped spatial axis.
        frames = satellite_data.transpose(-1, -2).reshape(
            batch_size * history_length,
            *expected_tail,
        )
        features = self.satellite_encoder(frames)
        image_tokens = self.homogenization_projection(features)
        image_tokens = image_tokens.reshape(
            batch_size,
            history_length,
            self.output_dim,
        )
        if not torch.isfinite(image_tokens).all():
            raise ValueError("image homogenizer produced NaN or Inf")
        return image_tokens
