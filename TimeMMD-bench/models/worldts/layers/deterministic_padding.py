"""Deterministic replicated padding for the WorldTS patch-state model."""

import torch


def deterministic_replicate_pad_1d(
    value: torch.Tensor,
    *,
    left: int = 0,
    right: int = 0,
) -> torch.Tensor:
    """Replicate boundary values without CUDA replication-pad backward."""

    left = int(left)
    right = int(right)
    if left == 0 and right == 0:
        return value
    parts = []
    if left:
        parts.append(value[..., :1].expand(*value.shape[:-1], left))
    parts.append(value)
    if right:
        parts.append(value[..., -1:].expand(*value.shape[:-1], right))
    return torch.cat(parts, dim=-1)
