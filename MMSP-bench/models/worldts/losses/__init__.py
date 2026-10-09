"""Losses used by WorldTS two-stage training."""

from .latent_state_loss import LatentStateLoss
from .sigreg_loss import SIGRegLoss

__all__ = ["LatentStateLoss", "SIGRegLoss"]
