import logging

import lightning.pytorch as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gluonts.core.component import validated
from gluonts.itertools import select
from gluonts.torch.model.lightning_util import has_validation_loop

logger = logging.getLogger(__name__)


def get_model(name, **kwargs):
    if name == "unica":
        from models.adapter.unica.module import UniCA
        logger.info("Using Conditional Token model")
        return UniCA(**kwargs)
    elif name == "sft":
        from .sft.module import SFTModel
        logger.info("Using SFT model")
        return SFTModel(**kwargs)
    elif name == "worldts_image":
        from .worldts_image.module import WorldTSImageModule
        logger.info("Using image-conditioned WorldTS model")
        return WorldTSImageModule(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")


class TSAdapterLightningModule(pl.LightningModule):
    """
    A ``pl.LightningModule`` class that can be used to train a
    ``TemporalFusionTransformerModel`` with PyTorch Lightning.

    This is a thin layer around a (wrapped) ``TemporalFusionTransformerModel``
    object, that exposes the methods to evaluate training and validation loss.

    Parameters
    ----------
    model_kwargs
        Keyword arguments to construct the ``TemporalFusionTransformerModel`` to be trained.
    lr
        Learning rate.
    weight_decay
        Weight decay regularization parameter.
    patience
        Patience parameter for learning rate scheduler.
    """

    @validated()
    def __init__(
            self,
            model_kwargs: dict,
            lr: float = 1e-3,
            patience: int = 10,
            weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        # name = model_kwargs.pop("module_name")
        name = model_kwargs["module_name"]
        # model_kwargs["model_wrapper"] = model_wrapper
        self.model = get_model(name, **model_kwargs)
        # self.model_wrapper = model_wrapper
        self.lr = lr
        self.patience = patience
        self.weight_decay = weight_decay
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute training step.
        """
        train_loss = self.model.loss(
            **select(self.inputs, batch, ignore_missing=True),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        ).mean()

        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        """
        Execute validation step.
        """
        val_loss = self.model.loss(
            **select(self.inputs, batch, ignore_missing=True),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        ).mean()

        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        monitor = (
            "val_loss" if has_validation_loop(self.trainer) else "train_loss"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.patience,
                ),
                "monitor": monitor,
            },
        }
