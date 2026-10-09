"""Stage-explicit Lightning training for the MMSP WorldTS adapter."""

import lightning.pytorch as pl
import torch
from gluonts.itertools import select

from .module import WorldTSImageModule


class WorldTSImageLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_kwargs: dict,
        stage1_epochs: int = 20,
        stage2_epochs: int = 20,
        stage1_lr: float = 1e-4,
        stage2_lr: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        clean_kwargs = dict(model_kwargs)
        clean_kwargs.pop("module_name", None)
        self.model = WorldTSImageModule(**clean_kwargs)
        self.stage1_epochs = int(stage1_epochs)
        self.stage2_epochs = int(stage2_epochs)
        self.stage1_lr = float(stage1_lr)
        self.stage2_lr = float(stage2_lr)
        self.weight_decay = float(weight_decay)
        if self.stage1_epochs <= 0 or self.stage2_epochs <= 0:
            raise ValueError("both WorldTS stages require positive epochs")
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()
        self._active_stage = 1
        self.set_stage(1)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def stage(self) -> int:
        return self._active_stage

    def set_stage(self, stage: int) -> None:
        if stage == 1:
            self.model.model.configure_for_stage1()
        elif stage == 2:
            self.model.model.configure_for_stage2()
            self.model.model.set_stage1_eval()
        else:
            raise ValueError(f"unsupported WorldTS stage: {stage}")
        self._active_stage = stage

    def _batch_inputs(self, batch):
        return select(self.inputs, batch, ignore_missing=True)

    def on_train_epoch_start(self) -> None:
        # Lightning calls ``train()`` at every epoch boundary. Re-assert the
        # frozen Stage-1 modules' evaluation mode while training Stage 2.
        self.set_stage(self.stage)

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        del batch, batch_idx
        if self.stage == 2:
            self.model.model.set_stage1_eval()

    def training_step(self, batch, batch_idx: int):
        del batch_idx
        inputs = self._batch_inputs(batch)
        targets = {
            "future_target": batch["future_target"],
            "future_observed_values": batch["future_observed_values"],
        }
        if self.stage == 1:
            loss, _ = self.model.stage1_loss(**inputs, **targets)
        else:
            loss, _ = self.model.stage2_loss(**inputs, **targets)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            f"stage{self.stage}_train_loss",
            loss,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, batch_idx: int):
        del batch_idx
        inputs = self._batch_inputs(batch)
        targets = {
            "future_target": batch["future_target"],
            "future_observed_values": batch["future_observed_values"],
        }
        if self.stage == 1:
            raw_loss, _ = self.model.stage1_loss(**inputs, **targets)
            self.log("stage1_val_loss", raw_loss, on_epoch=True)
        else:
            raw_loss = self.model.validation_forecast_loss(
                **inputs,
                **targets,
            )
            self.log("stage2_val_loss", raw_loss, on_epoch=True)
        self.log(
            "val_loss",
            raw_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return raw_loss

    def configure_optimizers(self):
        learning_rate = self.stage1_lr if self.stage == 1 else self.stage2_lr
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError(
                f"WorldTS Stage {self.stage} has no trainable parameters"
            )
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=self.weight_decay,
        )
