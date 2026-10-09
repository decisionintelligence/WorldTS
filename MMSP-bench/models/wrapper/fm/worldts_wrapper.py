"""Metadata wrapper allowing WorldTS to reuse the TS adapter data pipeline."""

from gluonts.model.forecast_generator import SampleForecastGenerator
from torch import nn

from .base import FMWrapperBase


class WorldTSWrapper(FMWrapperBase):
    def __init__(
        self,
        model_path,
        ds_freq: str,
        prediction_length: int,
        context_length: int,
        **kwargs,
    ) -> None:
        super().__init__()
        del model_path, kwargs
        self.ds_freq = ds_freq
        self.prediction_length = int(prediction_length)
        self._context_length = int(context_length)
        self.model = nn.Identity()

    @property
    def context_length(self):
        return self._context_length

    @property
    def quantiles(self):
        return [0.5]

    @property
    def model_dim(self):
        return 0

    def freeze_model(self):
        return None

    def forecast_generator(self):
        return SampleForecastGenerator()

    def tokenize(self, inputs):
        raise RuntimeError("WorldTS does not tokenize through a foundation model")

    def encode(self, inputs):
        raise RuntimeError("WorldTS does not use a foundation-model encoder")

    def predict(self, inputs):
        raise RuntimeError("WorldTS predicts through its latent PatchTST")

    def loss(self, inputs):
        raise RuntimeError("WorldTS loss is handled by its Lightning module")

    def process_forecast(self, outputs):
        return outputs
