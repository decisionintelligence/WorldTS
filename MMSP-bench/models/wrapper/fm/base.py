import logging
from abc import ABC
from typing import List

from torch import nn

logger = logging.getLogger(__name__)


class FMWrapperBase(nn.Module, ABC):
    """Base Foundation Model Wrapper Class

    defining standard methods and properties that foundation models should implement.

    Key features:
    - Provides a standardized model interface for time series forecasting
    - Defines a tokenize-encode-predict-decode workflow
    - Supports quantile forecasting and loss computation

    Subclasses must implement all methods marked with NotImplementedError.
    """

    @property
    def quantiles(self) -> List:
        raise NotImplementedError

    @property
    def model_dim(self):
        raise NotImplementedError

    def tokenize(self, inputs):
        raise NotImplementedError

    def encode(self, inputs):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def loss(self, inputs):
        raise NotImplementedError

    @property
    def context_length(self):
        raise NotImplementedError

    def forecast_generator(self):
        raise NotImplementedError

    def process_forecast(self, outputs):
        raise NotImplementedError

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles)
