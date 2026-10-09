from typing import Iterable

import numpy as np
from gluonts.dataset import Dataset, DataEntry
from gluonts.model import Estimator
from gluonts.torch import PyTorchPredictor
from gluonts.transform import (
    SimpleTransformation,
    DummyValueImputation, Transformation)
from lightning import pytorch as pl


class AllFieldImputation(SimpleTransformation):

    def __init__(self, imputation=DummyValueImputation()):
        self.imputation = imputation

    def transform(self, data: DataEntry) -> DataEntry:
        for field in data:
            if isinstance(data[field], np.ndarray) and np.issubdtype(data[field].dtype, np.number):
                data[field] = self.imputation(data[field])
        return data


class ImputationEstimatorWrapper(Estimator):
    def __init__(self, estimator,
                 imputation=DummyValueImputation()) -> None:
        super().__init__(estimator.lead_time)
        self.estimator = estimator
        self.imputation = imputation

    def create_transformation(self) -> Transformation:
        transformations = self.estimator.create_transformation()
        return transformations + AllFieldImputation(self.imputation)

    def create_lightning_module(self) -> pl.LightningModule:
        return self.estimator.create_lightning_module()

    def create_predictor(self, transformation: Transformation, module) -> PyTorchPredictor:
        return self.estimator.create_predictor(transformation, module)

    def create_training_data_loader(self, data: Dataset, module, **kwargs) -> Iterable:
        dset = self.estimator.create_training_data_loader(data, module, **kwargs)
        return dset

    def create_validation_data_loader(self, data: Dataset, module, **kwargs) -> Iterable:
        return self.estimator.create_validation_data_loader(data, module, **kwargs)
