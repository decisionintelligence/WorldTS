# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import lightning.pytorch as pl
import lightning.pytorch.callbacks
import numpy as np
import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.env import env
from gluonts.itertools import Cached, Cyclic
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.torch.distributions import Output, QuantileOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator, TrainOutput
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddConstFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    LastValueImputation,
    RemoveFields,
    SetField,
    TestSplitSampler,
    ValidationSplitSampler,
    VstackFeatures, Identity, DummyValueImputation
)
from gluonts.transform import Transformation
from gluonts.transform.sampler import InstanceSampler
from gluonts.transform.split import TFTInstanceSplitter
from lightning.pytorch.loggers.logger import Logger as PLLogger
from toolz import compose

from data.loader import TrainDataLoader
from data.transform import AddItemIDFeature, LastValueImputation2D
from .lightning_module import TSAdapterLightningModule

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "item_index",
    "feat_static_real",
    "feat_static_cat",
    "feat_dynamic_real",
    "feat_dynamic_cat",
    "past_feat_dynamic_real",
    "past_feat_dynamic_cat",
    "past_observed_feat_dynamic_real",
    "past_observed_feat_dynamic_cat",
    "observed_feat_dynamic_real",
    "observed_feat_dynamic_cat",
    "satellite_data",
    "text_data",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]
logger = logging.getLogger(__name__)


def dict_to_string_id(d):
    """Convert a dictionary to a unique string ID using MD5 hash."""
    # Sort keys for consistent ordering
    serialized = json.dumps(d, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()


def get_fm_wrapper(model_type, model_id_or_path, ds_freq, prediction_length, only_quantile_loss=False,
                   normalized_loss=False, *args, **kwargs):
    if model_type == "chronos":
        from models.wrapper.fm.chronos_wrapper import ChronosWrapper
        return ChronosWrapper(model_id_or_path, ds_freq=ds_freq, prediction_length=prediction_length,
                              )
    elif model_type == "timesfm":
        from models.wrapper.fm.timesfm_wrapper import TimesFMWrapper
        return TimesFMWrapper(model_id_or_path, ds_freq=ds_freq, prediction_length=prediction_length,
                              only_quantile_loss=only_quantile_loss, normalized_loss=normalized_loss,
                              sample_output=kwargs["sample_output"])
    elif model_type == "worldts":
        from models.wrapper.fm.worldts_wrapper import WorldTSWrapper
        return WorldTSWrapper(
            model_id_or_path,
            ds_freq=ds_freq,
            prediction_length=prediction_length,
            context_length=kwargs["context_length"],
        )
    else:
        raise ValueError(f"Unknown model name: {model_type}")


class NaNCheckCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        # Check for NaN in gradients
        for name, param in pl_module.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)


class TransformationWrapper:

    def __init__(self, transformation: Transformation):
        self.transformation = transformation

    def __call__(self, entry):
        t_entry = list(self.transformation([entry], is_train=True))
        assert len(t_entry) == 1
        return t_entry[0]


class TSAdapterEstimator(PyTorchLightningEstimator):
    """
    Estimator class to train a Temporal Fusion Transformer (TFT) model, as
    described in [LAL+21]_.

    TFT internally performs feature selection when making forecasts. For this
    reason, the dimensions of real-valued features can be grouped together if
    they correspond to the same variable (e.g., treat weather features as a
    one feature and holiday indicators as another feature).

    For example, if the dataset contains key "feat_static_real" with shape
    [batch_size, 3], we can, e.g.,
    - set ``static_dims = [3]`` to treat all three dimensions as a single feature
    - set ``static_dims = [1, 1, 1]`` to treat each dimension as a separate feature
    - set ``static_dims = [2, 1]`` to treat the first two dims as a single feature

    See ``gluonts.torch.model.tft.TemporalFusionTransformerModel.input_shapes``
    for more details on how the model configuration corresponds to the expected
    input shapes.


    Parameters
    ----------
    freq
        Frequency of the data to train on and predict.
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of previous time series values provided as input to the encoder.
        (default: None, in which case context_length = prediction_length).
    quantiles
        List of quantiles that the model will learn to predict.
        Defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    distr_output
        Distribution output to use (default: ``QuantileOutput``).
    num_heads
        Number of attention heads in self-attention layer in the decoder.
    hidden_dim
        Size of the LSTM & transformer hidden states.
    variable_dim
        Size of the feature embeddings.
    static_dims
        Sizes of the real-valued static features.
    dynamic_dims
        Sizes of the real-valued dynamic features that are known in the future.
    past_dynamic_dims
        Sizes of the real-valued dynamic features that are only known in the past.
    static_cardinalities
        Cardinalities of the categorical static features.
    dynamic_cardinalities
        Cardinalities of the categorical dynamic features that are known in the future.
    past_dynamic_cardinalities
        Cardinalities of the categorical dynamic features that are ony known in the past.
    time_features
        List of time features, from :py:mod:`gluonts.time_feature`, to use as
        dynamic real features in addition to the provided data (default: None,
        in which case these are automatically determined based on freq).
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay (default: ``1e-8``).
    dropout_rate
        Dropout regularization parameter (default: 0.1).
    patience
        Patience parameter for learning rate scheduler.
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch: int = 50,
        Number of batches to be processed in each training epoch (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    """

    @validated()
    def __init__(
            self,
            model_id_or_path: str,
            freq: str,
            prediction_length: int,
            model_type: str = "chronos",
            context_length: Optional[int] = None,
            quantiles: Optional[List[float]] = None,
            distr_output: Optional[Output] = None,
            num_heads: int = 4,
            hidden_dim: int = 32,
            variable_dim: int = 32,
            static_dims: Optional[List[int]] = None,
            dynamic_dims: Optional[List[int]] = None,
            past_dynamic_dims: Optional[List[int]] = None,
            static_cardinalities: Optional[List[int]] = None,
            dynamic_cardinalities: Optional[List[int]] = None,
            past_dynamic_cardinalities: Optional[List[int]] = None,
            time_features: Optional[List[TimeFeature]] = None,
            lr: float = 1e-3,
            weight_decay: float = 1e-8,
            dropout_rate: float = 0.1,
            patience: int = 10,
            batch_size: int = 32,
            num_batches_per_epoch: int = 50,
            model_kwargs: Optional[Dict[str, Any]] = None,
            trainer_kwargs: Optional[Dict[str, Any]] = None,
            train_sampler: Optional[InstanceSampler] = None,
            validation_sampler: Optional[InstanceSampler] = None,
            pl_logger: Optional[PLLogger] = None,
            num_workers: int = 0,
            indexed_sample: bool = False,
            item_map: dict = None,
            module_name: str = "sft",
            imputation="last",
            args=None,
            ckpt_info=None,
            use_satellite=True,
            use_text=True,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": None,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)
        self.args = args
        self.model_wrapper = get_fm_wrapper(model_type, model_id_or_path, ds_freq=freq,
                                            prediction_length=prediction_length,
                                            only_quantile_loss=args.only_quantile_loss,
                                            normalized_loss=args.normalized_loss,
                                            sample_output=args.sample_output,
                                            context_length=context_length)
        self.model_kwargs = model_kwargs or {}
        self.module_name = module_name
        self.imputation = imputation
        self.freq = freq
        self.prediction_length = prediction_length
        self.num_workers = num_workers
        self.indexed_sample = indexed_sample
        self.item_map = item_map
        self.context_length = (
            context_length if context_length is not None else self.model_wrapper.context_length
        )
        # Model architecture
        if distr_output is not None and quantiles is not None:
            raise ValueError(
                "Only one of `distr_output` and `quantiles` must be specified"
            )
        elif distr_output is not None:
            self.distr_output = distr_output
        else:
            if quantiles is None:
                quantiles = self.model_wrapper.quantiles
            self.distr_output = QuantileOutput(quantiles=quantiles)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.variable_dim = variable_dim

        self.static_dims = static_dims or []
        self.dynamic_dims = dynamic_dims or []
        self.past_dynamic_dims = past_dynamic_dims or []
        self.static_cardinalities = static_cardinalities or []
        self.dynamic_cardinalities = dynamic_cardinalities or []
        self.past_dynamic_cardinalities = past_dynamic_cardinalities or []

        if time_features is None:
            time_features = time_features_from_frequency_str(self.freq)
        self.time_features = [] if args.wo_time else time_features

        # Training procedure
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.pl_logger = pl_logger
        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length, min_past=self.context_length
        )
        self.ckpt_info = ckpt_info
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

        self.use_satellite = use_satellite
        self.use_text = use_text

    def create_transformation(self) -> Transformation:
        # transforms = []
        remove_field_names = []

        if not self.static_dims:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.dynamic_dims:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if not self.past_dynamic_dims:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
        if not self.static_cardinalities:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)
        if not self.dynamic_cardinalities:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_CAT)
        if not self.past_dynamic_cardinalities:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_CAT)

        if self.imputation == "last":
            imputation_1d = LastValueImputation()
            imputation_2d = LastValueImputation2D()
        else:
            imputation_1d = DummyValueImputation()
            imputation_2d = DummyValueImputation()

        transforms = [
            RemoveFields(field_names=remove_field_names),
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
                imputation_method=imputation_1d,
            ),
        ]
        if len(self.time_features) > 0:
            transforms.append(
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                )
            )
        else:
            # Add dummy dynamic feature if no time features are available
            transforms.append(
                AddConstFeature(
                    output_field=FieldName.FEAT_TIME,
                    target_field=FieldName.TARGET,
                    pred_length=self.prediction_length,
                    const=0.0,
                )
            )

        # Provide dummy values if static features are missing
        if not self.static_dims:
            transforms.append(
                SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0])
            )
        transforms.append(
            AsNumpyArray(field=FieldName.FEAT_STATIC_REAL, expected_ndim=1)
        )


        transforms.append(AddItemIDFeature(output_field="item_index", item_map=self.item_map))
        transforms.append(
            AsNumpyArray(
                field="item_index",
                expected_ndim=1,
                dtype=np.int64,
            )
        )
        if not self.static_cardinalities:
            transforms.append(
                SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0])
            )

        transforms.append(
            AsNumpyArray(
                field=FieldName.FEAT_STATIC_CAT,
                expected_ndim=1,
                dtype=np.int64,
            )
        )

        # Concat time features with known dynamic features
        input_fields = [FieldName.FEAT_TIME]
        if self.dynamic_dims:
            input_fields += [FieldName.FEAT_DYNAMIC_REAL]
        transforms.append(
            VstackFeatures(
                input_fields=input_fields,
                output_field=FieldName.FEAT_DYNAMIC_REAL,
            )
        )
        transforms.append(AsNumpyArray(
            field="feat_dynamic_real",
            expected_ndim=2,
            dtype=np.float32,
        ))
        transforms.append(AddObservedValuesIndicator(
            target_field="feat_dynamic_real",
            output_field="observed_feat_dynamic_real",
            imputation_method=imputation_2d,
        ))
        if self.past_dynamic_dims:
            transforms.append(AsNumpyArray(
                field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                expected_ndim=2,
                dtype=np.float32,
            ))
            transforms.append(AddObservedValuesIndicator(
                target_field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                output_field="past_observed_feat_dynamic_real",
                imputation_method=imputation_2d,
            ))
        if self.dynamic_cardinalities:
            transforms.append(
                AsNumpyArray(
                    field=FieldName.FEAT_DYNAMIC_CAT,
                    expected_ndim=2,
                    dtype=np.int64,
                )
            )
            transforms.append(AddObservedValuesIndicator(
                target_field=FieldName.FEAT_DYNAMIC_CAT,
                output_field="observed_feat_dynamic_cat",
                imputation_method=imputation_2d,
            ))
        if self.past_dynamic_cardinalities:
            transforms.append(
                AsNumpyArray(
                    field=FieldName.PAST_FEAT_DYNAMIC_CAT,
                    expected_ndim=2,
                    dtype=np.int64,
                )
            )
            transforms.append(AddObservedValuesIndicator(
                target_field=FieldName.PAST_FEAT_DYNAMIC_CAT,
                output_field="past_observed_feat_dynamic_cat",
                imputation_method=imputation_2d,
            ))


        return Chain(transforms)

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        past_ts_fields, ts_fields = self.get_ts_fields()
        return TFTInstanceSplitter(
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )

    def input_names(self):
        input_names = list(TRAINING_INPUT_NAMES)

        if not self.dynamic_cardinalities:
            input_names.remove("feat_dynamic_cat")
            input_names.remove("observed_feat_dynamic_cat")

        if not self.past_dynamic_cardinalities:
            input_names.remove("past_feat_dynamic_cat")
            input_names.remove("past_observed_feat_dynamic_cat")

        if not self.past_dynamic_dims:
            input_names.remove("past_feat_dynamic_real")
            input_names.remove("past_observed_feat_dynamic_real")

        if not self.use_satellite:
            input_names.remove("satellite_data")
        if not self.use_text:
            input_names.remove("text_data")
            # input_names.remove("satellite_coords")
        return input_names

    def create_training_data_loader(
            self,
            data,
            module: TSAdapterLightningModule,
            shuffle_buffer_length: Optional[int] = None,
            transformations: Transformation = Identity(),
            **kwargs,
    ) -> Iterable:
        if self.indexed_sample:
            dataset = data.dataset
            offset = data.offset
            base_transform = data.transform
            from data.loader import IndexedTrainDataLoader, MixedRandomDataset
            past_ts_fields, ts_fields = self.get_ts_fields()
            data = MixedRandomDataset(
                dataset=dataset,
                instance_sampler=self.train_sampler,
                past_length=self.context_length,
                future_length=self.prediction_length,
                time_series_fields=ts_fields,
                past_time_series_fields=past_ts_fields,
                offset=offset,
                is_train=True,
                field_names=self.input_names(),
                transform=compose(TransformationWrapper(transformations), base_transform)
            )
            return IndexedTrainDataLoader(data,
                                          batch_size=self.batch_size,
                                          num_batches_per_epoch=self.num_batches_per_epoch,
                                          num_workers=self.num_workers,
                                          shuffle=True)
        else:
            data = Cyclic(data)
            transformations += self._create_instance_splitter("training")
            return TrainDataLoader(data,
                                   batch_size=self.batch_size,
                                   transform=transformations,
                                   shuffle_buffer_length=shuffle_buffer_length,
                                   field_names=self.input_names(),
                                   output_type=torch.tensor,
                                   num_batches_per_epoch=self.num_batches_per_epoch,
                                   num_workers=self.num_workers if self.num_workers > 0 else None,
                                   )

    def create_validation_data_loader(
            self,
            data: Dataset,
            module: TSAdapterLightningModule,
            transformations: Transformation = Identity(),
            **kwargs,
    ) -> Iterable:

        if self.indexed_sample:
            dataset = data.dataset
            offset = data.offset
            base_transform = data.transform
            from data.loader import IndexedTrainDataLoader, MixedRandomDataset
            past_ts_fields, ts_fields = self.get_ts_fields()
            data = MixedRandomDataset(
                dataset=dataset,
                instance_sampler=self.validation_sampler,
                past_length=self.context_length,
                future_length=self.prediction_length,
                time_series_fields=ts_fields,
                past_time_series_fields=past_ts_fields,
                offset=offset,
                is_train=True,
                field_names=self.input_names(),
                transform=compose(TransformationWrapper(transformations), base_transform),
                subsample=1000
            )
            return IndexedTrainDataLoader(data,
                                          batch_size=self.batch_size,
                                          # num_batches_per_epoch=self.num_batches_per_epoch,
                                          num_workers=self.num_workers,
                                          shuffle=False)
        else:
            data = Cyclic(data)
            transformations += self._create_instance_splitter("validation")
            return TrainDataLoader(data,
                                   batch_size=self.batch_size,
                                   transform=transformations,
                                   field_names=self.input_names(),
                                   output_type=torch.tensor,
                                   # num_batches_per_epoch=self.num_batches_per_epoch,
                                   num_workers=self.num_workers if self.num_workers > 0 else None,
                                   )

    def get_ts_fields(self):
        ts_fields = [FieldName.FEAT_DYNAMIC_REAL, "observed_feat_dynamic_real"]
        if self.dynamic_cardinalities:
            ts_fields.append(FieldName.FEAT_DYNAMIC_CAT)
            ts_fields.append("observed_feat_dynamic_cat")
        past_ts_fields = []
        if self.past_dynamic_cardinalities:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_CAT)
            past_ts_fields.append("past_observed_feat_dynamic_cat")
        if self.past_dynamic_dims:
            past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
            past_ts_fields.append("past_observed_feat_dynamic_real")
        if self.use_satellite:
            past_ts_fields.append("satellite_data")
        if self.use_text:
            past_ts_fields.append("text_data")
        return past_ts_fields, ts_fields

    def create_lightning_module(
            self,
    ) -> TSAdapterLightningModule:
        if self.module_name == "worldts_image":
            from .worldts_image.lightning_module import (
                WorldTSImageLightningModule,
            )
            return WorldTSImageLightningModule(
                model_kwargs={
                    "context_length": self.context_length,
                    "prediction_length": self.prediction_length,
                    "d_feat_dynamic_real": [1] * max(len(self.time_features), 1)
                    + self.dynamic_dims,
                    "d_multi_modal": self.args.d_multi_modal,
                    "homogenizer_type": self.args.homogenizer_type,
                    **self.model_kwargs,
                },
                stage1_epochs=self.args.stage1_epochs,
                stage2_epochs=self.args.stage2_epochs,
                stage1_lr=self.args.stage1_lr,
                stage2_lr=self.args.stage2_lr,
                weight_decay=self.weight_decay,
            )
        return TSAdapterLightningModule(
            lr=self.lr,
            patience=self.patience,
            weight_decay=self.weight_decay,
            model_kwargs={
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "num_heads": self.num_heads,
                "distr_output": self.distr_output,
                "model_wrapper": self.model_wrapper,
                "d_past_feat_dynamic_real": self.past_dynamic_dims,
                "c_past_feat_dynamic_cat": self.past_dynamic_cardinalities,
                "d_feat_dynamic_real": [1] * max(len(self.time_features), 1)
                                       + self.dynamic_dims,
                "c_feat_dynamic_cat": self.dynamic_cardinalities,
                "d_feat_static_real": self.static_dims or [1],
                "c_feat_static_cat": self.static_cardinalities or [1],
                "c_item_index": 1 if self.item_map is None else len(self.item_map),
                "dropout_rate": self.dropout_rate,
                "module_name": self.module_name,
                "wo_future_token": self.args.wo_future_token,
                "full_ft": self.args.full_ft,
                "future_pos": self.args.future_pos,
                "past_pos": self.args.past_pos,
                "device_map": "auto",
                "use_satellite": self.use_satellite,
                "use_text": self.use_text,
                "d_multi_modal": self.args.d_multi_modal,
                **self.model_kwargs
            },
        )

    def create_predictor(
            self,
            transformation: Transformation,
            module: TSAdapterLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
            forecast_generator=self.model_wrapper.forecast_generator(),
        )

    def get_predictor(self, init_weight):
        training_network = self.create_lightning_module()
        ckpt_dict = torch.load(init_weight)
        training_network.load_state_dict(ckpt_dict["state_dict"])
        prediction_splitter = self._create_instance_splitter("test")
        transformation = self.create_transformation()
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=training_network,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
            forecast_generator=self.model_wrapper.forecast_generator(),
        )

    @staticmethod
    def _restore_lightning_checkpoint(
            training_network: pl.LightningModule,
            checkpoint: pl.callbacks.ModelCheckpoint,
            remove_after_restore: bool = True,
    ) -> None:
        best_path = checkpoint.best_model_path
        if not best_path:
            raise RuntimeError(
                f"No best checkpoint was produced for {checkpoint.monitor}"
            )
        logger.info(f"Loading best model from {best_path}")
        checkpoint_dict = torch.load(
            best_path,
            map_location="cpu",
            weights_only=False,
        )
        training_network.load_state_dict(
            checkpoint_dict["state_dict"],
            strict=True,
        )
        if remove_after_restore:
            os.remove(best_path)
            logger.info(f"Removed checkpoint at {best_path}")
        else:
            logger.info(f"Retained best checkpoint at {best_path}")

    def _train_worldts_two_stage(
            self,
            training_network,
            training_data_loader,
            validation_data_loader,
            ckpt_dir: Optional[str],
            custom_callbacks,
    ) -> pl.Trainer:
        if validation_data_loader is None:
            raise ValueError(
                "WorldTS two-stage training requires validation_data; "
                "enable --split_val"
            )

        stages = (
            (1, self.args.stage1_epochs, "stage1_val_loss"),
            (2, self.args.stage2_epochs, "stage2_val_loss"),
        )
        final_trainer = None
        for stage, max_epochs, monitor in stages:
            training_network.set_stage(stage)
            stage_dir = (
                os.path.join(ckpt_dir, f"stage{stage}")
                if ckpt_dir is not None
                else None
            )
            checkpoint = pl.callbacks.ModelCheckpoint(
                dirpath=stage_dir,
                filename=f"stage{stage}-best",
                monitor=monitor,
                mode="min",
                save_top_k=1,
                enable_version_counter=False,
                verbose=True,
            )
            early_stop = pl.callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.args.patience,
                mode="min",
                verbose=True,
                min_delta=0.0,
            )
            trainer_kwargs = dict(self.trainer_kwargs)
            trainer_kwargs.pop("callbacks", None)
            trainer_kwargs["max_epochs"] = int(max_epochs)
            try:
                stage_custom_callbacks = copy.deepcopy(custom_callbacks)
            except Exception:
                logger.warning(
                    "Could not copy custom callbacks for WorldTS Stage %s; "
                    "reusing the callback instances",
                    stage,
                )
                stage_custom_callbacks = list(custom_callbacks)
            trainer = pl.Trainer(
                logger=self.pl_logger,
                **{
                    "accelerator": "auto",
                    "callbacks": [
                        checkpoint,
                        early_stop,
                        lightning.pytorch.callbacks.ModelSummary(max_depth=2),
                        NaNCheckCallback(),
                    ] + stage_custom_callbacks,
                    **trainer_kwargs,
                },
            )
            logger.info(
                "Training WorldTS Stage %s for at most %s epochs; "
                "monitoring %s with patience=%s",
                stage,
                max_epochs,
                monitor,
                self.args.patience,
            )
            trainer.fit(
                model=training_network,
                train_dataloaders=training_data_loader,
                val_dataloaders=validation_data_loader,
            )
            self._restore_lightning_checkpoint(
                training_network,
                checkpoint,
                remove_after_restore=False,
            )
            final_trainer = trainer

        assert final_trainer is not None
        return final_trainer

    def train_model(
            self,
            training_data: Dataset,
            validation_data: Optional[Dataset] = None,
            from_predictor: Optional[PyTorchPredictor] = None,
            shuffle_buffer_length: Optional[int] = None,
            cache_data: bool = False,
            ckpt_path: Optional[str] = None,
            **kwargs,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        with env._let(max_idle_transforms=max(len(training_data), 100)):
            if cache_data:
                training_data = Cached(training_data)

            training_network = self.create_lightning_module()


            training_data_loader = self.create_training_data_loader(
                training_data,
                training_network,
                transformations=transformation,
                shuffle_buffer_length=shuffle_buffer_length,
            )

        validation_data_loader = None

        if validation_data is not None:
            with env._let(max_idle_transforms=max(len(validation_data), 100)):
                # transformed_validation_data: Dataset = transformation.apply(
                #     validation_data, is_train=True
                # )
                if cache_data:
                    validation_data = Cached(
                        validation_data
                    )

                validation_data_loader = self.create_validation_data_loader(
                    validation_data,
                    training_network,
                    transformations=transformation,
                )

        if from_predictor is not None:
            training_network.load_state_dict(
                from_predictor.network.state_dict()
            )

        if self.ckpt_info is not None:
            ckpt_dir = os.path.join("checkpoints", dict_to_string_id(self.ckpt_info))
            logger.info(f"Saving checkpoint at {ckpt_dir}")
        else:
            ckpt_dir = None

        custom_callbacks = list(self.trainer_kwargs.get("callbacks", []))
        if self.module_name == "worldts_image":
            if ckpt_path is not None:
                raise ValueError(
                    "ckpt_path resume is not yet supported for WorldTS's "
                    "two independent training stages"
                )
            worldts_ckpt_dir = ckpt_dir
            if worldts_ckpt_dir is None:
                output_dir = getattr(self.args, "output_dir", None)
                if output_dir:
                    worldts_ckpt_dir = os.path.join(output_dir, "checkpoints")
            trainer = self._train_worldts_two_stage(
                training_network=training_network,
                training_data_loader=training_data_loader,
                validation_data_loader=validation_data_loader,
                ckpt_dir=worldts_ckpt_dir,
                custom_callbacks=custom_callbacks,
            )
        else:
            monitor = "train_loss" if validation_data is None else "val_loss"
            checkpoint = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=monitor,
                mode="min",
                verbose=True,
            )
            early_stop = pl.callbacks.EarlyStopping(
                monitor=monitor,
                patience=20,
                mode="min",
                verbose=True,
                min_delta=1e-6,
            )
            trainer_kwargs = dict(self.trainer_kwargs)
            trainer_kwargs.pop("callbacks", None)
            trainer = pl.Trainer(
                logger=self.pl_logger,
                **{
                    "accelerator": "auto",
                    "callbacks": [
                        checkpoint,
                        early_stop,
                        lightning.pytorch.callbacks.ModelSummary(max_depth=2),
                        NaNCheckCallback(),
                    ] + custom_callbacks,
                    **trainer_kwargs,
                },
            )
            trainer.fit(
                model=training_network,
                train_dataloaders=training_data_loader,
                val_dataloaders=validation_data_loader,
                ckpt_path=ckpt_path,
            )
            self._restore_lightning_checkpoint(training_network, checkpoint)
        best_model = training_network
        predictor = self.create_predictor(transformation, best_model)
        if self.ckpt_info is not None:
            with open(os.path.join(ckpt_dir, "args.json"), "w") as f:
                json.dump(vars(self.args), f)
                logger.info(f"Saved cmd args at {ckpt_dir}")
            predictor.serialize(Path(ckpt_dir))
            logger.info(f"Saved predictor at {ckpt_dir}")
        return TrainOutput(
            transformation=transformation,
            trained_net=best_model,
            trainer=trainer,
            predictor=predictor,
        )
