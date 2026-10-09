import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
from gluonts.torch import PyTorchPredictor
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.patch_tst import PatchTSTEstimator
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator
from gluonts.torch.model.tide import TiDEEstimator
from gluonts.transform import UniformSplitSampler, DummyValueImputation
from statsforecast.models import Naive, SeasonalNaive, RandomWalkWithDrift, HistoricAverage

from models.predictor.naive import StatsForecastPredictor
from .predictor.chronos import ChronosPredictor

np.bool = np.bool_
from .wrapper.imputation_wrapper import ImputationEstimatorWrapper, AllFieldImputation
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

specialized_model_map = {
    "feedforward": SimpleFeedForwardEstimator,
    "patchtst": PatchTSTEstimator,
    "tft": TemporalFusionTransformerEstimator,
    "tide": TiDEEstimator,
    "deepar": DeepAREstimator,
}

neural_forecast_models = ["nbeatsx", "timexer", "timellm"]

naive_model_map = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "randomwalk": RandomWalkWithDrift,
    "historic_average": HistoricAverage
}


def convert_dimensions_to_nbeats_lists(
        static_dims: Optional[List[int]] = None,
        dynamic_dims: Optional[List[int]] = None,
        past_dynamic_dims: Optional[List[int]] = None,
        static_cardinalities: Optional[List[int]] = None,
) -> Dict[str, List[str]]:
    result = {}

    futr_exog_list = []
    hist_exog_list = []
    stat_exog_list = []

    if dynamic_dims:
        for i in range(len(dynamic_dims)):
            futr_exog_list.append(f"feat_dyn_{i}")

    if past_dynamic_dims:
        for i in range(len(past_dynamic_dims)):
            hist_exog_list.append(f"past_feat_{i}")

    if static_dims:
        for i in range(len(static_dims)):
            stat_exog_list.append(f"stat_{i}")

    if static_cardinalities:
        for i in range(len(static_cardinalities)):
            stat_exog_list.append(f"stat_cat_{i}")

    if futr_exog_list:
        result["futr_exog_list"] = futr_exog_list
    if hist_exog_list:
        result["hist_exog_list"] = hist_exog_list
    if stat_exog_list:
        result["stat_exog_list"] = stat_exog_list

    return result


def get_predictor(args, dataset, season_length, ds_config, pl_logger=None, load_ckpt=False, init_weight=None, **kwargs):
    name = args.model_name
    batch_size = args.batch_size
    model_dir = os.getenv("MODEL_PATH", "./models/")
    if args.save_ckpt:
        infos = dict(vars(args))
        del infos["save_ckpt"]
        del infos["datasets"]
        infos["ds_name"] = ds_config
    else:
        infos = None
    while True:
        try:
            if "ts_adapter" in name:
                from .adapter.estimator import TSAdapterEstimator
                logger.info("Using TSAdapterEstimator")
                _, module_name = name.split("/")
                path_dict = {
                    "chronos_bolt_base":
                        {
                            "model_id_or_path": os.path.join(model_dir, "chronos-bolt-base"),
                            "model_type": "chronos",
                        },
                    "timesfm_2_500m":
                        {
                            "model_id_or_path": os.path.join(model_dir, "timesfm-2.0-500m/torch_model.ckpt"),
                            "model_type": "timesfm",
                        },
                }
                if module_name == "worldts_image":
                    fm_config = {
                        "model_id_or_path": "",
                        "model_type": "worldts",
                        "context_length": args.seq_len,
                    }
                    worldts_model_kwargs = {
                        "image_hidden_dim": args.image_hidden_dim,
                        "modal_hidden_dim": args.modal_hidden_dim,
                        "modal_feature_dim": args.modal_feature_dim,
                        "latent_dim": args.latent_dim,
                        "condition_dim": args.condition_dim,
                        "use_revin": bool(args.use_revin),
                        "revin_eps": args.revin_eps,
                        "state_encoder": args.state_encoder,
                        "state_patch_len": args.state_patch_len,
                        "state_hidden_multiplier": args.state_hidden_multiplier,
                        "decoder_type": args.decoder_type,
                        "decoder_patch_len": args.decoder_patch_len,
                        "decoder_hidden_multiplier": args.decoder_hidden_multiplier,
                        "predictor_patch_len": args.predictor_patch_len,
                        "predictor_patch_stride": args.predictor_patch_stride,
                        "predictor_d_model": args.predictor_d_model,
                        "predictor_n_heads": args.predictor_n_heads,
                        "predictor_e_layers": args.predictor_e_layers,
                        "predictor_d_ff": args.predictor_d_ff,
                        "predictor_dropout": args.predictor_dropout,
                        "predictor_activation": args.predictor_activation,
                        "predictor_factor": args.predictor_factor,
                        "modal_gate_init_logit": args.modal_gate_init_logit,
                        "cfa_reduction_factor": args.cfa_reduction_factor,
                        "cfa_dropout": args.cfa_dropout,
                        "cfa_activation": args.cfa_activation,
                        "latent_alignment_loss": args.latent_alignment_loss,
                        "latent_mse_weight": args.latent_mse_weight,
                        "latent_cosine_weight": args.latent_cosine_weight,
                        "latent_sigreg_weight": args.latent_sigreg_weight,
                        "latent_detach_target": bool(args.latent_detach_target),
                        "sigreg_std_weight": args.sigreg_std_weight,
                        "sigreg_cov_weight": args.sigreg_cov_weight,
                        "sigreg_eps": args.sigreg_eps,
                        "decoder_warmup_weight": args.decoder_warmup_weight,
                        "loss": args.loss,
                    }
                else:
                    fm_config = path_dict[args.base_model]
                    worldts_model_kwargs = {}
                if args.sampler == "random":
                    from data.sampler import RandomSampler
                    logger.info("#### Using RandomSampler ####")
                    train_sampler = RandomSampler(
                        min_future=dataset.prediction_length
                    )
                else:
                    logger.info("#### Using UniformSplitSampler ####")
                    train_sampler = UniformSplitSampler(
                        p=1.0, min_future=dataset.prediction_length,
                    )
                from data.sampler import UniformWithStartSampler
                val_sampler = UniformWithStartSampler(
                    start=- dataset.split_point,
                    min_future=dataset.prediction_length,
                )
                estimator = TSAdapterEstimator(
                    freq=dataset.freq,
                    lr=args.lr,
                    prediction_length=dataset.prediction_length,
                    num_batches_per_epoch=args.num_batches_per_epoch,
                    batch_size=batch_size,
                    item_map=dataset.item_map,
                    weight_decay=args.weight_decay,
                    model_kwargs={
                        "with_gate": args.with_gate,
                        "future_with_gate": args.future_with_gate,
                        "with_future": args.with_future,
                        "with_dc": args.with_dc,
                        "future_skip": args.future_skip,
                        "future_shortcut": args.future_shortcut,
                        "fuse_type": args.fuse_type,
                        "with_static": args.with_static,
                        "with_past": args.with_past,
                        "add_item_id": args.add_item_id,
                        "shortcut_after_fuse": args.shortcut_after_fuse,
                        "encoder_path": os.path.join(model_dir, args.encoder_path),
                        "dropout_rate": args.dropout,
                        "homogenizer_type": args.homogenizer_type,
                        **worldts_model_kwargs,
                    },
                    pl_logger=pl_logger,
                    num_workers=args.num_workers,
                    train_sampler=train_sampler,
                    validation_sampler=val_sampler,
                    trainer_kwargs={"max_epochs": args.max_epochs, "accelerator": "gpu",
                                    "gradient_clip_val": args.gradient_clip},
                    indexed_sample=args.indexed_sample,
                    imputation=args.imputation,
                    module_name=module_name,
                    args=args,
                    ckpt_info=infos,
                    use_satellite=getattr(args, 'use_satellite', False),
                    use_text=getattr(args, 'use_text', False),
                    **dataset.dim_params,
                    **fm_config
                )
                if load_ckpt:
                    assert init_weight is not None, "Please provide the path to the checkpoint"
                    predictor = PyTorchPredictor.deserialize(Path(init_weight))
                else:
                    if args.split_val:
                        predictor = estimator.train(training_data=dataset.training_dataset,
                                                    validation_data=dataset.validation_dataset,
                                                    shuffle_buffer_length=100)
                    else:
                        predictor = estimator.train(dataset.validation_dataset,
                                                    shuffle_buffer_length=100)
            elif name in naive_model_map.keys():
                logger.info(f"Using {name} predictor")

                class NaivePredictor(StatsForecastPredictor):
                    """
                    A predictor wrapping the ``Naive`` model from `statsforecast`_.

                    See :class:`StatsForecastPredictor` for the list of arguments.

                    .. _statsforecast: https://github.com/Nixtla/statsforecast
                    """

                    ModelType = naive_model_map[args.model_name]

                predictor = NaivePredictor(
                    dataset.prediction_length,
                    season_length=season_length,
                    freq=dataset.freq,
                    quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    batch_size=512,
                )
            elif "timesfm" in name:
                from .predictor.timesfm import TimesFmPredictor
                logger.info(f"Using {name} predictor")
                name_path_map = {
                    "timesfm_2_500m": os.path.join(model_dir, "timesfm-2.0-500m/torch_model.ckpt"),
                }
                predictor = TimesFmPredictor(
                    model_path=name_path_map[name],
                    prediction_length=dataset.prediction_length,
                    ds_freq=dataset.freq,
                )

            elif "chronos" in name:
                logger.info(f"Using {name} predictor")
                name_path_map = {
                    "chronos_bolt": os.path.join(model_dir, "chronos-bolt-base"),
                    "chronos_base": os.path.join(model_dir, "chronos-t5-base"),
                }
                predictor = ChronosPredictor(
                    model_path=name_path_map[name],
                    num_samples=args.num_samples,
                    prediction_length=dataset.prediction_length,
                    device_map="cuda:0",
                )
            elif "moirai" in name:
                logger.info(f"Using {name} predictor")

                name_path_map = {
                    "moirai_small": os.path.join(model_dir, "moirai-1.1-R-small"),
                    "moirai_large": os.path.join(model_dir, "moirai-1.1-R-large"),
                }

                from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

                model = MoiraiForecast(
                    module=MoiraiModule.from_pretrained(name_path_map[args.model_name]),
                    prediction_length=dataset.prediction_length,
                    context_length=4000,
                    patch_size=32,
                    num_samples=args.num_samples,
                    target_dim=1,
                    feat_dynamic_real_dim=sum(dataset.dim_params.get('dynamic_dims', [0])),
                    past_feat_dynamic_real_dim=sum(dataset.dim_params.get('past_dynamic_dims', [0])),
                )
                # set the Moirai hyperparameter according to each dataset, then create the predictor
                model.hparams.prediction_length = dataset.prediction_length
                model.hparams.target_dim = dataset.target_dim
                # model.hparams.past_feat_dynamic_real_dim = dataset.past_feat_dynamic_real_dim

                predictor = model.create_predictor(batch_size=batch_size)
            elif "ttm" in name:
                from .predictor.ttm.gluonts_data_wrapper import TTM_MAX_FORECAST_HORIZON
                from .predictor.ttm.ttm_gluonts_predictor import TTMGluonTSPredictor
                from .predictor.ttm.utils import get_args
                # Get suitable context length for TTM for this dataset
                all_lengths = []
                for x in dataset.test_data:
                    if len(x[0]["target"].shape) == 1:
                        all_lengths.append(len(x[0]["target"]))
                        num_channels = 1
                    else:
                        all_lengths.append(x[0]["target"].shape[1])
                        num_channels = x[0]["target"].shape[0]

                min_context_length = min(all_lengths)
                print(
                    "Minimum context length among all time series in this dataset =",
                    min_context_length,
                )

                # Set channel indices
                num_prediction_channels = num_channels
                prediction_channel_indices = list(range(num_channels))

                # Check existence of "past_feat_dynamic_real"
                past_feat_dynamic_real_exist = False
                if "past_feat_dynamic_real" in x[0].keys():
                    num_exogs = x[0]["past_feat_dynamic_real"].shape[0]
                    print(f"Data has `past_feat_dynamic_real` features of size {num_exogs}.")
                    num_channels += num_exogs
                    past_feat_dynamic_real_exist = True

                if dataset.prediction_length > TTM_MAX_FORECAST_HORIZON:
                    # predict all channels, needed for recursive forecast
                    prediction_channel_indices = list(range(num_channels))

                print("prediction_channel_indices =", prediction_channel_indices)
                term = dataset.term
                ds_name = dataset.name
                # For very short series, force short context window creatiio for finetuning
                cmd_args = args
                args = get_args()
                if term == "short":
                    force_short_context = args.force_short_context
                else:
                    force_short_context = False

                # Instantiate the TTM GluonTS Predictor with the minimum context length in the dataset
                # The predictor will automatically choose the suitable context and forecast length
                # of the TTM model.

                predictor = TTMGluonTSPredictor(
                    context_length=min_context_length,
                    prediction_length=dataset.prediction_length,
                    # model_path=os.path.join(model_dir, "ibm-granite/granite-timeseries-ttm-r2"),
                    model_path="ibm-granite/granite-timeseries-ttm-r2",
                    test_data_label=dataset.test_data.label,
                    random_seed=cmd_args.seed,
                    scale=False,
                    term=term,
                    ds_name=ds_name,
                    # out_dir=OUT_DIR,
                    # scale=True,
                    upper_bound_fewshot_samples=args.upper_bound_fewshot_samples,
                    force_short_context=force_short_context,
                    min_context_mult=args.min_context_mult,
                    past_feat_dynamic_real_exist=past_feat_dynamic_real_exist,
                    num_prediction_channels=num_prediction_channels,
                    freq=dataset.freq,
                    use_valid_from_train=args.use_valid_from_train,
                    insample_forecast=args.insample_forecast,
                    insample_use_train=args.insample_use_train,
                    # TTM kwargs
                    head_dropout=args.head_dropout,
                    decoder_mode=args.decoder_mode,
                    num_input_channels=num_channels,
                    huber_delta=args.huber_delta,
                    quantile=args.quantile,
                    loss=args.loss,
                    prediction_channel_indices=prediction_channel_indices,
                )

                print(f"Number of channels in the dataset {ds_name} =", num_channels)
                if args.batch_size is None:
                    batch_size = None
                    optimize_batch_size = True
                else:
                    batch_size = args.batch_size
                    optimize_batch_size = False
                print("Batch size is set to", batch_size)

                finetune_train_num_samples = 0
                finetune_valid_num_samples = 0
                try:
                    # finetune the model on the train split
                    predictor.train(
                        train_dataset=dataset.training_dataset,
                        valid_dataset=dataset.validation_dataset,
                        batch_size=batch_size,
                        optimize_batch_size=optimize_batch_size,
                        freeze_backbone=args.freeze_backbone,
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs,
                        fewshot_fraction=args.fewshot_fraction,
                        fewshot_location=args.fewshot_location,
                        automate_fewshot_fraction=args.automate_fewshot_fraction,
                        automate_fewshot_fraction_threshold=args.automate_fewshot_fraction_threshold,
                    )
                    finetune_success = True
                    finetune_train_num_samples = predictor.train_num_samples
                    finetune_valid_num_samples = predictor.valid_num_samples
                except Exception as e:
                    print("Error in finetune workflow. Error =", e)
                    print("Fallback to zero-shot performance.")
                    finetune_success = False
            elif name in specialized_model_map.keys():
                default_trainer_kwargs = {"max_epochs": args.max_epochs,
                                          "accelerator": "auto"
                                          }
                from gluonts.mx.trainer import Trainer
                name_params_map = {
                    "feedforward": {
                        "hidden_dimensions": [20, 20],
                        "trainer_kwargs": default_trainer_kwargs,
                    },
                    "patchtst": {
                        "patch_len": 16,
                        "trainer_kwargs": default_trainer_kwargs,
                        "context_length": min(336, 10 * dataset.prediction_length),
                    },
                    "tft": {
                        "trainer_kwargs": default_trainer_kwargs,
                        "freq": dataset.freq,
                        **dataset.dim_params
                    },
                    "tide": {
                        "trainer_kwargs": default_trainer_kwargs,
                        "freq": dataset.freq,
                        "num_feat_dynamic_real": sum(dataset.dim_params.get('dynamic_dims', [0])),
                        "num_feat_static_real": sum(dataset.dim_params.get('static_dims', [0])),
                        "num_feat_static_cat": len(dataset.dim_params.get('static_cardinalities', [])),
                        "cardinality": dataset.dim_params.get('static_cardinalities', [])
                    },
                    "deepar": {
                        "trainer_kwargs": default_trainer_kwargs,
                        "freq": dataset.freq,
                        "num_feat_dynamic_real": sum(dataset.dim_params.get('dynamic_dims', [0])),
                        "num_feat_static_real": sum(dataset.dim_params.get('static_dims', [0])),
                        "num_feat_static_cat": len(dataset.dim_params.get('static_cardinalities', [])),
                        "cardinality": dataset.dim_params.get('static_cardinalities', [])
                    },
                    "nbeats": {
                        "freq": dataset.freq,
                        "loss_function": "MASE",
                        "trainer": Trainer(epochs=args.max_epochs, num_batches_per_epoch=args.num_batches_per_epoch,
                                           )
                    }
                }
                cls = specialized_model_map[args.model_name]
                estimator = cls(
                    prediction_length=dataset.prediction_length,
                    batch_size=batch_size,
                    **name_params_map[args.model_name],
                )
                estimator_trans_fn = estimator.create_transformation
                estimator.create_transformation = lambda **kwargs: estimator_trans_fn(**kwargs) + AllFieldImputation(
                    DummyValueImputation())
                predictor = estimator.train(training_data=dataset.validation_dataset,
                                            shuffle_buffer_length=10)

            else:
                raise ValueError(f"Unknown model name: {name}")
            break
        except torch.cuda.OutOfMemoryError as e:
            print(
                f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
            )
            batch_size //= 2
            if batch_size < 1:
                raise e
    return predictor
