import argparse
import json
import logging
import os
from typing import Iterable

import numpy as np
import torch
from gluonts.dataset import DataBatch, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import (
    batcher,
    rows_to_columns,
)
from gluonts.pydantic import BaseModel
from gluonts.transform import (
    Transformation,
)
from torch import nn
from torch.utils.data import IterableDataset

from data import Dataset
from data.multi_modal.timemmd import TextMultiModalDataset
from data.multi_modal.ts3m import ImageMultimodalDataset

logger = logging.getLogger(__name__)

DataLoader = Iterable[DataBatch]

multi_modal_datasets = ["mmsp"]
pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
dataset_properties_map = json.load(open("data/dataset_properties.json"))

datasets_dict = {
    "epf_sub": "epf/EPF_BE epf/EPF_DE epf/EPF_FR epf/EPF_NP epf/EPF_PJM".split(),
    "cov_all": "bull cockatoo covid19_energy gfc12_load gfc14_load gfc17_load hog pdb spain epf/EPF_test m5/M5_test retail/test".split(),
    "mmsp": "mmsp".split(),
    "m5": "m5/M5_test".split(),
    "time-mmd-sub": "time-mmd/Public_Health time-mmd/Traffic time-mmd/Climate time-mmd/Energy time-mmd/Security time-mmd/Environment time-mmd/SocialGood".split(),
    'time-mmd/Public_Health': 'time-mmd/Public_Health'.split(),
    'time-mmd/Traffic': 'time-mmd/Traffic'.split(),
    'time-mmd/Climate': 'time-mmd/Climate'.split(),
    'time-mmd/Energy': 'time-mmd/Energy'.split(),
    'time-mmd/Security': 'time-mmd/Security'.split(),
    'time-mmd/Environment': 'time-mmd/Environment'.split(),
    'time-mmd/SocialGood': 'time-mmd/SocialGood'.split()
    # "time-mmd": "time-mmd".split(),
}


def evaluate_model_by_item(predictor, test_data, metrics, batch_size=32, num_workers=0):
    torch.cuda.empty_cache()

    item_ids = {}
    item_data = {}

    for entry in test_data:
        item_id = entry.get(FieldName.ITEM_ID, None)
        if item_id is None:
            continue

        if item_id not in item_data:
            item_data[item_id] = []

        item_data[item_id].append(entry)

    from gluonts.dataset.loader import DataLoader

    all_results = {}

    for item_id, site_data in item_data.items():
        site_loader = DataLoader(site_data, batch_size=batch_size, num_workers=num_workers)

        site_predictions = []
        for batch in site_loader:
            batch_predictions = predictor.predict(batch)
            site_predictions.extend(list(batch_predictions))

        site_results = {}
        for metric in metrics:
            site_results[metric.__class__.__name__] = metric(site_predictions, site_data)

        all_results[item_id] = site_results

    all_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    all_predictions = []
    for batch in all_data_loader:
        batch_predictions = predictor.predict(batch)
        all_predictions.extend(list(batch_predictions))

    overall_results = {}
    for metric in metrics:
        overall_results[metric.__class__.__name__] = metric(all_predictions, test_data)

    all_results["overall"] = overall_results

    return all_results


class Batch(Transformation, BaseModel):
    batch_size: int

    def __call__(self, data, is_train):
        yield from batcher(data, self.batch_size)


class Stack(Transformation, BaseModel):
    def __call__(self, data, is_train):
        for batch in data:
            yield rows_to_columns(batch, np.array)


def count_trainable_parameters(model: nn.Module) -> str:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for unit in ['', 'K', 'M', 'B', 'T']:
        if num_params < 1000:
            return f"{num_params:.2f}{unit}"
        num_params /= 1000
    return f"{num_params:.2f}T"


class StreamingDataset(IterableDataset):
    def __init__(self, iterable):
        super().__init__()
        self.stream_generator = list(iterable)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process data loading
            yield from self.stream_generator()
        else:  # Multi-process data loading
            # Get worker details
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Create a generator that skips data based on worker ID
            def worker_generator():
                for i, item in enumerate(self.stream_generator()):
                    if i % num_workers == worker_id:
                        yield item

            yield from worker_generator()


def get_args_and_settings():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs for training")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples for prediction")
    parser.add_argument("--seed", type=int, default=42, help="Number of samples for prediction")
    parser.add_argument("--d_multi_modal", type=int, default=4, help="Number of samples for prediction")
    parser.add_argument("--d_down_sample", type=int, default=-1, help="Number of samples for prediction")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of samples for prediction")
    parser.add_argument("--num_scale", type=int, default=5, help="Number of samples for prediction")
    parser.add_argument("--sampler", type=str, default="random", help="Number of samples for prediction")
    parser.add_argument("--num_batches_per_epoch", "-n", type=int, default=500,
                        help="Number of samples for prediction")
    parser.add_argument("--model_name", type=str, default="ts_adaptor/bootstrap", help="Model name to run")
    parser.add_argument("--base_model", type=str, default="chronos_bolt_base", help="Model name to run")
    parser.add_argument("--past_pos", type=str, default="pre", help="Model name to run", choices=["pre", "post"])
    parser.add_argument("--future_pos", type=str, default="post", help="Model name to run", choices=["pre", "post"])
    parser.add_argument("--fuse_type", type=str, default="tft", help="Model name to run")
    parser.add_argument("--imputation", type=str, default="last", help="Model name to run")
    parser.add_argument("--encoder_path", type=str, default="avsolatorio/GIST-small-Embedding-v0",
                        help="Model path to the encoder")
    parser.add_argument("--pretrained", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--lr", type=float, default=5e-5, help="Model name to run")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Model name to run")
    parser.add_argument("--dropout", type=float, default=0.1, help="Model name to run")
    parser.add_argument("--gradient_clip", type=float, default=None, help="Model name to run")
    parser.add_argument("--with_gate", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--future_with_gate", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--with_future", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--indexed_sample", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--wo_time", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--aggregate_by_item", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--wo_future_token", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--only_quantile_loss", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--sample_output", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--normalized_loss", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--add_item_id", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--with_dc", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--full_ft", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--wo_nwp", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--split_val", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--no_past", dest="with_past", action="store_false", default=True,
                        help="Whether to use the gate mechanism")
    parser.add_argument("--no_cov", dest="with_cov", action="store_false", default=True)
    parser.add_argument("--no_static", dest="with_static", action="store_false", default=True,
                        help="Whether to use the gate mechanism")
    parser.add_argument("--future_skip", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--future_shortcut", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--shortcut_after_fuse", action="store_true", help="Whether to use the gate mechanism")
    parser.add_argument("--valid_on_test", action="store_true")
    # parser.add_argument("--mix_type", type=str, default="mlp", help="Type of mixing layer")
    parser.add_argument("--datasets", type=str, nargs="+", default=["gift", "covariate"],
                        choices=datasets_dict.keys(),
                        help="Datasets to train/evaluate the model on")
    parser.add_argument("--save_ckpt", action="store_true", help="Whether to save the model")
    parser.add_argument("--homogenizer_type", type=str, default="linear", help="Type of homogenizer")
    # WorldTS parameters use the same semantic names across all benchmarks.
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--image_hidden_dim", type=int, default=768)
    parser.add_argument("--modal_hidden_dim", type=int, default=256)
    parser.add_argument("--modal_feature_dim", type=int, default=12)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--condition_dim", type=int, default=64)
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_revin", type=int, choices=(0, 1), default=1)
    parser.add_argument("--revin_eps", type=float, default=1e-5)
    parser.add_argument(
        "--state_encoder",
        choices=("pointwise", "causal_patch"),
        default="causal_patch",
    )
    parser.add_argument("--state_patch_len", type=int, default=3)
    parser.add_argument("--state_hidden_multiplier", type=int, default=2)
    parser.add_argument(
        "--decoder_type",
        choices=("pointwise_mlp", "causal_patch"),
        default="pointwise_mlp",
    )
    parser.add_argument("--decoder_patch_len", type=int, default=3)
    parser.add_argument("--decoder_hidden_multiplier", type=int, default=2)
    parser.add_argument("--predictor_patch_len", type=int, default=4)
    parser.add_argument("--predictor_patch_stride", type=int, default=2)
    parser.add_argument("--predictor_d_model", type=int, default=64)
    parser.add_argument("--predictor_n_heads", type=int, default=4)
    parser.add_argument("--predictor_e_layers", type=int, default=2)
    parser.add_argument("--predictor_d_ff", type=int, default=128)
    parser.add_argument("--predictor_dropout", type=float, default=0.0)
    parser.add_argument(
        "--predictor_activation",
        choices=("relu", "gelu"),
        default="gelu",
    )
    parser.add_argument("--predictor_factor", type=int, default=1)
    parser.add_argument("--modal_gate_init_logit", type=float, default=-4.0)
    parser.add_argument("--cfa_reduction_factor", type=int, default=8)
    parser.add_argument("--cfa_dropout", type=float, default=0.0)
    parser.add_argument(
        "--cfa_activation",
        choices=("relu", "gelu"),
        default="gelu",
    )
    parser.add_argument(
        "--latent_alignment_loss",
        choices=("mae", "mse", "smooth_l1", "normalized_mse"),
        default="mae",
    )
    parser.add_argument("--latent_mse_weight", type=float, default=10.0)
    parser.add_argument("--latent_cosine_weight", type=float, default=15.0)
    parser.add_argument("--latent_sigreg_weight", type=float, default=0.1)
    parser.add_argument("--latent_detach_target", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sigreg_std_weight", type=float, default=1.0)
    parser.add_argument("--sigreg_cov_weight", type=float, default=0.04)
    parser.add_argument("--sigreg_eps", type=float, default=1e-4)
    parser.add_argument("--decoder_warmup_weight", type=float, default=0.3)
    parser.add_argument(
        "--loss",
        choices=("mse", "mae", "huber"),
        default="mse",
    )
    parser.add_argument("--init_weight", type=str, default=None, help="Whether to save the model")
    parser.add_argument("--output_dir", type=str, default=None, help="Whether to save the model")
    parser.add_argument("--des", type=str, default="fix attention multi modal experiments",
                        help="Description of the experiment")
    args = parser.parse_args()
    if args.patience < 0:
        parser.error("--patience must be non-negative")
    # Set the random seed for all the libraries
    set_seeds(args.seed)
    torch.set_num_threads(16)
    # torch.set_float32_matmul_precision('high')
    print(vars(args))
    datasets = []
    for ds in args.datasets:
        datasets.extend(datasets_dict[ds])
    return args, datasets


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(args, ds_name, term):
    if "/" in ds_name:
        ds_key = ds_name.split("/")[0]
        ds_freq = ds_name.split("/")[1]
        ds_key = ds_key
        ds_key = pretty_names.get(ds_key, ds_key)
    else:
        ds_key = ds_name
        ds_key = pretty_names.get(ds_key, ds_key)
        ds_freq = dataset_properties_map[ds_key].get("frequency", "W")
    ds_config = f"{ds_key}/{ds_freq}/{term}"
    logger.info(f"Running model on dataset: {ds_config}")

    args.ds_name = ds_name
    args.term = term
    is_text_dataset = "time-mmd" in ds_name
    is_image_dataset = "mmsp" in ds_name
    # Initialize the dataset
    if is_text_dataset:
        dataset = TextMultiModalDataset(
            name=ds_name,
            term=term,
            indexed_sample=args.indexed_sample,
            tokenizer_name=os.path.join(os.getenv("MODEL_PATH", "./models/"), args.encoder_path)
        )
        args.use_text = True
    elif is_image_dataset:
        dataset = ImageMultimodalDataset(
            name=ds_name,
            term=term,
            indexed_sample=args.indexed_sample,
            satellite_dir=args.satellite_dir if hasattr(args, 'satellite_dir') else 'satellite',
            num_sites=args.num_sites if hasattr(args, 'num_sites') else 10,
            num_ignored_sites=args.num_ignored_sites if hasattr(args, 'num_ignored_sites') else 0,
            norm_nwp=args.norm_nwp if hasattr(args, 'norm_nwp') else True,
            norm_stl=args.norm_stl if hasattr(args, 'norm_stl') else True,
            pretrain=args.pretrained,
            encoder_path=os.path.join(os.getenv("MODEL_PATH", "./models/"), args.encoder_path),
            wo_nwp=args.wo_nwp
        )
        if args.wo_nwp:
            ds_config = f"{ds_key}_no_nwp/{ds_freq}/{term}"
        args.use_satellite = True
    else:
        dataset = Dataset(name=ds_name, term=term, indexed_sample=args.indexed_sample)
        assert dataset.target_dim == 1

        args.use_satellite = False
        # assert dataset.target_dim == 1
    return dataset, ds_config, ds_key
