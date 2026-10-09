"""Isolated two-stage WorldTS forecasting on VoT's Dataset_Custom."""

import argparse
import copy
import os
import random
import time
from typing import Dict, Iterable

import numpy as np
import torch
from torch import nn, optim
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

from data_provider.data_factory import data_provider
from models.worldts import WorldTSModel
from models.worldts.losses import LatentStateLoss
from utils.metrics import metric
from utils.timefeatures import time_features_from_frequency_str


class Exp_WorldTS_Forecast:
    """Train WorldTS in latent and decoder stages without VoT fusion."""

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = WorldTSModel(args).float().to(self.device)
        self.llm_model, self.tokenizer = self._load_frozen_gpt2()
        self.latent_state_loss = LatentStateLoss(
            latent_mse_weight=args.latent_mse_weight,
            latent_alignment_loss=args.latent_alignment_loss,
            latent_cosine_weight=args.latent_cosine_weight,
            latent_sigreg_weight=args.latent_sigreg_weight,
            latent_detach_target=args.latent_detach_target,
            sigreg_std_weight=args.sigreg_std_weight,
            sigreg_cov_weight=args.sigreg_cov_weight,
            sigreg_eps=args.sigreg_eps,
        )
        self.forecast_criterion = self._build_forecast_criterion(args.loss)

    def _acquire_device(self) -> torch.device:
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: {device}")
            return device
        print("Use CPU")
        return torch.device("cpu")

    def _load_frozen_gpt2(self):
        if not self.args.use_modal_condition:
            print("[WorldTS] text conditioning disabled")
            return None, None
        try:
            config = GPT2Config.from_pretrained(
                self.args.llm_path,
                local_files_only=True,
            )
            config.num_hidden_layers = int(self.args.llm_layers)
            model = GPT2Model.from_pretrained(
                self.args.llm_path,
                local_files_only=True,
                config=config,
            )
            tokenizer = GPT2Tokenizer.from_pretrained(
                self.args.llm_path,
                local_files_only=True,
            )
        except (OSError, EnvironmentError):
            print("Configured GPT-2 path not found; downloading from Hugging Face")
            config = GPT2Config.from_pretrained(
                "openai-community/gpt2",
                local_files_only=False,
            )
            config.num_hidden_layers = int(self.args.llm_layers)
            model = GPT2Model.from_pretrained(
                "openai-community/gpt2",
                local_files_only=False,
                config=config,
            )
            tokenizer = GPT2Tokenizer.from_pretrained(
                "openai-community/gpt2",
                local_files_only=False,
            )
        tokenizer.pad_token = tokenizer.eos_token
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval().to(self.device)
        return model, tokenizer

    @staticmethod
    def _build_forecast_criterion(loss_name: str) -> nn.Module:
        normalized = str(loss_name).upper()
        if normalized == "MSE":
            return nn.MSELoss()
        if normalized in {"MAE", "L1"}:
            return nn.L1Loss()
        if normalized in {"HUBER", "SMOOTH_L1"}:
            return nn.HuberLoss(delta=0.5)
        raise ValueError(f"Unsupported forecast loss: {loss_name}")

    def _get_data(self, flag: str):
        return data_provider(
            self.args,
            flag,
            self.llm_model,
            self.tokenizer,
        )

    def _prepare_text_embeddings(self, dataset) -> None:
        if not self.args.use_modal_condition:
            return
        dataset.get_all_embeddings()

    def _limited_batches(self, loader, training: bool) -> Iterable:
        limit = (
            int(self.args.max_train_batches)
            if training
            else int(self.args.max_eval_batches)
        )
        for batch_index, batch in enumerate(loader):
            if limit > 0 and batch_index >= limit:
                break
            yield batch

    def _prepare_batch(self, dataset, batch) -> Dict[str, torch.Tensor]:
        batch_x, batch_y, batch_x_mark, _, index = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        future_target = batch_y[:, -self.args.pred_len:, :]
        modal_tokens = None
        if self.args.use_modal_condition:
            modal_tokens = (
                dataset.get_text_embeddings(index)
                .detach()
                .float()
                .to(self.device)
            )
        return {
            "input": batch_x,
            "future_target": future_target,
            "time_hist": (
                batch_x_mark if self.args.use_time_condition else None
            ),
            "time_future": None,
            "x_future": None,
            "modal_tokens": modal_tokens,
        }

    @staticmethod
    def _model_inputs(prepared: dict) -> dict:
        return {
            key: prepared[key]
            for key in (
                "input",
                "future_target",
                "time_hist",
                "time_future",
                "x_future",
                "modal_tokens",
            )
        }

    def _stage1_batch_loss(self, prepared: dict) -> torch.Tensor:
        output = self.model(**self._model_inputs(prepared))
        return self.latent_state_loss(
            h_pred=output["h_pred"],
            h_target=output["h_target"],
        )

    def _stage2_batch_loss(self, prepared: dict) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(**self._model_inputs(prepared))
        prediction = self.model.decode(
            output["h_pred"].detach(),
            target_revin_stats=output.get("target_revin_stats"),
            history_tokens=output["h_hist"].detach(),
        )
        reconstruction = self.model.decode(
            output["h_target"].detach(),
            target_revin_stats=output.get("target_revin_stats"),
            history_tokens=output["h_hist"].detach(),
        )
        target = prepared["future_target"]
        prediction_loss = self.forecast_criterion(prediction, target)
        reconstruction_loss = self.forecast_criterion(reconstruction, target)
        return (
            prediction_loss
            + float(self.args.decoder_warmup_weight) * reconstruction_loss
        )

    def _forecast(self, prepared: dict) -> torch.Tensor:
        return self.model.forecast(
            input=prepared["input"],
            time_hist=prepared["time_hist"],
            time_future=None,
            x_future=None,
            modal_tokens=prepared["modal_tokens"],
        )

    def _validate(self, dataset, loader, stage: int) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self._limited_batches(loader, training=False):
                prepared = self._prepare_batch(dataset, batch)
                if stage == 1:
                    loss = self._stage1_batch_loss(prepared)
                else:
                    prediction = self._forecast(prepared)
                    loss = self.forecast_criterion(
                        prediction,
                        prepared["future_target"],
                    )
                losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    @staticmethod
    def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(model.state_dict())

    def _checkpoint_payload(self, stage: str, validation_loss: float) -> dict:
        config = {
            key: value
            for key, value in vars(self.args).items()
            if key != "huggingface_token"
        }
        return {
            "Model": self._clone_state_dict(self.model),
            "stage": stage,
            "seed": int(self.args.seed),
            "best_validation_loss": float(validation_loss),
            "config": config,
        }

    def _save_checkpoint(
        self,
        path: str,
        stage: str,
        validation_loss: float,
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            self._checkpoint_payload(stage, validation_loss),
            path,
        )

    def _load_checkpoint(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get("Model", checkpoint), strict=True)
        return checkpoint

    def _trainable_parameters(self):
        return [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

    def _run_stage(
        self,
        stage: int,
        train_data,
        train_loader,
        vali_data,
        vali_loader,
        stage_dir: str,
    ) -> str:
        if stage == 1:
            stage_name = "stage1_latent"
            checkpoint_name = "stage1_best.pth"
            epochs = int(self.args.stage1_epochs)
            learning_rate = float(self.args.stage1_lr)
            self.model.configure_for_stage1()
        elif stage == 2:
            stage_name = "stage2_decoder_mix"
            checkpoint_name = "stage2_best.pth"
            epochs = int(self.args.stage2_epochs)
            learning_rate = float(self.args.stage2_lr)
            self.model.configure_for_stage2()
            self.model.set_stage1_eval()
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        optimizer = optim.Adam(
            self._trainable_parameters(),
            lr=learning_rate,
        )
        print(
            f"[WorldTS] stage={stage_name} epochs={epochs} "
            f"lr={learning_rate}"
        )

        best_loss = float("inf")
        best_state = None
        patience_count = 0
        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            if stage == 2:
                self.model.set_stage1_eval()
            train_losses = []
            for batch in self._limited_batches(train_loader, training=True):
                optimizer.zero_grad()
                prepared = self._prepare_batch(train_data, batch)
                if stage == 1:
                    loss = self._stage1_batch_loss(prepared)
                else:
                    loss = self._stage2_batch_loss(prepared)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            validation_loss = self._validate(
                vali_data,
                vali_loader,
                stage,
            )
            train_loss = float(np.mean(train_losses))
            print(
                f"[WorldTS] stage={stage_name} epoch={epoch + 1}/{epochs} "
                f"train={train_loss:.6f} val={validation_loss:.6f} "
                f"time={time.time() - start_time:.2f}s"
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = self._clone_state_dict(self.model)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= int(self.args.patience):
                    print(f"[WorldTS] early stopping {stage_name}")
                    break

        self.model.load_state_dict(best_state, strict=True)
        path = os.path.join(stage_dir, checkpoint_name)
        self._save_checkpoint(path, stage_name, best_loss)
        return path

    def train(self, setting: str):
        train_data, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")
        self._prepare_text_embeddings(train_data)
        self._prepare_text_embeddings(vali_data)
        stage_dir = os.path.join(self.args.checkpoints, setting)
        os.makedirs(stage_dir, exist_ok=True)

        stage1_path = self._run_stage(
            1,
            train_data,
            train_loader,
            vali_data,
            vali_loader,
            stage_dir,
        )
        self._load_checkpoint(stage1_path)
        stage2_path = self._run_stage(
            2,
            train_data,
            train_loader,
            vali_data,
            vali_loader,
            stage_dir,
        )
        stage2_checkpoint = self._load_checkpoint(stage2_path)
        final_path = os.path.join(stage_dir, "final.pth")
        torch.save(stage2_checkpoint, final_path)
        print(f"[WorldTS] final checkpoint: {final_path}")
        return self.model

    def evaluate(
        self,
        setting: str,
        split: str = "test",
        load_checkpoint: bool = True,
    ):
        eval_data, eval_loader = self._get_data(split)
        self._prepare_text_embeddings(eval_data)
        if load_checkpoint:
            self._load_checkpoint(
                os.path.join(self.args.checkpoints, setting, "final.pth")
            )
        self.model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch in self._limited_batches(eval_loader, training=False):
                prepared = self._prepare_batch(eval_data, batch)
                prediction = self._forecast(prepared)
                predictions.append(prediction.detach().cpu().numpy())
                targets.append(
                    prepared["future_target"].detach().cpu().numpy()
                )

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        mae, mse, rmse, mape, mspe = metric(predictions, targets)
        print(
            f"[WorldTS] {split} mse={mse:.6f} mae={mae:.6f} "
            f"rmse={rmse:.6f} mape={mape:.6f} mspe={mspe:.6f}"
        )
        result_dir = os.path.join(self.args.results, setting)
        os.makedirs(result_dir, exist_ok=True)
        np.save(os.path.join(result_dir, "pred.npy"), predictions)
        np.save(os.path.join(result_dir, "true.npy"), targets)
        np.save(
            os.path.join(result_dir, "metrics.npy"),
            np.array([mae, mse, rmse, mape, mspe]),
        )
        return mse, mae

    def test(self, setting: str, load_checkpoint: bool = True):
        return self.evaluate(
            setting,
            split="test",
            load_checkpoint=load_checkpoint,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VoT Dataset_Custom + two-stage WorldTS"
    )
    parser.add_argument("--is_training", type=int, choices=(0, 1), default=1)
    parser.add_argument("--model_id", type=str, default="worldts")
    parser.add_argument(
        "--model",
        type=str,
        choices=("WorldTS",),
        default="WorldTS",
    )
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, choices=("S",), default="S")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="d")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--label_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)

    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints_worldts_tats_interface",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="./results_worldts_tats_interface",
    )
    parser.add_argument("--setting", type=str, default="")
    parser.add_argument("--des", type=str, default="worldts_tats_interface")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--loss", type=str, default="MAE")
    parser.add_argument("--stage1_epochs", type=int, default=20)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--decoder_warmup_weight", type=float, default=0.3)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument(
        "--eval_split",
        type=str,
        choices=("val", "test"),
        default="test",
        help="Evaluate this split after training; tuning runs should use val.",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        choices=("GPT2",),
        default="GPT2",
    )
    parser.add_argument("--llm_dim", type=int, default=768)
    parser.add_argument(
        "--llm_path",
        type=str,
        default="./language_model/openai-community/gpt2",
    )
    parser.add_argument("--llm_layers", type=int, default=6)
    parser.add_argument(
        "--pool_type",
        type=str,
        choices=("avg", "max", "min"),
        default="avg",
    )
    parser.add_argument("--text_embedding_batch_size", type=int, default=128)
    parser.add_argument(
        "--use_modal_condition",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--use_time_condition",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument("--modal_feature_dim", type=int, default=12)
    parser.add_argument("--modal_hidden_dim", type=int, default=256)
    parser.add_argument("--time_feature_dim", type=int, default=12)
    parser.add_argument("--time_hidden_dim", type=int, default=64)

    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--series_dim", type=int, default=1)
    parser.add_argument("--future_x_dim", type=int, choices=(0,), default=0)
    parser.add_argument("--condition_dim", type=int, choices=(0,), default=0)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--use_revin", type=int, choices=(0, 1), default=1)
    parser.add_argument("--revin_eps", type=float, default=1e-5)
    parser.add_argument(
        "--decoder_type",
        type=str,
        choices=("pointwise_mlp", "causal_patch"),
        default="pointwise_mlp",
    )
    parser.add_argument(
        "--state_encoder",
        type=str,
        choices=("pointwise", "causal_patch"),
        default="pointwise",
    )
    parser.add_argument("--state_patch_len", type=int, default=3)
    parser.add_argument(
        "--state_hidden_multiplier",
        type=int,
        default=2,
    )
    parser.add_argument("--decoder_patch_len", type=int, default=3)
    parser.add_argument(
        "--decoder_hidden_multiplier",
        type=int,
        default=2,
    )
    parser.add_argument("--predictor_patch_len", type=int, default=4)
    parser.add_argument("--predictor_patch_stride", type=int, default=2)
    parser.add_argument("--predictor_d_model", type=int, default=64)
    parser.add_argument("--predictor_n_heads", type=int, default=4)
    parser.add_argument("--predictor_e_layers", type=int, default=2)
    parser.add_argument("--predictor_d_ff", type=int, default=128)
    parser.add_argument("--predictor_dropout", type=float, default=0.0)
    parser.add_argument(
        "--predictor_activation",
        type=str,
        choices=("relu", "gelu"),
        default="gelu",
    )
    parser.add_argument("--predictor_factor", type=int, default=1)
    parser.add_argument("--modal_gate_init_logit", type=float, default=-4.0)
    parser.add_argument("--cfa_reduction_factor", type=int, default=8)
    parser.add_argument("--cfa_dropout", type=float, default=0.0)
    parser.add_argument(
        "--cfa_activation",
        type=str,
        choices=("relu", "gelu"),
        default="gelu",
    )

    parser.add_argument(
        "--latent_alignment_loss",
        type=str,
        choices=("mae", "mse", "smooth_l1", "normalized_mse"),
        default="mae",
    )
    parser.add_argument("--latent_mse_weight", type=float, default=10.0)
    parser.add_argument("--latent_cosine_weight", type=float, default=15.0)
    parser.add_argument("--latent_sigreg_weight", type=float, default=0.1)
    parser.add_argument(
        "--latent_detach_target",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument("--sigreg_std_weight", type=float, default=1.0)
    parser.add_argument("--sigreg_cov_weight", type=float, default=0.04)
    parser.add_argument("--sigreg_eps", type=float, default=1e-4)

    parser.add_argument("--use_gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gpu", type=int, default=0)

    # Dataset_Custom compatibility fields.  They are intentionally inactive.
    parser.set_defaults(
        use_closedllm=0,
        seasonal_patterns=None,
        use_future_x_condition=False,
        allow_missing_future_covariate=False,
        numeric_only=False,
        cache_text_embeddings_on_cpu=True,
    )
    return parser


def finalize_args(args):
    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()
    args.use_modal_condition = bool(args.use_modal_condition)
    args.use_time_condition = bool(args.use_time_condition)
    args.use_revin = bool(args.use_revin)
    args.latent_detach_target = bool(args.latent_detach_target)
    if args.use_time_condition:
        if args.embed == "timeF":
            args.time_input_dim = len(
                time_features_from_frequency_str(args.freq)
            )
        else:
            args.time_input_dim = 4
    else:
        args.time_input_dim = 0
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_setting(args, iteration: int) -> str:
    if args.setting:
        return args.setting
    dataset = args.data_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return (
        f"{args.model_id}_{dataset}_{args.model}_sl{args.seq_len}_"
        f"pl{args.pred_len}_ld{args.latent_dim}_seed{args.seed}_"
        f"se{args.state_encoder}_sp{args.state_patch_len}_"
        f"sh{args.state_hidden_multiplier}_"
        f"{args.des}_{iteration}"
    )


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    print(
        "[WorldTS] isolated interface: "
        f"text={args.use_modal_condition} time={args.use_time_condition} "
        f"state_encoder={args.state_encoder} "
        f"state_patch={args.state_patch_len} "
        "future_text=False future_marks=False prior=False AFF=False"
    )
    for iteration in range(args.itr):
        seed_everything(args.seed + iteration)
        setting = build_setting(args, iteration)
        experiment = Exp_WorldTS_Forecast(args)
        if args.is_training:
            print(f">>>>>>> training: {setting}")
            experiment.train(setting)
            print(f">>>>>>> evaluating {args.eval_split}: {setting}")
            experiment.evaluate(
                setting,
                split=args.eval_split,
                load_checkpoint=True,
            )
        else:
            print(f">>>>>>> evaluating {args.eval_split} only: {setting}")
            experiment.evaluate(
                setting,
                split=args.eval_split,
                load_checkpoint=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
