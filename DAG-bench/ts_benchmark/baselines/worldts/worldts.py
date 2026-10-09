# -*- coding: utf-8 -*-
import copy
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from ts_benchmark.baselines.worldts.losses.latent_state_loss import LatentStateLoss
from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split


MODEL_HYPER_PARAMS = {
    'use_amp': 0,
    'batch_size': 32,
    'lradj': 'type3',
    'lr': 1e-4,
    'stage1_lr': 0.0,
    'stage2_lr': 0.0,
    'num_epochs': 20,
    'num_workers': 0,
    'patience': 5,
    'seq_len': 96,
    'horizon': 24,
    'latent_dim': 32,
    'condition_dim': 0,
    'loss': 'MAE',
    'allow_missing_future_covariate': False,
    'decoder_type': 'pointwise_mlp',
    'latent_alignment_loss': 'mae',
    'training_stage': 'stage1_latent',
    'training_stage_plan': '',
    'stage1_epochs': 0,
    'stage2_warmup_epochs': 0,
    'stage2_pred_epochs': 0,
    'stage2_epochs': 0,
    'use_time_condition': False,
    'use_future_x_condition': True,
    'use_modal_condition': False,
    'use_revin': True,
    'revin_eps': 1e-5,
    'debug_eval_during_fit': 0,
    'debug_eval_stage': 'stage2_decoder_mix',
    'latent_mse_weight': 1.0,
    'latent_cosine_weight': 0.0,
    'latent_sigreg_weight': 0.01,
    'latent_detach_target': True,
    'sigreg_std_weight': 1.0,
    'sigreg_cov_weight': 0.04,
    'sigreg_eps': 1e-4,
    'latent_regularizer_type': 'legacy_vc',
    'visreg_scope': 'encoder_all',
    'visreg_use_projector': True,
    'visreg_pooling': 'mean',
    'visreg_projector_dim': 0,
    'visreg_projector_hidden_dim': 2048,
    'visreg_num_projections': 256,
    'visreg_scale_weight': 1.0,
    'visreg_shape_weight': 1.0,
    'visreg_center_weight': 1.0,
    'visreg_eps': 1e-6,
    'decoder_warmup_weight': 0.1,
    'forecast_mse_weight': 1.0,
    'forecast_mae_weight': 0.5,
    'log_stage1_loss_components': 0,
    'log_stage2_loss_components': 0,
    'stage1_checkpoint_path': '',
    'final_checkpoint_path': '',
    'stage_save_dir': '',
}


PATCH_STATE_DEFAULTS = {
    # This baseline is numerical-only and uses known future covariates.
    "use_future_x_condition": True,
    "allow_missing_future_covariate": False,
    "use_modal_condition": False,
    "use_time_condition": False,
    "decoder_type": "causal_patch",
    "decoder_patch_len": 3,
    "decoder_hidden_multiplier": 2,
    "use_revin": True,
    "revin_eps": 1e-5,
    # Local causal state encoder.
    "state_encoder": "causal_patch",
    "state_patch_len": 3,
    "state_hidden_multiplier": 2,
    # Latent PatchTST transition predictor.
    "predictor_patch_len": 4,
    "predictor_patch_stride": 2,
    "predictor_d_model": 64,
    "predictor_n_heads": 4,
    "predictor_e_layers": 2,
    "predictor_d_ff": 128,
    "predictor_dropout": 0.0,
    "predictor_activation": "gelu",
    "predictor_factor": 1,
    # Match the current VoT-side stage-1 objective defaults.
    "latent_alignment_loss": "mae",
    "latent_mse_weight": 10.0,
    "latent_cosine_weight": 15.0,
    "latent_sigreg_weight": 0.1,
    "latent_detach_target": True,
    "sigreg_std_weight": 1.0,
    "sigreg_cov_weight": 0.04,
    "sigreg_eps": 1e-4,
    # Preserve the historical path unless an experiment opts into VISReg.
    "latent_regularizer_type": "legacy_vc",
    "visreg_scope": "encoder_all",
    "visreg_use_projector": True,
    "visreg_pooling": "mean",
    "visreg_projector_dim": 0,
    "visreg_projector_hidden_dim": 2048,
    "visreg_num_projections": 256,
    "visreg_scale_weight": 1.0,
    "visreg_shape_weight": 1.0,
    "visreg_center_weight": 1.0,
    "visreg_eps": 1e-6,
    "decoder_warmup_weight": 0.3,
    "lradj": "constant",
}


class ZeroForecastLoss(nn.Module):
    def forward(self, output, target):
        return output.sum() * 0.0


class WeightedMSEMAELoss(nn.Module):
    def __init__(self, mse_weight: float = 1.0, mae_weight: float = 0.5):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)

    def forward(self, output, target):
        return (
            self.mse_weight * torch.nn.functional.mse_loss(output, target)
            + self.mae_weight * torch.nn.functional.l1_loss(output, target)
        )


class WorldTS(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        obsolete = sorted(
            key for key in kwargs
            if key.startswith("cody_") or key in {
                "stage1_num_epochs", "stage2_mix_num_epochs",
                "stage2_warmup_num_epochs", "stage2_pred_num_epochs", "text_dim",
            }
        )
        if obsolete:
            raise ValueError(
                "Obsolete WorldTS parameter names: {}. Use the renamed parameters "
                "in scripts/w_future/worldts.sh.".format(obsolete)
            )
        config = dict(PATCH_STATE_DEFAULTS)
        config.update(kwargs)
        super(WorldTS, self).__init__(MODEL_HYPER_PARAMS, **config)
        self.check_point = None
        # DeepForecastingModelBase.batch_forecast expects this compatibility
        # hook even when covariates are handled directly by the model.
        self.CovariateFusion = None
        self.latent_state_loss = LatentStateLoss(
            latent_mse_weight=float(getattr(self.config, 'latent_mse_weight', 1.0)),
            latent_alignment_loss=str(getattr(self.config, 'latent_alignment_loss', 'mse')),
            latent_cosine_weight=float(getattr(self.config, 'latent_cosine_weight', 0.0)),
            latent_sigreg_weight=float(getattr(self.config, 'latent_sigreg_weight', 0.01)),
            latent_detach_target=bool(getattr(self.config, 'latent_detach_target', True)),
            sigreg_std_weight=float(getattr(self.config, 'sigreg_std_weight', 1.0)),
            sigreg_cov_weight=float(getattr(self.config, 'sigreg_cov_weight', 0.04)),
            sigreg_eps=float(getattr(self.config, 'sigreg_eps', 1e-4)),
            latent_regularizer_type=str(
                getattr(self.config, 'latent_regularizer_type', 'legacy_vc')
            ),
            visreg_scope=str(getattr(self.config, 'visreg_scope', 'encoder_all')),
            visreg_num_projections=int(
                getattr(self.config, 'visreg_num_projections', 256)
            ),
            visreg_scale_weight=float(
                getattr(self.config, 'visreg_scale_weight', 1.0)
            ),
            visreg_shape_weight=float(
                getattr(self.config, 'visreg_shape_weight', 1.0)
            ),
            visreg_center_weight=float(
                getattr(self.config, 'visreg_center_weight', 1.0)
            ),
            visreg_eps=float(getattr(self.config, 'visreg_eps', 1e-6)),
        )

    @property
    def model_name(self):
        return 'WorldTS'

    def _init_criterion(self):
        stage = getattr(self.config, 'training_stage', 'stage1_latent')
        if stage == 'stage1_latent':
            criterion = ZeroForecastLoss()
        elif self.config.loss == 'MAE':
            criterion = nn.L1Loss()
        elif self.config.loss == 'MSE':
            criterion = nn.MSELoss()
        elif str(self.config.loss).upper() in ('MSE_MAE', 'MSEMAE', 'MSE+MAE', 'MSE_L1'):
            criterion = WeightedMSEMAELoss(
                mse_weight=float(getattr(self.config, 'forecast_mse_weight', 1.0)),
                mae_weight=float(getattr(self.config, 'forecast_mae_weight', 0.5)),
            )
        else:
            criterion = nn.HuberLoss(delta=0.5)
        self.config.criterion = criterion
        return criterion

    def _init_model(self):
        model = WorldTSModel(self.config)
        stage = getattr(self.config, "training_stage", "stage1_latent")
        if str(stage).startswith("stage2"):
            model.set_stage1_eval()
            ckpt_path = getattr(self.config, "stage1_checkpoint_path", "")
            if ckpt_path:
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                state_dict = checkpoint.get("Model", checkpoint)
                missing, unexpected = model.load_state_dict(
                    state_dict,
                    strict=True,
                )
                print("[WorldTS] missing keys:", missing)
                print("[WorldTS] unexpected keys:", unexpected)
            model.configure_for_stage2()
        else:
            model.configure_for_stage1()
        return model

    def _stage_lr(self, stage: str) -> float:
        base_lr = float(self.config.lr)
        if stage == 'stage1_latent':
            override = float(getattr(self.config, 'stage1_lr', 0.0))
        elif str(stage).startswith('stage2'):
            override = float(getattr(self.config, 'stage2_lr', 0.0))
        else:
            override = 0.0
        return override if override > 0.0 else base_lr

    def _stage_lr_config(self, stage_lr: float):
        lr_config = copy.copy(self.config)
        lr_config.lr = float(stage_lr)
        return lr_config

    def _init_optimizer(self, lr: Optional[float] = None):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError('No trainable parameters found.')
        return optim.Adam(params, lr=float(self.config.lr if lr is None else lr))

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    def _clone_state_dict(self, module):
        return copy.deepcopy(module.state_dict())

    def _unwrap_module(self, module):
        return module.module if hasattr(module, 'module') else module

    def _load_model_state_dict(self, state_dict):
        model = self._unwrap_module(self.model)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        print('[WorldTS] missing keys:', missing)
        print('[WorldTS] unexpected keys:', unexpected)

    def _load_checkpoint_state(self, stage: str):
        save_dir = getattr(self.config, 'stage_save_dir', '')
        if not save_dir:
            return
        ckpt_path = os.path.join(save_dir, f'{stage}.pt')
        if not os.path.exists(ckpt_path):
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model_state = ckpt.get('Model', ckpt)
        self._load_model_state_dict(model_state)

    def _save_best_checkpoint(self, stage: str):
        self._save_stage_checkpoint(stage)

    def _validate_latent(self, valid_data_loader, device, exog_dim):
        if valid_data_loader is None:
            return None
        model = self._unwrap_model(self.model)
        model.eval()
        total_loss = []
        component_losses = {
            'latent_mse_loss': [],
            'latent_cosine_loss': [],
            'latent_sigreg_loss': [],
            'latent_visreg_scale_loss': [],
            'latent_visreg_shape_loss': [],
            'latent_visreg_center_loss': [],
            'latent_visreg_hist_loss': [],
            'latent_visreg_target_loss': [],
            'latent_visreg_pred_loss': [],
        }
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                exog_future = target[:, -self.config.horizon:, self.config.series_dim:] if exog_dim > 0 else None
                out_loss = self._process(input, target, input_mark, target_mark, exog_future)
                latent_loss = out_loss.get('additional_loss', None)
                if latent_loss is None:
                    continue
                total_loss.append(float(latent_loss.detach().cpu()))
                for name in component_losses:
                    value = out_loss.get(name, None)
                    if value is not None:
                        component_losses[name].append(float(value.detach().cpu()))
        self.model.train()
        self._last_stage1_valid_components = {
            name: float(np.mean(values)) for name, values in component_losses.items() if values
        }
        return float(np.mean(total_loss)) if total_loss else None

    def _validate_forecast(self, valid_data_loader, series_dim, device, criterion, exog_dim):
        if valid_data_loader is None:
            return None
        model = self._unwrap_model(self.model)
        model.eval()
        total_loss = []
        component_losses = {'stage_forecast_loss': []}
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                exog_future = target[:, -self.config.horizon:, series_dim:] if exog_dim > 0 else None
                output = self._predict_only(
                    input=input,
                    input_mark=input_mark,
                    target_mark=target_mark,
                    exog_future=exog_future,
                )
                target_slice = target[:, -self.config.horizon:, :series_dim]
                output = output[:, -self.config.horizon:, :series_dim]
                output, target_slice = self._post_process(output, target_slice)
                forecast_loss = criterion(output, target_slice)
                total_loss.append(float(forecast_loss.detach().cpu()))
                component_losses['stage_forecast_loss'].append(float(forecast_loss.detach().cpu()))
        self.model.train()
        self._last_stage2_valid_components = {
            name: float(np.mean(values)) for name, values in component_losses.items() if values
        }
        return float(np.mean(total_loss)) if total_loss else None

    @staticmethod
    def _decode_model_states(model, states, model_output):
        decode_kwargs = {
            'target_revin_stats': model_output.get('target_revin_stats'),
        }
        if getattr(model, 'decoder_type', 'pointwise_mlp') == 'causal_patch':
            history_tokens = model_output.get('h_hist')
            if history_tokens is None:
                raise ValueError(
                    'causal patch decoder requires h_hist in model output'
                )
            decode_kwargs['history_tokens'] = history_tokens.detach()
        return model.decode(states, **decode_kwargs)

    def _predict_only(self, input, input_mark, target_mark, exog_future=None):
        """Decode the deployment path without constructing a future target latent."""
        model = self._unwrap_model(self.model)
        model.set_stage1_eval()
        horizon = int(self.config.horizon)
        use_time_condition = bool(getattr(self.config, 'use_time_condition', False))
        time_hist = input_mark if use_time_condition else None
        time_future = (
            target_mark[:, -horizon:, :]
            if use_time_condition and target_mark is not None
            else None
        )
        x_future = None
        if getattr(self.config, 'use_future_x_condition', False):
            if exog_future is not None and exog_future.shape[-1] > 0:
                x_future = exog_future

        with torch.no_grad():
            out = self.model(
                input=input,
                future_target=None,
                time_hist=time_hist,
                time_future=time_future,
                x_future=x_future,
                modal_tokens=None,
            )
        return self._decode_model_states(
            model,
            out['h_pred'].detach(),
            out,
        )

    def _process_forecast(
        self, input, target, input_mark, target_mark, exog_future=None
    ):
        del target
        return {
            'output': self._predict_only(
                input=input,
                input_mark=input_mark,
                target_mark=target_mark,
                exog_future=exog_future,
            )
        }

    def _save_stage_checkpoint(self, stage: str):
        save_dir = getattr(self.config, 'stage_save_dir', '')
        if not save_dir:
            return
        os.makedirs(save_dir, exist_ok=True)
        payload = {'Model': self._clone_state_dict(self.model)}
        torch.save(payload, os.path.join(save_dir, f'{stage}.pt'))

    def _build_stage_plan(self):
        plan = getattr(self.config, 'training_stage_plan', None)
        if plan in (None, ''):
            return ['stage1_latent', 'stage2_decoder_pred']
        if isinstance(plan, str):
            plan = plan.strip()
            if plan.startswith('['):
                try:
                    parsed = json.loads(plan)
                    if isinstance(parsed, list):
                        plan = parsed
                        self._reject_stage3_plan(plan)
                        return plan
                except Exception:
                    pass
            plan = [item.strip() for item in plan.split(',') if item.strip()]
            self._reject_stage3_plan(plan)
            return plan
        plan = list(plan)
        self._reject_stage3_plan(plan)
        return plan

    @staticmethod
    def _reject_stage3_plan(plan) -> None:
        stage3 = [stage for stage in plan if str(stage).startswith('stage3')]
        if stage3:
            raise ValueError(
                'v46B-lite disables stage3; use stage1_latent and stage2_* stages only. '
                f'Got {stage3}.'
            )

    def _get_stage_num_epochs(self, stage: str) -> int:
        if stage == 'stage1_latent':
            value = getattr(self.config, 'stage1_epochs', 0)
        elif stage == 'stage2_decoder_warmup':
            value = getattr(self.config, 'stage2_warmup_epochs', 0)
        elif stage == 'stage2_decoder_pred':
            value = getattr(self.config, 'stage2_pred_epochs', 0)
        elif stage == 'stage2_decoder_mix':
            value = getattr(self.config, 'stage2_epochs', 0)
        else:
            value = 0
        value = int(value)
        return value if value > 0 else int(self.config.num_epochs)

    def _run_stage_train(
        self,
        stage,
        train_data_loader,
        valid_data_loader,
        series_dim,
        device,
        exog_dim,
        debug_batch_forecast_callback=None,
    ):
        if str(stage).startswith('stage3'):
            raise ValueError('v46B-lite disables stage3 training.')
        self.config.training_stage = stage
        model = self._unwrap_model(self.model)
        if stage == 'stage1_latent':
            model.configure_for_stage1()
        elif stage.startswith('stage2'):
            model.configure_for_stage2()
        else:
            raise ValueError(f'Unknown training_stage: {stage}')
        criterion = self._init_criterion()
        stage_lr = self._stage_lr(stage)
        stage_lr_config = self._stage_lr_config(stage_lr)
        optimizer = self._init_optimizer(lr=stage_lr)
        scaler = torch.cuda.amp.GradScaler() if self.config.use_amp == 1 else None
        num_epochs = self._get_stage_num_epochs(stage)

        try:
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f'[WorldTS] stage={stage} trainable_params={total_params} '
                f'num_epochs={num_epochs} lr={stage_lr}'
            )
        except ValueError:
            print(
                f'[WorldTS] stage={stage} trainable_params=lazy-uninitialized '
                f'num_epochs={num_epochs} lr={stage_lr}'
            )

        stage_best_loss = None
        stage_best_state = None
        early_stop = (
            self._init_early_stopping()
            if valid_data_loader is not None
            and stage in (
                'stage1_latent',
                'stage2_decoder_warmup',
                'stage2_decoder_pred',
                'stage2_decoder_mix',
            )
            else None
        )

        for epoch in range(num_epochs):
            self.model.train()
            if stage.startswith('stage2'):
                model.set_stage1_eval()

            for input, target, input_mark, target_mark in train_data_loader:
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                exog_future = target[:, -self.config.horizon:, series_dim:] if exog_dim > 0 else None
                out_loss = self._process(input, target, input_mark, target_mark, exog_future)
                output = out_loss['output']
                additional_loss = out_loss.get('additional_loss', 0)

                target_slice = target[:, -self.config.horizon:, :series_dim]
                output = output[:, -self.config.horizon:, :series_dim]
                output, target_slice = self._post_process(output, target_slice)
                loss = criterion(output, target_slice)
                total_loss = loss + additional_loss

                if self.config.use_amp == 1:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

            if valid_data_loader is not None:
                if stage == 'stage1_latent':
                    valid_loss = self._validate_latent(valid_data_loader, device, exog_dim)
                elif stage in (
                    'stage2_decoder_warmup',
                    'stage2_decoder_pred',
                    'stage2_decoder_mix',
                ):
                    valid_loss = self._validate_forecast(valid_data_loader, series_dim, device, criterion, exog_dim)
                else:
                    valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                if valid_loss is not None:
                    print(f'[WorldTS][validate] stage={stage} epoch={epoch + 1}/{num_epochs} valid_loss={valid_loss:.6f}')
                    if stage == 'stage1_latent' and bool(getattr(self.config, 'log_stage1_loss_components', 0)):
                        components = getattr(self, '_last_stage1_valid_components', {})
                        if components:
                            formatted = ' '.join(f'{name}={value:.6f}' for name, value in sorted(components.items()))
                            print(f'[WorldTS][stage1_components] epoch={epoch + 1}/{num_epochs} total={valid_loss:.6f} {formatted}')
                    if stage.startswith('stage2') and bool(getattr(self.config, 'log_stage2_loss_components', 0)):
                        components = getattr(self, '_last_stage2_valid_components', {})
                        if components:
                            formatted = ' '.join(f'{name}={value:.6f}' for name, value in sorted(components.items()))
                            print(f'[WorldTS][stage2_components] stage={stage} epoch={epoch + 1}/{num_epochs} total={valid_loss:.6f} {formatted}')
                    if np.isnan(valid_loss):
                        raise ValueError(f'valid loss is nan in stage {stage}')
                    if early_stop is not None:
                        improved = early_stop(valid_loss, self.model)
                        if improved:
                            stage_best_loss = valid_loss
                            stage_best_state = self._clone_state_dict(self.model)
                        if early_stop.early_stop:
                            break

            if self.config.lradj != 'TST':
                self._adjust_lr(optimizer, epoch + 1, stage_lr_config)

            if (
                debug_batch_forecast_callback is not None
                and bool(getattr(self.config, 'debug_eval_during_fit', 0))
                and stage == str(getattr(self.config, 'debug_eval_stage', 'stage2_decoder_pred'))
            ):
                debug_batch_forecast_callback(self, stage, epoch + 1)

        if stage_best_state is not None:
            self._load_model_state_dict(stage_best_state)
            self._save_best_checkpoint(stage)
        else:
            self._save_stage_checkpoint(stage)

        self._load_checkpoint_state(stage)

        if stage == 'stage1_latent':
            self.config.stage1_checkpoint_path = os.path.join(getattr(self.config, 'stage_save_dir', ''), f'{stage}.pt')
        elif stage in (
            'stage2_decoder_warmup',
            'stage2_decoder_pred',
            'stage2_decoder_mix',
        ):
            self.config.final_checkpoint_path = os.path.join(getattr(self.config, 'stage_save_dir', ''), f'{stage}.pt')

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> 'ModelBase':
        if covariates is None:
            covariates = {}
        debug_batch_forecast_callback = kwargs.get('debug_batch_forecast_callback', None)

        series_dim = train_valid_data.shape[-1]
        exog_data = covariates.get('exog', None)
        if exog_data is not None:
            train_valid_data = pd.concat([train_valid_data, exog_data], axis=1)
            exog_dim = exog_data.shape[-1]
        else:
            exog_dim = 0

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        self.config.series_dim = series_dim
        self.config.input_dim = series_dim + exog_dim
        self.config.output_dim = series_dim
        self.config.future_x_dim = exog_dim

        self.model = self._init_model()

        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == 'DP':
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

        print('----------------------------------------------------------', self.model_name)
        config = self.config
        train_data, valid_data = train_val_split(train_valid_data, train_ratio_in_tv, config.seq_len)

        if exog_dim > 0:
            self.scaler1.fit(train_data.values[:, :series_dim])
            self.scaler2.fit(train_data.values[:, series_dim:])
            if config.norm:
                scaled_series = self.scaler1.transform(train_data.values[:, :series_dim])
                scaled_exog = self.scaler2.transform(train_data.values[:, series_dim:])
                train_data = pd.DataFrame(
                    np.concatenate((scaled_series, scaled_exog), axis=1),
                    columns=train_data.columns,
                    index=train_data.index,
                )
        else:
            self.scaler1.fit(train_data.values)
            if config.norm:
                train_data = pd.DataFrame(
                    self.scaler1.transform(train_data.values),
                    columns=train_data.columns,
                    index=train_data.index,
                )

        valid_data_loader = None
        if train_ratio_in_tv != 1:
            if config.norm:
                if exog_dim > 0:
                    scaled_series = self.scaler1.transform(valid_data.values[:, :series_dim])
                    scaled_exog = self.scaler2.transform(valid_data.values[:, series_dim:])
                    valid_data = pd.DataFrame(
                        np.concatenate((scaled_series, scaled_exog), axis=1),
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
                else:
                    valid_data = pd.DataFrame(
                        self.scaler1.transform(valid_data.values),
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
            _, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        _, train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.early_stopping = self._init_early_stopping()
        self.model.to(device)

        stage_plan = self._build_stage_plan()
        if stage_plan and stage_plan[0] != 'stage1_latent':
            print('[WorldTS] warning: stage plan does not start with stage1_latent')
        for stage in stage_plan:
            self._run_stage_train(
                stage=stage,
                train_data_loader=train_data_loader,
                valid_data_loader=valid_data_loader,
                series_dim=series_dim,
                device=device,
                exog_dim=exog_dim,
                debug_batch_forecast_callback=debug_batch_forecast_callback,
            )

        final_path = getattr(self.config, 'final_checkpoint_path', '')
        if final_path:
            os.makedirs(os.path.dirname(final_path) or '.', exist_ok=True)
            payload = {'Model': self._clone_state_dict(self.model)}
            torch.save(payload, final_path)

        return self

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        stage = getattr(self.config, 'training_stage', 'stage1_latent')
        if stage.startswith('stage2'):
            self.model.set_stage1_eval()
        horizon = int(self.config.horizon)
        series_dim = int(self.config.series_dim)
        future_target = target[:, -horizon:, :series_dim]
        use_time_condition = bool(getattr(self.config, 'use_time_condition', False))
        time_hist = input_mark if use_time_condition else None
        time_future = target_mark[:, -horizon:, :] if use_time_condition and target_mark is not None else None
        x_future = None
        if getattr(self.config, 'use_future_x_condition', False):
            if exog_future is not None and exog_future.shape[-1] > 0:
                x_future = exog_future
        modal_tokens = None

        # Stage 1 learns only the latent representation and transition dynamics.
        if stage == 'stage1_latent':
            out = self.model(
                input=input,
                future_target=future_target,
                time_hist=time_hist,
                time_future=time_future,
                x_future=x_future,
                modal_tokens=modal_tokens,
            )
            latent_loss = self.latent_state_loss(
                h_pred=out['h_pred'],
                h_target=out['h_target'],
                z_visreg_hist=out.get('z_visreg_hist'),
                z_visreg_target=out.get('z_visreg_target'),
                z_visreg_pred=out.get('z_visreg_pred'),
            )
            dummy_output = torch.zeros_like(future_target)
            result = {
                'output': dummy_output,
                'additional_loss': latent_loss['loss'],
                'latent_mse_loss': latent_loss['mse_loss'],
                'latent_cosine_loss': latent_loss['cosine_loss'],
                'latent_sigreg_loss': latent_loss['sigreg_loss'],
            }
            if str(
                getattr(self.config, 'latent_regularizer_type', 'legacy_vc')
            ).lower() == 'visreg':
                result.update(
                    {
                        'latent_visreg_scale_loss': latent_loss['visreg_scale_loss'],
                        'latent_visreg_shape_loss': latent_loss['visreg_shape_loss'],
                        'latent_visreg_center_loss': latent_loss['visreg_center_loss'],
                        'latent_visreg_hist_loss': latent_loss['visreg_hist_loss'],
                        'latent_visreg_target_loss': latent_loss['visreg_target_loss'],
                        'latent_visreg_pred_loss': latent_loss['visreg_pred_loss'],
                    }
                )
            return result

        # warmup用的是h_target解码训练；pred是用的pred解码训练；mix是两者结合。
        if stage == 'stage2_decoder_warmup':
            with torch.no_grad():
                out = self.model(
                    input=input,
                    future_target=future_target,
                    time_hist=time_hist,
                    time_future=time_future,
                    x_future=x_future,
                    modal_tokens=modal_tokens,
                )
            y_from_target = self._decode_model_states(
                self._unwrap_model(self.model),
                out['h_target'].detach(),
                out,
            )

            # decode也是同时训练了真实的h_target到真实和预测的h_pred到真实的映射
            warm_loss = self.forecast_loss_for_additional(y_from_target, future_target)
            return {'output': y_from_target, 'decoder_warm_loss': warm_loss.detach()}

        if stage == 'stage2_decoder_pred':
            y_pred = self._predict_only(
                input=input,
                input_mark=input_mark,
                target_mark=target_mark,
                exog_future=exog_future,
            )
            pred_loss = self.forecast_loss_for_additional(y_pred, future_target)
            return {'output': y_pred, 'decoder_pred_loss': pred_loss.detach()}

        if stage == 'stage2_decoder_mix':
            with torch.no_grad():
                out = self.model(
                    input=input,
                    future_target=future_target,
                    time_hist=time_hist,
                    time_future=time_future,
                    x_future=x_future,
                    modal_tokens=modal_tokens,
                )
            model = self._unwrap_model(self.model)
            y_pred = self._decode_model_states(
                model,
                out['h_pred'].detach(),
                out,
            )
            y_from_target = self._decode_model_states(
                model,
                out['h_target'].detach(),
                out,
            )
            pred_loss = self.forecast_loss_for_additional(y_pred, future_target)
            warm_loss = self.forecast_loss_for_additional(y_from_target, future_target)
            warm_weighted_loss = getattr(self.config, 'decoder_warmup_weight', 0.1) * warm_loss
            return {
                'output': y_pred,
                'additional_loss': warm_weighted_loss,
                'decoder_pred_loss': pred_loss.detach(),
                'decoder_warm_loss': warm_loss.detach(),
                'decoder_warm_weighted_loss': warm_weighted_loss.detach(),
            }

        raise ValueError(f'Unknown training_stage: {stage}')

    def _forecast_loss_by_name(self, output, target, loss_name: str):
        loss_name = str(loss_name or self.config.loss).strip()
        if not loss_name or loss_name.lower() == 'same':
            loss_name = str(self.config.loss)
        normalized = loss_name.upper()
        if normalized in ('MAE', 'L1'):
            return torch.nn.functional.l1_loss(output, target)
        if normalized == 'MSE':
            return torch.nn.functional.mse_loss(output, target)
        if normalized in ('MSE_MAE', 'MSEMAE', 'MSE+MAE', 'MSE_L1'):
            mse_weight = float(getattr(self.config, 'forecast_mse_weight', 1.0))
            mae_weight = float(getattr(self.config, 'forecast_mae_weight', 0.5))
            return (
                mse_weight * torch.nn.functional.mse_loss(output, target)
                + mae_weight * torch.nn.functional.l1_loss(output, target)
            )
        if normalized in ('HUBER', 'SMOOTHL1', 'SMOOTH_L1'):
            return torch.nn.functional.huber_loss(output, target, delta=0.5)
        raise ValueError(f'Unknown forecast loss: {loss_name}')

    def forecast_loss_for_additional(self, output, target):
        return self._forecast_loss_by_name(output, target, self.config.loss)
