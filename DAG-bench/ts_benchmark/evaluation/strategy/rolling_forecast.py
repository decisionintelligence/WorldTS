# -*- coding: utf-8 -*-
import itertools
import time
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.utils.data_processing import split_time
from ts_benchmark.utils.data_processing import split_channel


class RollingForecastEvalBatchMaker:
    def __init__(
        self,
        series: pd.DataFrame,
        index_list: List[int],
        covariates: Optional[dict] = None,
    ):
        self.series = series
        self.index_list = index_list
        self.current_sample_count = 0
        self.covariates = covariates

    def make_batch_predict(self, batch_size: int, win_size: int) -> dict:
        """
        Return a batch of data with index and column to be used for batch prediction.

        :param batch_size: The size of batch.
        :param win_size: The length of data used for prediction.
        :return: a batch of data and its time stamps.
        """
        index_list = self.index_list[
            self.current_sample_count : self.current_sample_count + batch_size
        ]
        series = self.series.values
        predict_batch = self._make_batch_data(
            series, np.array(index_list) - win_size, win_size
        )

        indexes = self.series.index
        time_stamps_batch = self._make_batch_data(
            indexes, np.array(index_list) - win_size, win_size
        )
        covariates_batch = self._make_batch_covariates(
            np.array(index_list) - win_size, win_size
        )
        self.current_sample_count += len(index_list)
        return {
            "input": predict_batch,
            "time_stamps": time_stamps_batch,
            "covariates": covariates_batch,
        }

    def make_batch_eval(self, horizon: int) -> dict:
        """
        Return all data to be used for batch evaluation.

        :param horizon: The size of horizon.
        :return: All data to be used for batch evaluation.
        """
        series = self.series.values
        test_batch = self._make_batch_data(series, np.array(self.index_list), horizon)
        covariates_batch = self._make_batch_covariates(
            np.array(self.index_list), horizon
        )
        return {
            "target": test_batch,
            "covariates": covariates_batch,
        }

    def _make_batch_covariates(self, index_list: np.ndarray, win_size: int) -> Dict:
        """
        Create a batch of covariates

        :param index_list: An array of starting indices for each window.
        :param win_size: The size of each window.
        :return: A batch of covariates.
        """
        covariates = {} if self.covariates is None else self.covariates
        covariates_batch = {}
        if covariates.get("exog") is not None:
            covariates_batch["exog"] = self._make_batch_data(
                self.covariates["exog"], index_list, win_size
            )
        return covariates_batch

    @staticmethod
    def _make_batch_data(
        data: Any, index_list: np.ndarray, win_size: int
    ) -> np.ndarray:
        """
        Create a batch of data

        :param data: Array_like. Array to create the batch.
        :param index_list: An array of starting indices for each window.
        :param win_size: The size of each window.
        :return: A batch of data.
        """
        windows = sliding_window_view(data, window_shape=(win_size, *data.shape[1:]))
        data_batch = windows[index_list]
        data_batch = np.squeeze(data_batch, axis=tuple(range(1, np.ndim(data))))
        return data_batch

    def has_more_batches(self) -> bool:
        """
        Check if there are more batches to process.

        :return: True if there are more batches, False otherwise.
        """
        return self.current_sample_count < len(self.index_list)


class RollingForecastPredictBatchMaker(BatchMaker):
    def __init__(self, batch_maker: RollingForecastEvalBatchMaker):
        self._batch_maker = batch_maker

    def make_batch(self, batch_size: int, win_size: int) -> dict:
        """
        Return a batch of data to be used for batch prediction.

        :param batch_size: The size of batch.
        :param win_size: The length of data used for prediction.
        :return: A batch of data.
        """
        return self._batch_maker.make_batch_predict(batch_size, win_size)

    def has_more_batches(self) -> bool:
        """
        Check if there are more batches to process.

        :return: True if there are more batches, False otherwise.
        """
        return self._batch_maker.has_more_batches()


class RollingForecast(ForecastingStrategy):
    """
    Rolling forecast strategy class

    This strategy defines a forecasting task that fits once on the training set and
    forecasts on the testing set in a rolling window style.

    The required strategy configs include:

    - horizon (int): The length of each prediction;
    - tv_ratio (float): The ratio of the train-validation series when performing
      train-test split;
    - train_ratio_in_tv (float): The ratio of the training series when performing
      train-validation split;
    - stride (int): Rolling stride, i.e. the interval between two windows;
    - num_rollings (int): The maximum number of steps to forecast;

    The accepted metrics include all regression metrics.

    The return fields other than the specified metrics are (in order):

    - FieldNames.FILE_NAME: The name of the series;
    - FieldNames.FIT_TIME: The training time;
    - FieldNames.INFERENCE_TIME: The inference time;
    - FieldNames.ACTUAL_DATA: The true test data, encoded as a string.
    - FieldNames.INFERENCE_DATA: The predicted data, encoded as a string.
    - FieldNames.LOG_INFO: Any log returned by the evaluator.
    """

    REQUIRED_CONFIGS = [
        "horizon",
        "tv_ratio",
        "train_ratio_in_tv",
        "stride",
        "num_rollings",
        "save_true_pred",
        "target_channel",
    ]

    @staticmethod
    def _get_index(
        train_length: int, test_length: int, horizon: int, stride: int
    ) -> List[int]:
        """
        Get the index list of the rolling windows.

        :param train_length: Training data length.
        :param test_length: Test data length.
        :param horizon: Prediction length.
        :param stride: Rolling stride.
        :return: Index list of the rolling windows.
        """
        data_len = train_length + test_length
        index_list = list(range(train_length, data_len - horizon + 1, stride)) + (
            [data_len - horizon] if (test_length - horizon) % stride != 0 else []
        )
        return index_list

    def _get_split_lens(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        tv_ratio: float,
    ) -> Tuple[int, int]:
        """
        Gets the size of the train-validation series and the test series

        :param series: Target series.
        :param meta_info: Meta-information of the target series.
        :param tv_ratio: The ratio of the train-validation series when performing
            train-test split;
        :return: The length of the train-validation series, and the length of the test series.
        """
        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = int(tv_ratio * data_len)
        test_length = data_len - train_length
        if train_length <= 0 or test_length <= 0:
            raise ValueError(
                "The length of training or testing data is less than or equal to 0"
            )
        return train_length, test_length

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        """
        The entry function of execution pipeline of forecasting tasks

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model_factory: The factory to create models.
        :param series_name: the name of the target series.
        :return: The evaluation results.
        """
        model = model_factory()
        if model.batch_forecast.__annotations__.get("not_implemented_batch"):
            return self._eval_sample(series, meta_info, model, series_name)
        else:
            return self._eval_batch(series, meta_info, model, series_name)

    def _eval_sample(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        """
        The sample execution pipeline of forecasting tasks.

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model: The model used for prediction.
        :param series_name: the name of the target series.
        :return: The evaluation results.
        """
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

        train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid_data, test_data = split_time(series, train_length)

        target_train_valid_data, exog_data = split_channel(
            train_valid_data, target_channel
        )
        covariates_train = {}
        covariates_train["exog"] = exog_data

        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid_data,
            covariates=covariates_train,
            train_ratio_in_tv=train_ratio_in_tv,
        )
        end_fit_time = time.time()

        eval_scaler = self._get_eval_scaler(target_train_valid_data, train_ratio_in_tv)

        index_list = self._get_index(train_length, test_length, horizon, stride)
        total_inference_time = 0
        all_test_results = []
        all_rolling_actual = []
        all_rolling_predict = []
        for i, index in itertools.islice(enumerate(index_list), num_rollings):
            train, rest = split_time(series, index)
            test, _ = split_channel(split_time(rest, horizon)[0], target_channel)
            target_train, exog_train = split_channel(train, target_channel)
            covariates_forecast = {}
            covariates_forecast["exog"] = exog_train

            start_inference_time = time.time()
            predict = model.forecast(
                horizon, target_train, covariates=covariates_forecast
            )
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time

            single_series_result = self.evaluator.evaluate(
                test.to_numpy(), predict, eval_scaler, target_train_valid_data.values
            )
            inference_data = pd.DataFrame(
                predict, columns=test.columns, index=test.index
            )

            all_rolling_actual.append(test)
            all_rolling_predict.append(inference_data)
            all_test_results.append(single_series_result)

        average_inference_time = float(total_inference_time) / min(
            len(index_list), num_rollings
        )
        single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = (
            self._encode_data(all_rolling_actual) if save_true_pred else np.nan
        )
        inference_data_encoded = (
            self._encode_data(all_rolling_predict) if save_true_pred else np.nan
        )

        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            average_inference_time,
            actual_data_encoded,
            inference_data_encoded,
            "",
        ]
        return single_series_results

    def _eval_batch(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        """
        The batch execution pipeline of forecasting tasks.

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model: The model used for prediction.
        :param series_name: The name of the target series.
        :return: The evaluation results.
        """
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        stride = self._get_scalar_config_value("stride", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)

        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

        train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid_data, test_data = split_time(series, train_length)

        target_train_valid_data, exog_train_valid_data = split_channel(
            train_valid_data, target_channel
        )
        target4batch, exog_data4batch = split_channel(series, target_channel)
        covariates_train, covariates4batch = {}, {}
        covariates_train["exog"] = exog_train_valid_data
        covariates4batch["exog"] = exog_data4batch

        eval_scaler = self._get_eval_scaler(target_train_valid_data, train_ratio_in_tv)

        index_list = self._get_index(train_length, test_length, horizon, stride)
        index_list = index_list[:num_rollings]

        debug_batch_forecast_callback = None
        model_config = getattr(model, "config", None)
        if bool(getattr(model_config, "debug_eval_during_fit", 0)):
            metric_names = self.evaluator.metric_names

            def debug_batch_forecast_callback(fitted_model, stage, epoch):
                # Keep this diagnostic path identical to the final batch
                # evaluation boundary: predictions must come from
                # fitted_model.batch_forecast(...), not from a custom forward.
                debug_batch_maker = RollingForecastEvalBatchMaker(
                    target4batch,
                    list(index_list),
                    covariates4batch,
                )
                debug_predict_batch_maker = RollingForecastPredictBatchMaker(
                    debug_batch_maker
                )
                debug_eval_batch = debug_batch_maker.make_batch_eval(horizon)
                debug_exog_futures = debug_eval_batch["covariates"].get("exog", None)
                debug_targets = debug_eval_batch["target"]
                debug_predicts = []
                debug_i = 0
                while debug_predict_batch_maker.has_more_batches():
                    debug_predicts.append(
                        fitted_model.batch_forecast(
                            horizon,
                            debug_predict_batch_maker,
                            debug_exog_futures,
                            debug_i,
                        )
                    )
                    debug_i += 1
                debug_predicts = np.concatenate(debug_predicts, axis=0)
                if len(debug_targets) != len(debug_predicts):
                    raise RuntimeError(
                        "Diagnostic predictions' len don't equal targets' len!"
                    )
                debug_results = []
                for predicts, target in zip(debug_predicts, debug_targets):
                    debug_results.append(
                        self.evaluator.evaluate(
                            target,
                            predicts,
                            eval_scaler,
                            target_train_valid_data.values,
                        )
                    )
                debug_mean = np.mean(np.stack(debug_results), axis=0).tolist()
                debug_text = ", ".join(
                    f"{name}={value:.6f}" for name, value in zip(metric_names, debug_mean)
                )
                print(
                    f"[batch_forecast][diagnostic] stage={stage} epoch={epoch} {debug_text}"
                )
                return debug_mean

        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid_data,
            covariates=covariates_train,
            train_ratio_in_tv=train_ratio_in_tv,
            debug_batch_forecast_callback=debug_batch_forecast_callback,
        )
        end_fit_time = time.time()

        batch_maker = RollingForecastEvalBatchMaker(
            target4batch,
            index_list,
            covariates4batch,
        )

        all_predicts = []
        condition_mod_diags = []
        hist_future_fusion_diags = []
        total_inference_time = 0
        predict_batch_maker = RollingForecastPredictBatchMaker(batch_maker)
        exog_futures = batch_maker.make_batch_eval(horizon)["covariates"].get(
            "exog", None
        )
        i = 0
        while predict_batch_maker.has_more_batches():
            start_inference_time = time.time()
            predicts = model.batch_forecast(
                horizon, predict_batch_maker, exog_futures, i
            )
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time
            all_predicts.append(predicts)
            condition_mod_getter = getattr(model, "get_last_condition_mod_diagnostics", None)
            condition_mod_diag = condition_mod_getter() if condition_mod_getter is not None else None
            if condition_mod_diag is not None:
                condition_mod_diags.append(condition_mod_diag)
            hist_future_getter = getattr(model, "get_last_hist_future_fusion_diagnostics", None)
            hist_future_diag = hist_future_getter() if hist_future_getter is not None else None
            if hist_future_diag is not None:
                hist_future_fusion_diags.append(hist_future_diag)
            i = i + 1

        all_predicts = np.concatenate(all_predicts, axis=0)
        targets = batch_maker.make_batch_eval(horizon)["target"]
        if len(targets) != len(all_predicts):
            raise RuntimeError("Predictions' len don't equal targets' len!")

        all_test_results = []
        for predicts, target in zip(all_predicts, targets):
            single_series_results = self.evaluator.evaluate(
                target,
                predicts,
                eval_scaler,
                target_train_valid_data.values,
            )
            all_test_results.append(single_series_results)
        single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()
        log_messages = []
        if condition_mod_diags:
            condition_mod_info = (
                "[condition_mod_diagnostic] "
                f"horizon={horizon} "
                f"gate_mean={np.mean([item['gate_mean'] for item in condition_mod_diags]):.6f} "
                f"gate_std={np.mean([item['gate_std'] for item in condition_mod_diags]):.6f} "
                f"scale_mean={np.mean([item['scale_mean'] for item in condition_mod_diags]):.6f} "
                f"scale_std={np.mean([item['scale_std'] for item in condition_mod_diags]):.6f} "
                f"adaptive_scale_mean={np.nanmean([item.get('adaptive_scale_mean', np.nan) for item in condition_mod_diags]):.6f} "
                f"adaptive_scale_std={np.nanmean([item.get('adaptive_scale_std', np.nan) for item in condition_mod_diags]):.6f} "
                f"adaptive_multiplier_mean={np.nanmean([item.get('adaptive_multiplier_mean', np.nan) for item in condition_mod_diags]):.6f} "
                f"adaptive_multiplier_std={np.nanmean([item.get('adaptive_multiplier_std', np.nan) for item in condition_mod_diags]):.6f} "
                f"residual_mean_abs={np.mean([item['residual_mean_abs'] for item in condition_mod_diags]):.9f} "
                f"delta_norm_ratio={np.mean([item['delta_norm_ratio'] for item in condition_mod_diags]):.9f}"
            )
            print(condition_mod_info)
            log_messages.append(condition_mod_info)
        if hist_future_fusion_diags:
            hist_future_fusion_info = (
                "[hist_future_fusion_diagnostic] "
                f"horizon={horizon} "
                f"hist_mix_mean={np.mean([item['hist_mix_mean'] for item in hist_future_fusion_diags]):.6f} "
                f"hist_mix_std={np.mean([item['hist_mix_std'] for item in hist_future_fusion_diags]):.6f} "
                f"future_mix_mean={np.mean([item['future_mix_mean'] for item in hist_future_fusion_diags]):.6f} "
                f"future_mix_std={np.mean([item['future_mix_std'] for item in hist_future_fusion_diags]):.6f} "
                f"future_gate_mean={np.mean([item['future_gate_mean'] for item in hist_future_fusion_diags]):.6f} "
                f"future_gate_std={np.mean([item['future_gate_std'] for item in hist_future_fusion_diags]):.6f} "
                f"future_condition_norm={np.mean([item['future_condition_norm'] for item in hist_future_fusion_diags]):.9f} "
                f"h_base_norm={np.mean([item['h_base_norm'] for item in hist_future_fusion_diags]):.9f} "
                f"h_fused_shift_ratio={np.mean([item['h_fused_shift_ratio'] for item in hist_future_fusion_diags]):.9f}"
            )
            print(hist_future_fusion_info)
            log_messages.append(hist_future_fusion_info)
        log_info = "\n".join(log_messages)

        average_inference_time = float(total_inference_time) / min(
            len(index_list), num_rollings
        )

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = self._encode_data(targets) if save_true_pred else np.nan
        inference_data_encoded = (
            self._encode_data(all_predicts) if save_true_pred else np.nan
        )

        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            average_inference_time,
            actual_data_encoded,
            inference_data_encoded,
            log_info,
        ]
        return single_series_results

    @staticmethod
    def accepted_metrics() -> List[str]:
        return regression_metrics.__all__

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]
