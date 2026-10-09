import inspect
import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Type

import numpy as np
import pandas as pd
from gluonts.core.component import validated
from gluonts.dataset import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import LastValueImputation, MissingValueImputation
from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive,
)


@dataclass
class ModelConfig:
    quantile_levels: Optional[List[float]] = None
    forecast_keys: List[str] = field(init=False)
    statsforecast_keys: List[str] = field(init=False)
    intervals: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return

        intervals = set()

        for quantile_level in self.quantile_levels:
            interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")

        self.intervals = sorted(intervals)


class StatsForecastPredictor(RepresentablePredictor):
    """
    A predictor type that wraps models from the `statsforecast`_ package.

    This class is used via subclassing and setting the ``ModelType`` class
    attribute to specify the ``statsforecast`` model type to use.

    .. _statsforecast: https://github.com/Nixtla/statsforecast

    Parameters
    ----------
    prediction_length
        Prediction length for the model to use.
    quantile_levels
        Optional list of quantile levels that we want predictions for.
        Note: this is only supported by specific types of models, such as
        ``AutoARIMA``. By default this is ``None``, giving only the mean
        prediction.
    **model_params
        Keyword arguments to be passed to the model type for construction.
        The specific arguments accepted or required depend on the
        ``ModelType``; please refer to the documentation of ``statsforecast``
        for details.
    """

    ModelType: Type

    @validated()
    def __init__(
            self,
            prediction_length: int,
            season_length: int,
            freq: str,
            quantile_levels: Optional[List[float]] = None,
            imputation_method: MissingValueImputation = LastValueImputation(),
            max_length: Optional[int] = None,
            batch_size: int = 1,
            parallel: bool = False,
            **model_params,
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        if "season_length" in inspect.signature(self.ModelType.__init__).parameters:
            model_params["season_length"] = season_length

        self.freq = freq
        self.model = StatsForecast(
            models=[self.ModelType(**model_params)],
            freq=freq,
            fallback_model=SeasonalNaive(season_length=season_length),
            n_jobs=-1 if parallel else 1,
        )
        self.fallback_model = StatsForecast(
            # Fallback model when main model returns NaNs
            models=[SeasonalNaive(season_length=season_length)],
            freq=freq,
            n_jobs=-1 if parallel else 1,
        )
        self.config = ModelConfig(quantile_levels=quantile_levels)
        self.imputation_method = imputation_method
        self.batch_size = batch_size
        self.max_length = max_length

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        batch = {}
        total_series = len(dataset)
        self.logger.info(f"Starting prediction on {total_series} series.")

        for idx, entry in enumerate(dataset):
            assert entry["target"].ndim == 1, "only for univariate time series"
            assert (
                    len(entry["target"]) >= 1
            ), "all time series should have at least one data point"

            if self.max_length is not None:
                entry["start"] += len(entry["target"][: -self.max_length])
                entry["target"] = entry["target"][-self.max_length:]

            target = np.asarray(entry["target"], np.float32)
            if np.isnan(target).any():
                target = target.copy()
                target = self.imputation_method(target)

            unique_id = (
                f"{entry['item_id']}_{str(forecast_start(entry))}_{str(len(batch))}"
            )
            start = entry["start"]
            batch[unique_id] = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": pd.date_range(
                        start=start.to_timestamp(),
                        periods=len(target),
                        freq=start.freq,
                    ).to_numpy(),
                    "y": target,
                }
            )

            if len(batch) == self.batch_size:
                self.logger.info(f"Processing batch {idx // self.batch_size + 1}.")
                results = self.sf_predict(pd.concat(batch.values()))
                yield from self.yield_forecast(batch.keys(), results)
                batch = {}

        if len(batch) > 0:
            self.logger.info(f"Processing final batch.")
            results = self.sf_predict(pd.concat(batch.values()))
            yield from self.yield_forecast(batch.keys(), results)

        self.logger.info("Prediction completed.")

    def sf_predict(self, Y_df: pd.DataFrame) -> pd.DataFrame:
        kwargs = {}
        if self.config.intervals is not None:
            kwargs["level"] = self.config.intervals
        results = self.model.forecast(
            df=Y_df,
            h=self.prediction_length,
            **kwargs,
        )
        # replace nan results with fallback
        row_nan = results.isnull().values.any(axis=-1)
        if row_nan.any():
            nan_ids = results[row_nan].index.values
            nan_df = Y_df[Y_df["unique_id"].isin(nan_ids)]
            fallback_results = self.fallback_model.forecast(
                df=nan_df,
                h=self.prediction_length,
                **kwargs,
            )
            results = pd.concat(
                [
                    results[~results.index.isin(nan_ids)],
                    fallback_results,
                ]
            )

        return results

    def yield_forecast(
            self, item_ids, results: pd.DataFrame
    ) -> Iterator[QuantileForecast]:
        # set the index to the unique_id
        if results.index.name != "unique_id":
            results = results.set_index("unique_id")
        for idx in item_ids:
            prediction = results.loc[idx]
            forecast_arrays = []
            model_name = self.ModelType.__name__
            for key in self.config.statsforecast_keys:
                if key == "mean":
                    forecast_arrays.append(prediction.loc[:, model_name].to_numpy())
                else:
                    forecast_arrays.append(
                        prediction.loc[:, f"{model_name}-{key}"].to_numpy()
                    )

            yield QuantileForecast(
                forecast_arrays=np.stack(forecast_arrays, axis=0),
                forecast_keys=self.config.forecast_keys,
                start_date=prediction.ds.iloc[0].to_period(freq=self.freq),
                item_id=idx,
            )

