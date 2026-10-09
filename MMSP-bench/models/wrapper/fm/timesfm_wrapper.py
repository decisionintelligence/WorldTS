import logging
from typing import Sequence

import numpy as np
import timesfm
import torch
from gluonts.model.forecast_generator import QuantileForecastGenerator, SampleForecastGenerator
from gluonts.torch.distributions import QuantileOutput
from timesfm.pytorch_patched_decoder import ResidualBlock

from .base import FMWrapperBase

logger = logging.getLogger(__name__)
_TOL = 1e-6


def strip_leading_nans(arr):
    """
    Removes contiguous NaN values from the beginning of a NumPy array.

    Args:
      arr: The input NumPy array.

    Returns:
      A new NumPy array with leading NaN values removed.
      If the array is all NaNs or empty, returns an empty array.
    """

    isnan = torch.isnan(arr)
    first_valid_index = torch.argmax(~isnan)
    return arr[first_valid_index:]


def linear_interpolation(arr):
    """
      Performs linear interpolation to fill NaN values in a 1D numpy array.

      Args:
          arr: The 1D numpy array containing NaN values.

      Returns:
          A new numpy array with NaN values filled using linear interpolation,
          or the original array if no NaNs are present.
          Returns None if the input is not a 1D array.
          Returns the original array if there are no NaN values.
      """

    nans = torch.isnan(arr)
    if not torch.any(nans):  # Check if there are any NaNs
        return arr

    def x(z):
        return z.nonzero()[0]

    nans_indices = x(nans)
    non_nans_indices = x(~nans)
    non_nans_values = arr[~nans]

    try:
        arr[nans] = np.interp(nans_indices, non_nans_indices, non_nans_values)
    except ValueError:
        if len(non_nans_values) > 0:
            mu = torch.nanmean(arr)
        else:
            mu = 0.0
        arr = torch.where(torch.isfinite(arr), arr, mu)
    return arr


# Per time series normalization: forward.
def _normalize(batch):
    stats = [
        (torch.mean(x), torch.where((w := torch.std(x)) > _TOL, w, 1.0)) for x in batch
    ]
    new_batch = [(x - stat[0]) / stat[1] for x, stat in zip(batch, stats)]
    return new_batch, stats


# Per time series normalization: inverse.
def _renormalize(batch, stats):
    return [x * stat[1] + stat[0] for x, stat in zip(batch, stats)]


class TimesFMWrapper(FMWrapperBase):
    @property
    def context_length(self):
        return self.tfm.context_len

    def __init__(self, model_path, ds_freq: str, prediction_length: int, only_quantile_loss: bool = False,
                 normalized_loss: bool = False, sample_output=False):
        super().__init__()
        self.model_path = model_path
        self.ds_freq = ds_freq
        self.freq = timesfm.freq_map(ds_freq)
        self.tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                context_len=2048,
                num_layers=50,
                use_positional_embedding=False,
                output_patch_len=128,
                horizon_len=128
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                path=model_path,
                version="torch"),
        )
        self.model = self.tfm._model
        self.prediction_length = prediction_length
        if self.prediction_length > self.tfm.horizon_len:
            self.tfm.horizon_len = (
                                           (self.prediction_length + self.tfm.output_patch_len - 1) //
                                           self.tfm.output_patch_len) * self.tfm.output_patch_len
            logger.info(f'Jitting for new prediction length {self.tfm.horizon_len}.')
        self.extra_length = self.prediction_length - self.model.config.horizon_len
        if self.extra_length > 0:
            self.extra_ff_layer = ResidualBlock(
                input_dims=self.model.config.hidden_size,
                output_dims=self.extra_length * (1 + len(self.model.config.quantiles)),
                hidden_dims=self.model.config.intermediate_size,
            )
        self.loss_fn = (lambda x, y: torch.mean((x - y.squeeze(-1)) ** 2))
        self.only_quantile_loss = only_quantile_loss
        logger.info("use only quantile loss: %s", self.only_quantile_loss)
        self.quantile_output = QuantileOutput(self.quantiles)
        self.normalized_loss = normalized_loss
        logger.info("use normalized loss: %s", self.normalized_loss)
        self.sample_output = sample_output
        logger.info("use sample output: %s", self.sample_output)

    @property
    def quantiles(self):
        return self.tfm.quantiles

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles) + 1

    @property
    def model_dim(self):
        return self.tfm.model_dims

    def _preprocess(
            self, inputs: Sequence[torch.Tensor],
            input_padding: Sequence[torch.Tensor],
            freq: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Formats and pads raw inputs to feed into the model.

        This function both pads each time series to match the context length, and
        pads the inputs to meet the SPMD shape requirement.

        Args:
          inputs: A list of 1d JTensors. Each JTensor is the context time series of
            a single forecast task.
          freq: list of frequencies

        Returns:
        A tuple of:
        - the padded input time series to meet the model required context.
        - the padding indicator.
        - the frequency of each input time series.
        - the number of padded examples for SPMD so that each core has the same
            number (a multiple of `batch_size`) of examples.
        """

        b, input_len = inputs.size(0), inputs.size(1)
        inputs_ts = inputs
        remainder = input_len % self.model.config.patch_len
        if remainder > 0:
            pad = self.model.config.patch_len - remainder
            inputs_ts = torch.cat([torch.zeros(b, pad).to(inputs.device), inputs_ts], dim=-1)
            input_padding = torch.cat([torch.ones(b, pad).to(inputs.device), input_padding], dim=-1)

        return (
            inputs_ts,
            input_padding,
            torch.LongTensor(freq).reshape(-1, 1).to(inputs.device),
            0
        )

    def tokenize(self, inputs):
        input_ts = inputs["context"]
        input_padding = 1 - inputs["observed_values"]
        freq = [self.freq] * input_ts.size(0)
        # inp_freq = torch.LongTensor(freq).reshape(-1, 1).to(input_ts.device)

        input_ts, input_padding, inp_freq, pmap_pad = self._preprocess(input_ts, input_padding, freq)
        model_input, patched_padding, stats, _ = self.model._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )
        f_emb = self.model.freq_emb(inp_freq)  # B x 1 x D
        model_input += f_emb
        return {"token_embeddings": model_input, "patched_padding": patched_padding,
                "stats": stats, "attention_mask": 1 - patched_padding}

    def encode(self, inputs):
        # model_output = self.stacked_transformer(model_input, patched_padding)
        model_output = self.model.stacked_transformer(hidden_states=inputs["token_embeddings"],
                                                      paddings=inputs["patched_padding"])
        return {"encoded_embeddings": model_output, "stats": inputs["stats"]}

    def predict(self, model_output):
        # output_ts = self._postprocess_output(model_output, num_outputs, stats)
        model_output["num_outputs"] = len(self.quantiles) + 1
        output_ts = self.model._postprocess_output(model_output=model_output["encoded_embeddings"],
                                                   num_outputs=model_output["num_outputs"],
                                                   stats=model_output["stats"])
        if self.extra_length > 0:
            # B x N x (H.Q)
            extra_output_ts = self.extra_ff_layer(model_output["encoded_embeddings"])
            # Reshape using view
            b, n, _ = extra_output_ts.shape
            extra_output_ts = extra_output_ts.view(b, n, self.extra_length, model_output["num_outputs"])
            extra_output_ts = self.model._reverse_transform(extra_output_ts, model_output["stats"])
            output_ts = torch.cat([output_ts, extra_output_ts], dim=-2)
        return {"prediction": output_ts[:, -1, :self.prediction_length], "stats": model_output["stats"]}

    def loss(self, inputs):
        predictions = inputs["prediction"]
        x_future = inputs["future_target"]
        stats = inputs["stats"]
        if self.normalized_loss:
            mu, sigma = stats
            predictions = (predictions - mu[:, None, None]) / sigma[:, None, None]
            x_future = (x_future - mu[:, None]) / sigma[:, None]
        # predictions_mean = predictions[..., 0]
        last_patch_pred = predictions[:, :, 0]
        if self.only_quantile_loss:
            loss = 0
        else:
            loss = self.loss_fn(last_patch_pred, x_future.squeeze(-1))
        loss += self.quantile_output.loss(x_future, (predictions[:, :, 1:],)).mean()
        return loss

    def _quantile_loss(self, pred: torch.Tensor, actual: torch.Tensor,
                       quantile: float) -> torch.Tensor:
        """Calculates quantile loss.
            Args:
                pred: Predicted values
                actual: Actual values
                quantile: Quantile at which loss is computed
            Returns:
                Quantile loss
            """
        dev = actual - pred
        loss_first = dev * quantile
        loss_second = -dev * (1.0 - quantile)
        return 2 * torch.where(loss_first >= 0, loss_first, loss_second)

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forecast_generator(self):
        if self.sample_output:
            return SampleForecastGenerator()
        else:
            return QuantileForecastGenerator(self.quantiles)

    def process_forecast(self, outputs):
        prediction = outputs["prediction"]
        if self.sample_output:
            return prediction[:, :self.prediction_length, 1:].transpose(1, 2)
        else:
            return (prediction[:, :self.prediction_length, 1:],), None, None

    def __getnewargs_ex__(self):
        # Store constructor arguments
        return (self.model_path, self.ds_freq, self.prediction_length, self.only_quantile_loss, self.normalized_loss,
                self.sample_output), {}
