
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import Output
from gluonts.torch.distributions.quantile_output import QuantileOutput
from gluonts.torch.scaler import StdScaler
from gluonts.torch.util import weighted_average

from models.wrapper.fm.base import FMWrapperBase

logger = logging.getLogger(__name__)


class FlattenFeatureEmbedding(nn.Module):
    def __init__(self, cardinalities, embedding_dim):
        super().__init__()
        self.embedder = nn.Embedding(sum(cardinalities), embedding_dim)
        self.register_buffer("offset",
                             torch.tensor([0] + list(np.cumsum(cardinalities)[:-1]), dtype=torch.long).unsqueeze(0))
        self.embedding_dim = embedding_dim
        self.num_vars = len(cardinalities)

    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, self.num_vars)
        x = x + self.offset
        return self.embedder(x).view(*x_shape[:-1], self.num_vars * self.embedding_dim)


class SFTModel(nn.Module):

    @validated()
    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            model_wrapper: Optional[FMWrapperBase] = None,
            num_heads: int = 4,
            dropout_rate: float = 0.1,
            distr_output: Optional[Output] = None,
            future_shortcut=False,
            add_item_id=False,
            *args,
            **kwargs
    ):
        super().__init__()

        self.model_wrapper = model_wrapper
        # freeze the inner model

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_heads = num_heads
        self.d_hidden = self.model_wrapper.model_dim
        self.d_var = self.model_wrapper.model_dim
        self.dropout_rate = dropout_rate
        self.add_item_id = add_item_id
        self.future_shortcut = future_shortcut

        if distr_output is None:
            distr_output = QuantileOutput(
                self.model_wrapper.quantiles
            )
        self.distr_output = distr_output
        self.args_proj = self.distr_output.get_args_proj(
            in_features=self.d_hidden
        )
        self.scaler = StdScaler(dim=1, keepdim=True)


    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
            },
            torch.zeros,
        )

    def input_types(self) -> Dict[str, torch.dtype]:
        return {
            "past_target": torch.float,
            "past_observed_values": torch.float,
            "feat_static_real": torch.float,
            "feat_static_cat": torch.long,
            "feat_dynamic_real": torch.float,
            "feat_dynamic_cat": torch.long,
            "past_feat_dynamic_real": torch.float,
            "past_observed_feat_dynamic_real": torch.float,
            "observed_feat_dynamic_real": torch.float,
        }

    def forward(
            self,
            past_target: torch.Tensor,  # [N, T]
            past_observed_values: torch.Tensor,  # [N, T]
            is_train=False,
            **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:

        # past_target, loc, scale = self.scaler(past_target, weights=past_observed_values)
        inputs = {"context": past_target, "observed_values": past_observed_values}
        target_tokenize_output = self.model_wrapper.tokenize(inputs)

        ctx_input = target_tokenize_output["token_embeddings"]
        target_tokenize_output["token_embeddings"] = ctx_input
        encoded_output = self.model_wrapper.encode(target_tokenize_output)
        decode_output = self.model_wrapper.predict(encoded_output)
        if is_train:
            return decode_output
        else:
            return self.model_wrapper.process_forecast(decode_output)

    def loss(
            self,
            future_target: torch.Tensor,  # [N, H]
            future_observed_values: torch.Tensor,  # [N, H]
            **input_kwargs
    ) -> torch.Tensor:
        output = self(
            is_train=True,
            **input_kwargs
        )
        inputs = {
            "future_target": future_target,
            "future_observed_values": future_observed_values,
            **output
        }
        return self.model_wrapper.loss(inputs)
