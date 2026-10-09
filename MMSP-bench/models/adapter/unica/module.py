
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import Output
from gluonts.torch.distributions.quantile_output import QuantileOutput
from gluonts.torch.modules.feature import FeatureEmbedder as BaseFeatureEmbedder
from gluonts.torch.scaler import StdScaler
from transformers import AutoConfig

from models.adapter.layers import (
    FeatureEmbedder,
    FeatureProjector,
    GatedResidualNetwork,
    ParameterizedGatedLinearUnit,
    VariableSelectionNetwork
)
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
        # return self.embedder(x + self.offset)


class UniCA(nn.Module):

    @validated()
    def __init__(self, context_length: int, prediction_length: int, model_wrapper: Optional[FMWrapperBase] = None,
                 d_feat_static_real: Optional[List[int]] = None, c_feat_static_cat: Optional[List[int]] = None,
                 d_feat_dynamic_real: Optional[List[int]] = None, c_feat_dynamic_cat: Optional[List[int]] = None,
                 d_past_feat_dynamic_real: Optional[List[int]] = None,
                 c_past_feat_dynamic_cat: Optional[List[int]] = None, c_item_index: int = 1, num_heads: int = 1,
                 dropout_rate: float = 0.1, distr_output: Optional[Output] = None, with_gate=False,
                 future_with_gate=False, with_future=False, with_dc=False, with_static=True, with_past=True,
                 past_pos="pre", future_pos="post",
                 satellite_encoder=None,
                 use_satellite=True,
                 satellite_fusion="concat",
                 d_multi_modal=4,  # "concat", "attention", "addition"
                 d_down_sample=-1,
                 use_text=False,
                 encoder_path="bert-base",  # "concat", "attention", "addition"
                 homogenizer_type="linear",
                 *args, **kwargs):
        super().__init__()

        self.model_wrapper = model_wrapper
        self.d_down_sample = d_down_sample
        # freeze the inner model
        logger.info("%%%%%%% Freezing the inner model %%%%%%%")
        self.model_wrapper.freeze_model()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_heads = num_heads
        self.d_hidden = self.model_wrapper.model_dim
        self.d_var = self.model_wrapper.model_dim
        self.dropout_rate = dropout_rate
        self.past_pos = past_pos
        self.future_pos = future_pos
        if d_down_sample > 0:
            self.down_sample_network = nn.Linear(d_feat_static_real, d_down_sample)

        logger.info(f"&&&&&&&&&&&&&&&& fuse past at {self.past_pos} &&&&&&&&&&&&&&&&&")
        logger.info(f"&&&&&&&&&&&&&&&& fuse future at {self.future_pos} &&&&&&&&&&&&&&&&&")

        if distr_output is None:
            distr_output = QuantileOutput(
                self.model_wrapper.quantiles
            )
        self.distr_output = distr_output
        self.args_proj = self.distr_output.get_args_proj(
            in_features=self.d_hidden
        )

        self.d_feat_static_real = d_feat_static_real or [1]
        self.d_feat_dynamic_real = d_feat_dynamic_real or [1]
        self.d_past_feat_dynamic_real = d_past_feat_dynamic_real or []
        self.c_feat_static_cat = c_feat_static_cat or [1]
        self.c_feat_dynamic_cat = c_feat_dynamic_cat or []
        self.c_past_feat_dynamic_cat = c_past_feat_dynamic_cat or []
        self.num_feat_static = len(self.d_feat_static_real) + len(
            self.c_feat_static_cat
        )
        self.num_feat_dynamic = sum(self.d_feat_dynamic_real)
        self.num_past_feat_dynamic = sum(self.d_past_feat_dynamic_real)
        self.with_past = with_past
        # self.with_dc = with_dc
        # if with_dc:
        self.num_feat_dynamic += len(self.c_feat_dynamic_cat)
        self.num_past_feat_dynamic += len(self.c_past_feat_dynamic_cat)
        self.scaler = StdScaler(dim=1, keepdim=True)

        self.target_proj = nn.Linear(in_features=1, out_features=self.d_var)
        self.with_gate = with_gate
        self.with_future = with_future
        if with_gate:
            self.cov_gate = nn.Sequential(
                nn.Linear(
                    in_features=self.d_hidden,
                    out_features=self.d_hidden * 2,
                ),
                ParameterizedGatedLinearUnit(self.d_hidden, nonlinear=False),
            )
        self.future_with_gate = future_with_gate
        if with_future:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.d_hidden,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
            )
            #
            # self.attention = CrossAttentionLayer(d_hidden=self.d_hidden,
            #                                      dropout=dropout_rate, dim_feedforward=self.d_hidden)
            # self.attention = nn.TransformerEncoderLayer(
            #     d_model=self.d_hidden, nhead=1, batch_first=True,
            #     dropout=dropout_rate, dim_feedforward=self.d_hidden
            # )
            self.past_future_indicator = nn.Parameter(torch.randn(1, 2, self.d_hidden))
            if future_with_gate:
                self.future_cov_gate = nn.Sequential(
                    nn.Linear(
                        in_features=self.d_hidden,
                        out_features=self.d_hidden * 2,
                    ),
                    ParameterizedGatedLinearUnit(self.d_hidden, nonlinear=False),
                )

        if self.c_past_feat_dynamic_cat:
            self.past_feat_dynamic_embed: Optional[FeatureEmbedder] = (
                BaseFeatureEmbedder(
                    cardinalities=self.c_past_feat_dynamic_cat,
                    embedding_dims=[1]
                                   * len(self.c_past_feat_dynamic_cat),
                )
            )
        else:
            self.past_feat_dynamic_embed = None

        if self.c_feat_dynamic_cat:
            self.feat_dynamic_embed: Optional[FeatureEmbedder] = (
                BaseFeatureEmbedder(
                    cardinalities=self.c_feat_dynamic_cat,
                    embedding_dims=[1] * len(self.c_feat_dynamic_cat),
                )
            )
            # self.back_project = nn.Linear(
            #     in_features=self.d_var * len(self.c_feat_dynamic_cat),
            #     out_features=len(self.c_feat_dynamic_cat)
            # )
            # self.feat_dynamic_embed = FlattenFeatureEmbedding(
            #     cardinalities=self.c_feat_dynamic_cat,
            #     embedding_dim=1
            # )

        else:
            self.feat_dynamic_embed = None
        self.with_static = with_static
        if self.with_static:
            # Static features
            if self.d_feat_static_real:
                self.feat_static_proj: Optional[FeatureProjector] = (
                    FeatureProjector(
                        feature_dims=self.d_feat_static_real,
                        embedding_dims=[self.d_var] * len(self.d_feat_static_real),
                    )
                )
            else:
                self.feat_static_proj = None

            if self.c_feat_static_cat:
                self.feat_static_embed: Optional[FeatureEmbedder] = (
                    FeatureEmbedder(
                        cardinalities=self.c_feat_static_cat,
                        embedding_dims=[self.d_var] * len(self.c_feat_static_cat),
                    )
                )
            else:
                self.feat_static_embed = None
        self.static_selector = VariableSelectionNetwork(
            d_hidden=self.d_var,
            num_vars=self.num_feat_static,
            dropout=self.dropout_rate,
        )

        self.ctx_selector = VariableSelectionNetwork(
            d_hidden=self.d_var,
            num_vars=self.num_past_feat_dynamic + self.num_feat_dynamic + 1 + (
                d_multi_modal if (use_satellite or use_text) else 0),
            add_static=self.with_static,
            dropout=self.dropout_rate,
        )
        self.tgt_selector = VariableSelectionNetwork(
            d_hidden=self.d_var,
            num_vars=self.num_feat_dynamic,
            add_static=self.with_static,
            dropout=self.dropout_rate,
        )
        self.selection = GatedResidualNetwork(
            d_hidden=self.d_var,
            dropout=self.dropout_rate,
        )

        self.use_satellite = use_satellite
        self.satellite_fusion = satellite_fusion
        self.d_multi_modal = d_multi_modal
        if use_satellite:
            if satellite_encoder is None:
                self.satellite_encoder = nn.Sequential(
                    nn.Conv2d(4, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(32 * (64 // 4) * (64 // 4), self.d_hidden)
                )
            else:
                self.satellite_encoder = satellite_encoder
            if homogenizer_type == "linear":
                self.homogenization_projection = nn.Linear(self.d_hidden, d_multi_modal)
            elif homogenizer_type == "mlp":
                self.homogenization_projection = nn.Sequential(
                    nn.Linear(self.d_hidden, self.d_hidden),
                    nn.ReLU(),
                    nn.Linear(self.d_hidden, d_multi_modal)
                )
            else:
                raise ValueError(f"Unknown homogenizer type: {homogenizer_type}")
        self.use_text = use_text
        if use_text:
            self.bert_config = AutoConfig.from_pretrained(encoder_path)
            self.d_text_hidden = self.bert_config.hidden_size
            if homogenizer_type == "linear":
                self.homogenization_projection = nn.Linear(self.d_text_hidden, d_multi_modal)
            elif homogenizer_type == "mlp":
                self.homogenization_projection = nn.Sequential(
                    nn.Linear(self.d_text_hidden, self.d_text_hidden),
                    nn.ReLU(),
                    nn.Linear(self.d_text_hidden, d_multi_modal)
                )
            else:
                raise ValueError(f"Unknown homogenizer type: {homogenizer_type}")

    def describe_inputs(self, batch_size=1) -> InputSpec:
        input_dict = {
            "past_target": Input(
                shape=(batch_size, self.context_length), dtype=torch.float
            ),
            "past_observed_values": Input(
                shape=(batch_size, self.context_length), dtype=torch.float
            ),
            "feat_static_real": Input(
                shape=(batch_size, sum(self.d_feat_static_real)),
                dtype=torch.float,
            ),
            "item_index": Input(
                shape=(batch_size, 1),
                dtype=torch.long,
            ),
            "feat_static_cat": Input(
                shape=(batch_size, len(self.c_feat_static_cat)),
                dtype=torch.long,
            ),
            "feat_dynamic_real": Input(
                shape=(
                    batch_size,
                    self.context_length + self.prediction_length,
                    sum(self.d_feat_dynamic_real),
                ),
                dtype=torch.float,
            ),
            "observed_feat_dynamic_real": Input(
                shape=(
                    batch_size,
                    self.context_length + self.prediction_length,
                    sum(self.d_feat_dynamic_real),
                ),
                dtype=torch.float,
            ),
            "feat_dynamic_cat": Input(
                shape=(
                    batch_size,
                    self.context_length + self.prediction_length,
                    len(self.c_feat_dynamic_cat),
                ),
                dtype=torch.long,
            ),
            "observed_feat_dynamic_cat": Input(
                shape=(
                    batch_size,
                    self.context_length + self.prediction_length,
                    len(self.c_feat_dynamic_cat),
                ),
                dtype=torch.float,
            ),
            "past_feat_dynamic_real": Input(
                shape=(
                    batch_size,
                    self.context_length,
                    sum(self.d_past_feat_dynamic_real),
                ),
                dtype=torch.float,
            ),
            "past_observed_feat_dynamic_real": Input(
                shape=(
                    batch_size,
                    self.context_length,
                    sum(self.d_past_feat_dynamic_real),
                ),
                dtype=torch.float,
            ),
            "past_feat_dynamic_cat": Input(
                shape=(
                    batch_size,
                    self.context_length,
                    len(self.c_past_feat_dynamic_cat),
                ),
                dtype=torch.long,
            ),
            "past_observed_feat_dynamic_cat": Input(
                shape=(
                    batch_size,
                    self.context_length,
                    len(self.c_past_feat_dynamic_cat),
                ),
                dtype=torch.float,
            ),
        }
        if self.use_satellite:
            input_dict["satellite_data"] = Input(
                shape=(batch_size, self.context_length, 4, 64, 64),
                dtype=torch.float,
                required=False,
            )
        if self.use_text:
            input_dict["text_data"] = Input(
                shape=(batch_size, self.context_length, self.d_text_hidden),
                dtype=torch.float,
                required=False,
            )
        return InputSpec(
            input_dict,
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
            item_index: torch.Tensor,
            feat_static_real: Optional[torch.Tensor],  # [N, D_sr]
            feat_static_cat: Optional[torch.Tensor],  # [N, D_sc]
            feat_dynamic_real: Optional[torch.Tensor],  # [N, T + H, D_dr]
            feat_dynamic_cat: Optional[torch.Tensor] = None,  # [N, T + H, D_dc]
            past_feat_dynamic_real: Optional[torch.Tensor] = None,  # [N, T, D_pr]
            past_feat_dynamic_cat: Optional[torch.Tensor] = None,  # [N, T, D_pc]
            observed_feat_dynamic_real: Optional[torch.Tensor] = None,
            observed_feat_dynamic_cat: Optional[torch.Tensor] = None,
            past_observed_feat_dynamic_real: Optional[torch.Tensor] = None,
            past_observed_feat_dynamic_cat: Optional[torch.Tensor] = None,
            is_train=False,
            **input_kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        # past_target, loc, scale = self.scaler(past_target, weights=past_observed_values)
        future_covariates = []
        future_mask = []
        if self.with_static:
            static_covariates = []
            if self.feat_static_proj is not None:
                projs = self.feat_static_proj(feat_static_real)
                static_covariates.extend(projs)
            if self.feat_static_embed is not None:
                embs = self.feat_static_embed(feat_static_cat)
                static_covariates.extend(embs)
            static_var, weight = self.static_selector(static_covariates)  # [N, d_var]
            c_selection = self.selection(static_var).unsqueeze(1)  # [N, 1, d_var]
        else:
            c_selection = None
        # target_embedding, target_attention_mask = tokenize(self.inner_model, past_target, past_observed_values)
        inputs = {"context": past_target, "observed_values": past_observed_values}
        target_tokenize_output = self.model_wrapper.tokenize(inputs)

        # past_covariates = [target_embedding]
        past_covariates = [target_tokenize_output["token_embeddings"]]
        past_mask = [target_tokenize_output["attention_mask"].unsqueeze(
            -1)] if "attention_mask" in target_tokenize_output else []
        if self.use_satellite and "satellite_data" in input_kwargs:
            satellite_data = input_kwargs["satellite_data"]  # [B, T, H, W, C]
            # transpose the data
            satellite_data = satellite_data.permute(0, 4, 3, 2, 1)  # [B, T, H, W, C]
            B, H, W, C, T = satellite_data.shape

            sat_batch = rearrange(satellite_data, "b h w c t -> (b t) c h w")  # [B*T, C, H, W]
            sat_feat_batch = self.satellite_encoder(sat_batch)  # [B*T, d_model]
            satellite_feats = sat_feat_batch.reshape(B, T, -1)  # [B, T, d_model]

            satellite_feats = self.homogenization_projection(satellite_feats)  # [B, T, d_satellite]

            sat_len = min(satellite_feats.size(1), self.context_length)
            sat_feats_trimmed = satellite_feats[:, :sat_len]

            sat_mask = torch.ones(B, sat_len, self.d_multi_modal, device=satellite_data.device)
            # set to 0 according to past_observed_values
            sat_mask = sat_mask * past_observed_values[:, :sat_len].unsqueeze(-1)
            feat_dynamic_real_past, observed_feat_dynamic_real_past = (
                rearrange(sat_feats_trimmed, "b t c -> (b c) t"),
                rearrange(sat_mask, "b t c -> (b c) t")
            )
            sat_token_output = self.model_wrapper.tokenize(
                {"context": feat_dynamic_real_past,
                 "observed_values": observed_feat_dynamic_real_past}
            )

            past_covariates.extend(list(
                rearrange(sat_token_output["token_embeddings"], "(b c) t d -> c b t d", b=B, c=self.d_multi_modal)))
            if "attention_mask" in sat_token_output:
                past_mask.append(
                    rearrange(sat_token_output["attention_mask"], "(b c) t -> b t c", b=B, c=self.d_multi_modal))

        if self.use_text:
            text_data = input_kwargs["text_data"]
            B, T, L = text_data.shape
            # text_batch = rearrange(text_data, "b t l -> (b t) l")
            text_features = self.homogenization_projection(text_data)  # [B, T, d_satellite]

            sat_mask = torch.ones(B, T, self.d_multi_modal, device=text_features.device)
            sat_mask = sat_mask * past_observed_values.unsqueeze(-1)
            feat_dynamic_real_past, observed_feat_dynamic_real_past = (
                rearrange(text_features, "b t c -> (b c) t"),
                rearrange(sat_mask, "b t c -> (b c) t")
            )
            sat_token_output = self.model_wrapper.tokenize(
                {"context": feat_dynamic_real_past,
                 "observed_values": observed_feat_dynamic_real_past}
            )

            past_covariates.extend(list(
                rearrange(sat_token_output["token_embeddings"], "(b c) t d -> c b t d", b=B, c=self.d_multi_modal)))
            if "attention_mask" in sat_token_output:
                past_mask.append(
                    rearrange(sat_token_output["attention_mask"], "(b c) t -> b t c", b=B, c=self.d_multi_modal))

        if self.past_feat_dynamic_embed is not None:
            remap = self.past_feat_dynamic_embed(past_feat_dynamic_cat)
            past_feat_dynamic_real = torch.cat([past_feat_dynamic_real, remap], dim=-1)
            past_observed_feat_dynamic_real = torch.cat(
                [past_observed_feat_dynamic_real, past_observed_feat_dynamic_cat], dim=-1)

        if self.feat_dynamic_embed is not None:
            remap_feat_dynamic_cat = self.feat_dynamic_embed(feat_dynamic_cat)
            if feat_dynamic_real is not None:
                feat_dynamic_real = torch.cat([feat_dynamic_real, remap_feat_dynamic_cat], dim=-1)
            else:
                feat_dynamic_real = remap_feat_dynamic_cat
            if observed_feat_dynamic_real is not None:
                observed_feat_dynamic_real = torch.cat([observed_feat_dynamic_real, observed_feat_dynamic_cat],
                                                       dim=-1)
            else:
                observed_feat_dynamic_real = observed_feat_dynamic_cat
        if feat_dynamic_real is not None:
            b, t, c = feat_dynamic_real.size()
            feat_dynamic_real_past, observed_feat_dynamic_real_past = (
                rearrange(feat_dynamic_real[:, :self.context_length], "b t c -> (b c) t"),
                rearrange(observed_feat_dynamic_real[:, :self.context_length], "b t c -> (b c) t")
            )
            feat_dynamic_real_past_token_output = self.model_wrapper.tokenize(
                {"context": feat_dynamic_real_past, "observed_values": observed_feat_dynamic_real_past}
            )

            past_covariates.extend(list(
                rearrange(feat_dynamic_real_past_token_output["token_embeddings"], "(b c) t d -> c b t d", b=b, c=c)))
            if "attention_mask" in feat_dynamic_real_past_token_output:
                past_mask.append(
                    rearrange(feat_dynamic_real_past_token_output["attention_mask"], "(b c) t -> b t c", b=b, c=c))
            feat_dynamic_real_future, observed_feat_dynamic_real_future = (
                rearrange(feat_dynamic_real[:, -self.prediction_length:], "b t c -> (b c) t"),
                rearrange(observed_feat_dynamic_real[:, -self.prediction_length:], "b t c -> (b c) t")
            )
            feat_dynamic_real_future_token_output = self.model_wrapper.tokenize(
                {"context": feat_dynamic_real_future, "observed_values": observed_feat_dynamic_real_future}
            )
            future_covariates.extend(
                list(rearrange(feat_dynamic_real_future_token_output["token_embeddings"], "(b c) t d -> c b t d", b=b,
                               c=c))
            )
            if "attention_mask" in feat_dynamic_real_future_token_output:
                future_mask.append(
                    rearrange(feat_dynamic_real_future_token_output["attention_mask"],
                              "(b c) t -> b t c", b=b, c=c))
        if self.d_past_feat_dynamic_real:
            b, t, c = past_feat_dynamic_real.size()
            past_feat_dynamic_real_token_output = self.model_wrapper.tokenize(
                {"context": rearrange(
                    past_feat_dynamic_real,
                    "b t c -> (b c) t"),
                    "observed_values": rearrange(
                        past_observed_feat_dynamic_real,
                        "b t c -> (b c) t")
                }
            )

            past_covariates.extend(list(
                rearrange(past_feat_dynamic_real_token_output["token_embeddings"], "(b c) t d -> c b t d", b=b, c=c)))
            if "attention_mask" in past_feat_dynamic_real_token_output:
                past_mask.append(
                    rearrange(past_feat_dynamic_real_token_output["attention_mask"], "(b c) t -> b t c", b=b, c=c)
                )

        past_mask = torch.cat(past_mask, dim=-1).bool() if len(past_mask) > 0 else None
        if self.with_past:
            past_cov, past_weight = self.ctx_selector(
                past_covariates, c_selection, past_mask
            )  # [N, T, d_var]
            if self.with_gate:
                past_cov = self.cov_gate(past_cov)
        else:
            past_cov = 0

        pre_hidden = target_tokenize_output["token_embeddings"]
        if self.with_future and self.future_pos == "pre":
            future_cov = self.get_future_fused_token(c_selection, future_covariates, future_mask,
                                                     pre_hidden)
            pre_hidden = future_cov + pre_hidden

        if self.past_pos == "pre":
            pre_hidden = past_cov + pre_hidden
        target_tokenize_output["token_embeddings"] = pre_hidden
        encoded_output = self.model_wrapper.encode(target_tokenize_output)
        hidden_states = encoded_output["encoded_embeddings"]
        if self.with_future and self.future_pos == "post":
            future_cov = self.get_future_fused_token(c_selection, future_covariates, future_mask,
                                                     hidden_states)
            hidden_states = future_cov + hidden_states

        if self.past_pos == "post":
            hidden_states = past_cov + hidden_states

        encoded_output["encoded_embeddings"] = hidden_states
        decode_output = self.model_wrapper.predict(encoded_output)
        quantile_preds = decode_output["prediction"]
        decode_output["prediction"] = quantile_preds
        if is_train:
            return decode_output
        else:
            return self.model_wrapper.process_forecast(decode_output)

    def get_future_fused_token(self, c_selection, future_covariates, future_mask, hidden_states):
        future_mask = torch.cat(future_mask, dim=-1).bool() if len(future_mask) > 0 else None
        tgt_input, future_weight = self.tgt_selector(
            future_covariates, c_selection, future_mask
        )  # [N, H, d_var]
        # add past future indicator
        past_tokens = hidden_states + self.past_future_indicator[:, 0:1, :]
        future_tokens = tgt_input + self.past_future_indicator[:, 1:2, :]
        past_future_tokens = torch.cat([past_tokens,
                                        future_tokens], dim=1)
        fused_features = self.attention(past_future_tokens, past_future_tokens, past_future_tokens)[0]
        past_fused_features, future_fused_features = fused_features[:, :hidden_states.size(1)], fused_features[:,
                                                                                                hidden_states.size(
                                                                                                    1):]
        if self.future_with_gate:
            past_fused_features = self.future_cov_gate(past_fused_features)
        future_cov = past_fused_features
        return future_cov

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
