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

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.torch.modules.feature import (
    FeatureEmbedder as BaseFeatureEmbedder,
)
from torch.nn import Linear, Dropout, LayerNorm, MultiheadAttention


def fix_attention_mask(attn_mask):
    """
    Fixes attention mask to prevent NaN when a row is all True.
    When a row is all True (all tokens masked), make each token attend to itself
    to ensure there's always at least one valid attention weight per token.

    Args:
        attn_mask: Boolean attention mask where True means token is masked (invalid)
                  Shape: [batch_size, seq_len, seq_len]

    Returns:
        Fixed attention mask where diagonal elements are False (valid)
    """
    # Find rows that are completely masked (all True)
    all_masked_rows = attn_mask.all(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]

    # Create identity matrix of the same size as one attention slice
    seq_len = attn_mask.size(1)
    device = attn_mask.device
    identity = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0).expand_as(attn_mask)

    # Where rows are all masked, make diagonal False (valid) - attend to self
    fixed_mask = torch.where(
        all_masked_rows,  # condition: rows that are all True
        ~identity,  # if condition is met: attend to self (diagonal is False)
        attn_mask  # otherwise: use original mask
    )

    return fixed_mask


class FeatureEmbedder(BaseFeatureEmbedder):
    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        concat_features = super().forward(features=features)

        if self._num_features > 1:
            return torch.chunk(concat_features, self._num_features, dim=-1)
        else:
            return [concat_features]


class FeatureProjector(nn.Module):
    @validated()
    def __init__(
            self,
            feature_dims: List[int],
            embedding_dims: List[int],
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert len(feature_dims) > 0, "Expected len(feature_dims) > 1"
        assert len(feature_dims) == len(
            embedding_dims
        ), "Length of `feature_dims` and `embedding_dims` should match"
        assert all(
            c > 0 for c in feature_dims
        ), "Elements of `feature_dims` should be > 0"
        assert all(
            d > 0 for d in embedding_dims
        ), "Elements of `embedding_dims` should be > 0"

        self.feature_dims = feature_dims
        self._num_features = len(feature_dims)

        self._projectors = nn.ModuleList(
            [
                nn.Linear(out_features=d, in_features=c)
                for c, d in zip(feature_dims, embedding_dims)
            ]
        )

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """

        Parameters
        ----------
        features
            Numerical features with shape (..., sum(self.feature_dims)).

        Returns
        -------
        projected_features
            List of project features, with shapes
            [(..., self.embedding_dims[i]) for i in self.embedding_dims]
        """
        if self._num_features > 1:
            feature_slices = torch.split(features, self.feature_dims, dim=-1)
        else:
            feature_slices = tuple([features])

        return [
            proj(feat_slice)
            for proj, feat_slice in zip(self._projectors, feature_slices)
        ]


class GatedLinearUnit(nn.Module):
    @validated()
    def __init__(self, dim: int = -1, nonlinear: bool = True):
        super().__init__()
        self.dim = dim
        self.nonlinear = nonlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = torch.chunk(x, chunks=2, dim=self.dim)
        if self.nonlinear:
            value = torch.tanh(value)
        gate = torch.sigmoid(gate)
        return gate * value


class ParameterizedGatedLinearUnit(nn.Module):
    @validated()
    def __init__(self, d_hidden, dim: int = -1, nonlinear: bool = True):
        super().__init__()
        self.dim = dim
        self.nonlinear = nonlinear
        self.weight = nn.Parameter(torch.zeros((d_hidden,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = torch.chunk(x, chunks=2, dim=self.dim)
        if self.nonlinear:
            value = torch.tanh(value)
        gate = torch.sigmoid(gate)
        return gate * value * torch.tanh(self.weight)


class GatedResidualNetwork(nn.Module):
    @validated()
    def __init__(
            self,
            d_hidden: int,
            d_input: Optional[int] = None,
            d_output: Optional[int] = None,
            d_static: Optional[int] = None,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_input = d_input or d_hidden
        self.d_static = d_static or 0
        if d_output is None:
            self.d_output = self.d_input
            self.add_skip = False
        else:
            self.d_output = d_output
            if d_output != self.d_input:
                self.add_skip = True
                self.skip_proj = nn.Linear(
                    in_features=self.d_input,
                    out_features=self.d_output,
                )
            else:
                self.add_skip = False

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.d_input + self.d_static,
                out_features=self.d_hidden,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=self.d_hidden,
                out_features=self.d_hidden,
            ),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=self.d_hidden,
                out_features=self.d_output * 2,
            ),
            GatedLinearUnit(nonlinear=False),
        )
        self.layer_norm = nn.LayerNorm([self.d_output])

    def forward(
            self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.add_skip:
            skip = self.skip_proj(x)
        else:
            skip = x
        if self.d_static > 0 and c is None:
            raise ValueError("static variable is expected.")
        if self.d_static == 0 and c is not None:
            raise ValueError("static variable is not accepted.")
        if c is not None:
            x = torch.cat([x, c], dim=-1)
        x = self.mlp(x)
        x = self.layer_norm(x + skip)
        return x


class VariableSelectionNetwork(nn.Module):
    @validated()
    def __init__(
            self,
            d_hidden: int,
            num_vars: int,
            dropout: float = 0.0,
            add_static: bool = False,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.num_vars = num_vars
        self.add_static = add_static
        self.var_hidden = self.d_hidden // (2 ** (int(np.log2(num_vars))))
        self.proj = nn.Linear(self.d_hidden, self.var_hidden)

        self.weight_network = GatedResidualNetwork(
            d_hidden=self.d_hidden,
            d_input=self.var_hidden * self.num_vars,
            d_output=self.num_vars,
            d_static=self.d_hidden if add_static else None,
            dropout=dropout,
        )
        # self.variable_networks = nn.ModuleList(
        #     [
        #         GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout)
        #         for _ in range(num_vars)
        #     ]
        # )
        self.variable_networks = GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout)

    def forward(
            self,
            variables: List[torch.Tensor],
            static: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stack = torch.stack(variables, dim=-2)
        proj = self.proj(stack)
        flatten = proj.flatten(start_dim=-2)
        if static is not None:
            static = static.expand_as(variables[0])
        # flatten = self.proj(flatten)
        weight = self.weight_network(flatten, static)
        if mask is not None:
            # set the masked value to -inf
            weight = weight.masked_fill(~mask, -1e8)
        weight = torch.softmax(weight.unsqueeze(-2), dim=-1)

        # var_encodings = [
        #     net(var) for var, net in zip(variables, self.variable_networks)
        # ]
        # var_encodings = torch.stack(var_encodings, dim=-1)
        var_encodings = self.variable_networks(stack)
        var_encodings = torch.sum(var_encodings.transpose(-2, -1) * weight, dim=-1)

        return var_encodings, weight


class VariableAggregationNetwork(nn.Module):
    @validated()
    def __init__(
            self,
            d_hidden: int,
            num_vars: int,
            dropout: float = 0.0,
            add_static: bool = False,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.num_vars = num_vars
        self.add_static = add_static

        self.weight_network = GatedResidualNetwork(
            d_hidden=self.d_hidden,
            d_input=self.d_hidden,
            d_output=1,
            # d_static=self.d_hidden if add_static else None,
            dropout=dropout,
        )
        self.position_embedding = nn.Parameter(torch.zeros((1, 1, self.num_vars, self.d_hidden)))

    def forward(
            self,
            variables: List[torch.Tensor],
            static: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flatten = torch.stack(variables, dim=-2) + self.position_embedding
        if static is not None:
            flatten = flatten + static.unsqueeze(1)
        weight = self.weight_network(flatten).squeeze()
        if mask is not None:
            # set the masked value to -inf
            weight = weight.masked_fill(~mask, -1e8)
        weight = torch.softmax(weight.unsqueeze(-2), dim=-1)

        # var_encodings = [
        #     net(var) for var, net in zip(variables, self.variable_networks)
        # ]
        var_encodings = torch.stack(variables, dim=-1)

        var_encodings = torch.sum(var_encodings * weight, dim=-1)

        return var_encodings, weight


class ConditionalGlobalAttention(nn.Module):
    @validated()
    def __init__(
            self,
            d_hidden: int,
            num_vars: int,
            dropout: float = 0.0,
            add_static: bool = False,
            nhead: int = 1,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.num_vars = num_vars
        self.add_static = add_static

        # self.cross_attention = CrossAttentionLayer(
        #     d_hidden=d_hidden,
        #     nhead=nhead,
        #     dropout=dropout,
        #     dim_feedforward=self.d_hidden
        # )
        self.layer = nn.TransformerEncoderLayer(d_model=d_hidden, nhead=nhead, dim_feedforward=d_hidden,
                                                batch_first=True)
        self.position_embedding = nn.Parameter(torch.zeros((1, 1, self.num_vars, self.d_hidden)))
        # nn.TransformerDecoderLayer
        self.cls_token = nn.Parameter(torch.zeros((1, 1, 1, self.d_hidden)))
        # self.variable_networks = nn.ModuleList(
        #     [
        #         GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout)
        #         for _ in range(num_vars)
        #     ]
        # )

    def forward(
            self,
            variables: List[torch.Tensor],
            static: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_tokens = torch.stack(variables, dim=-2) + self.position_embedding
        if static is not None:
            temporal_tokens = temporal_tokens + static.unsqueeze(1)
        # concat cls tokens
        # if mask is not None:
        #     # set the masked value to -inf
        #     weight = weight.masked_fill(mask, -1e8)
        cls_token = self.cls_token.expand(temporal_tokens.size(0), temporal_tokens.size(1), -1, -1)
        temporal_tokens = torch.cat([cls_token, temporal_tokens], dim=-2)

        batch_size = mask.size(0)
        if mask is not None:
            #     # attn mask is the outer product of mask
            #     # mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            flat_mask = torch.cat([torch.ones((batch_size, mask.size(1), 1), device=mask.device), ~mask], dim=-1)
            attn_mask = (1 - flat_mask.unsqueeze(-1) * flat_mask.unsqueeze(
                -2)).bool()
        else:
            attn_mask = None
        # mask = torch.cat([torch.ones(batch_size, 1), mask], dim=-1)
        # mask = (1 - mask.unsqueeze(-1) * mask.unsqueeze(-2)).bool()

        # cls_token = self.cls_token.expand(temporal_tokens.size(0), temporal_tokens.size(1), -1, -1)

        # var_encodings, weight = self.cross_attention(cls_token.flatten(0, 1), temporal_tokens.flatten(0, 1),
        #                                              attn_mask=attn_mask.flatten(0, 1))
        var_encodings = self.layer(temporal_tokens.flatten(0, 1),
                                   src_mask=fix_attention_mask(attn_mask.flatten(0, 1)))[:, 0, :]
        # reshape the first two axis and keep the rest axis
        var_encodings = var_encodings.view(batch_size, -1, *var_encodings.shape[1:])
        # mask the encodings to zero if the mask is True
        # if mask is not None:
        #     var_encodings = var_encodings.masked_fill(torch.any(mask, dim=-1, keepdim=True), 0)

        return var_encodings, None


class ConditionalSelfAttention(nn.Module):
    @validated()
    def __init__(
            self,
            d_hidden: int,
            num_vars: int,
            dropout: float = 0.0,
            add_static: bool = False,
            nhead: int = 1,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.num_vars = num_vars
        self.add_static = add_static

        # self.cross_attention = CrossAttentionLayer(
        #     d_hidden=d_hidden,
        #     nhead=nhead,
        #     dropout=dropout,
        #     dim_feedforward=self.d_hidden
        # )
        self.layer = nn.TransformerEncoderLayer(d_model=d_hidden, nhead=nhead, dim_feedforward=d_hidden,
                                                batch_first=True)
        self.position_embedding = nn.Parameter(torch.zeros((1, 1, self.num_vars, self.d_hidden)))
        # nn.TransformerDecoderLayer
        self.cls_token = nn.Parameter(torch.zeros((1, 1, 1, self.d_hidden)))
        # self.variable_networks = nn.ModuleList(
        #     [
        #         GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout)
        #         for _ in range(num_vars)
        #     ]
        # )

    def forward(
            self,
            variables: List[torch.Tensor],
            static: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_tokens = torch.stack(variables, dim=-2) + self.position_embedding
        if static is not None:
            temporal_tokens = temporal_tokens + static.unsqueeze(1)
        # concat cls tokens
        # if mask is not None:
        #     # set the masked value to -inf
        #     weight = weight.masked_fill(mask, -1e8)
        cls_token = self.cls_token.expand(temporal_tokens.size(0), temporal_tokens.size(1), -1, -1)
        temporal_tokens = torch.cat([cls_token, temporal_tokens], dim=-2)

        batch_size = mask.size(0)
        if mask is not None:
            #     # attn mask is the outer product of mask
            #     # mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            flat_mask = torch.cat([torch.ones((batch_size, mask.size(1), 1), device=mask.device), mask], dim=-1)
            attn_mask = (1 - flat_mask.unsqueeze(-1) * flat_mask.unsqueeze(
                -2)).bool()
        else:
            attn_mask = None
        # mask = torch.cat([torch.ones(batch_size, 1), mask], dim=-1)
        # mask = (1 - mask.unsqueeze(-1) * mask.unsqueeze(-2)).bool()

        # cls_token = self.cls_token.expand(temporal_tokens.size(0), temporal_tokens.size(1), -1, -1)

        # var_encodings, weight = self.cross_attention(cls_token.flatten(0, 1), temporal_tokens.flatten(0, 1),
        #                                              attn_mask=attn_mask.flatten(0, 1))
        var_encodings = self.layer(temporal_tokens.flatten(0, 1),
                                   src_mask=fix_attention_mask(attn_mask.flatten(0, 1)))[:, 0, :]
        # reshape the first two axis and keep the rest axis
        var_encodings = var_encodings.view(batch_size, -1, *var_encodings.shape[1:])
        # mask the encodings to zero if the mask is True
        # if mask is not None:
        #     var_encodings = var_encodings.masked_fill(torch.any(mask, dim=-1, keepdim=True), 0)

        return var_encodings, None


class CrossAttentionLayer(nn.Module):
    @validated()
    def __init__(
            self,
            d_hidden: int,
            nhead: int = 1,
            dim_feedforward: int = 2048,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-5,
            batch_first: bool = True,
            norm_first: bool = False,
            bias: bool = True,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.d_hidden = d_hidden
        self.norm_first = norm_first

        # Cross attention mechanism
        self.cross_attn = MultiheadAttention(
            d_hidden, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs
        )

        # Feed-forward network
        self.linear1 = Linear(d_hidden, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_hidden, bias=bias, **factory_kwargs)

        # Layer normalization
        self.norm1 = LayerNorm(d_hidden, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_hidden, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        # Dropout layers
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Activation
        self.activation = nn.ReLU()

    def forward(
            self,
            query: torch.Tensor,
            key_value: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Cross-attention layer where query attends to key_value.

        Args:
            query: Query tensor (batch_size, query_len, d_hidden)
            key_value: Key/Value tensor (batch_size, kv_len, d_hidden)
            attn_mask: Attention mask (query_len, kv_len) or (batch_size, query_len, kv_len)
            key_padding_mask: Padding mask for keys (batch_size, kv_len)
            is_causal: Whether to apply causal mask

        Returns:
            Output tensor after cross-attention and feed-forward (batch_size, query_len, d_hidden)
        """
        x = query
        if self.norm_first:
            # Pre-normalization architecture (like in Transformer)
            out, attn = self._cross_attention_block(
                self.norm1(x), key_value, attn_mask, key_padding_mask, is_causal
            )
            x = x + out
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-normalization architecture (traditional)
            out, attn = self._cross_attention_block(
                x, key_value, attn_mask, key_padding_mask, is_causal
            )

            x = self.norm1(x + out)
            x = self.norm2(x + self._ff_block(x))
        return x, attn

    def _cross_attention_block(
            self,
            x: torch.Tensor,
            mem: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            key_padding_mask: Optional[torch.Tensor],
            is_causal: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Cross-attention block"""
        x = self.cross_attn(
            query=x,
            key=mem,
            value=mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False
        )
        return self.dropout1(x[0]), x[1]

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward block"""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TemporalFusionEncoder(nn.Module):
    @validated()
    def __init__(
            self,
            d_input: int,
            d_hidden: int,
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=d_input, hidden_size=d_hidden, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=d_input, hidden_size=d_hidden, batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        if d_input != d_hidden:
            self.skip_proj = nn.Linear(
                in_features=d_input, out_features=d_hidden
            )
            self.add_skip = True
        else:
            self.add_skip = False

        self.lnorm = nn.LayerNorm(d_hidden)

    def forward(
            self,
            ctx_input: torch.Tensor,
            tgt_input: Optional[torch.Tensor] = None,
            states: Optional[List[torch.Tensor]] = None,
    ):
        ctx_encodings, states = self.encoder_lstm(ctx_input, states)

        if tgt_input is not None:
            tgt_encodings, _ = self.decoder_lstm(tgt_input, states)
            encodings = torch.cat((ctx_encodings, tgt_encodings), dim=1)
            skip = torch.cat((ctx_input, tgt_input), dim=1)
        else:
            encodings = ctx_encodings
            skip = ctx_input

        if self.add_skip:
            skip = self.skip_proj(skip)
        encodings = self.gate(encodings)
        encodings = self.lnorm(skip + encodings)
        return encodings


class TemporalFusionDecoder(nn.Module):
    @validated()
    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            d_hidden: int,
            d_var: int,
            num_heads: int,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.enrich = GatedResidualNetwork(
            d_hidden=d_hidden,
            d_static=d_var,
            dropout=dropout,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.att_net = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        self.att_lnorm = nn.LayerNorm(d_hidden)

        self.ff_net = nn.Sequential(
            GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout),
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        self.ff_lnorm = nn.LayerNorm(d_hidden)

    def forward(
            self,
            x: torch.Tensor,
            static: torch.Tensor,
            mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_static = static.repeat(
            (1, self.context_length + self.prediction_length, 1)
        )

        skip = x[:, self.context_length:, ...]
        x = self.enrich(x, expanded_static)

        mask_pad = torch.ones_like(mask)[:, 0:1, ...]
        mask_pad = mask_pad.repeat((1, self.prediction_length))
        key_padding_mask = (1.0 - torch.cat((mask, mask_pad), dim=1)).bool()

        query_key_value = x
        attn_output, _ = self.attention(
            query=query_key_value[:, self.context_length:, ...],
            key=query_key_value,
            value=query_key_value,
            key_padding_mask=key_padding_mask,
        )
        att = self.att_net(attn_output)

        x = x[:, self.context_length:, ...]
        x = self.att_lnorm(x + att)
        x = self.ff_net(x)
        x = self.ff_lnorm(x + skip)

        return x
