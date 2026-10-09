"""Minimal PatchTST primitives used by the isolated WorldTS predictor.

These definitions preserve the implementation used in VoT while avoiding a
dependency on VoT's top-level ``layers`` package.
"""

from math import log, sqrt

import torch
import torch.nn.functional as F
from torch import nn

from .deterministic_padding import deterministic_replicate_pad_1d


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.padding = int(padding)
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        n_vars = x.shape[1]
        x = deterministic_replicate_pad_1d(x, right=self.padding)
        x = x.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale=None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        del factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask,
        tau=None,
        delta=None,
    ):
        del tau, delta
        batch_size, query_length, _, embedding_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(embedding_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                mask = torch.triu(
                    torch.ones(
                        query_length,
                        keys.shape[1],
                        dtype=torch.bool,
                        device=queries.device,
                    ),
                    diagonal=1,
                )
                mask = mask.view(1, 1, query_length, keys.shape[1])
            else:
                mask = getattr(attn_mask, "mask", attn_mask)
            scores = scores.masked_fill(mask, -torch.inf)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attention, values)
        return output.contiguous(), attention if self.output_attention else None


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys=None,
        d_values=None,
    ):
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask,
        tau=None,
        delta=None,
    ):
        batch_size, query_length, _ = queries.shape
        key_length = keys.shape[1]
        queries = self.query_projection(queries).view(
            batch_size, query_length, self.n_heads, -1
        )
        keys = self.key_projection(keys).view(
            batch_size, key_length, self.n_heads, -1
        )
        values = self.value_projection(values).view(
            batch_size, key_length, self.n_heads, -1
        )
        output, attention = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
        )
        output = output.reshape(batch_size, query_length, -1)
        return self.out_projection(output), attention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff=None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attention = self.attention(
            x,
            x,
            x,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attention


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attentions = []
        for attention_layer in self.attn_layers:
            x, attention = attention_layer(
                x,
                attn_mask=attn_mask,
                tau=tau,
                delta=delta,
            )
            attentions.append(attention)
        if self.norm is not None:
            x = self.norm(x)
        return x, attentions
