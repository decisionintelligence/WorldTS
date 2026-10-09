from typing import Union, List, Tuple

import torch
from einops import rearrange, repeat, einsum
from torch import nn
import torch.nn.functional as F


def rotate_every_two(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class CrossPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_src = nn.LayerNorm(dim)
        self.norm_tgt = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, ctx, src_pos_emb, ts, tgt_pos_emb):
        return self.fn(self.norm_src(ctx), src_pos_emb, self.norm_tgt(ts), tgt_pos_emb)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, use_glu=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.0,
            use_rotary=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, pos_emb):
        """
        Args:
            x: Sequence of shape [B, N, D]
            pos_emb: Positional embedding of sequence's tokens of shape [B, N, D]
        """

        q = self.to_q(x)

        qkv = (q, *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), qkv
        )

        if self.use_rotary:
            sin, cos = map(
                lambda t: repeat(t, "b n d -> (b h) n d", h=self.heads), pos_emb
            )
            dim_rotary = sin.shape[-1]

            # handle the case where rotary dimension < head dimension

            (q, q_pass), (k, k_pass) = map(
                lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k)
            )
            q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
            q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))

        dots = einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out), attn


class VisionTransformer(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            dim_head: int,
            mlp_dim: int,
            image_size: Union[List[int], Tuple[int], int],
            dropout: float = 0.0,
            use_rotary: bool = True,
            use_glu: bool = True,
    ):
        super().__init__()
        self.image_size = image_size

        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SelfAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_rotary=use_rotary,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(
            self,
            src: torch.Tensor,
            src_pos_emb: torch.Tensor,
    ):
        """
        Performs the following computation in each layer:
            1. Self-Attention on the source sequence
            2. FFN on the source sequence
        Args:
            src: Source sequence of shape [B, N, D]
            src_pos_emb: Positional embedding of source sequence's tokens of shape [B, N, D]
        """

        attention_scores = {}
        for i in range(len(self.blocks)):
            sattn, sff = self.blocks[i]

            out, sattn_scores = sattn(src, pos_emb=src_pos_emb)
            attention_scores["self_attention"] = sattn_scores
            src = out + src
            src = sff(src) + src

        return src, attention_scores
