# SPDX-License-Identifier: Apache-2.0
"""Gated full attention for Qwen3.8 text layers.

Head dim is 256, so this does not call the NKI attention kernels (those cap
head dim at 128). The math matches ``Qwen3_5Attention`` in Transformers:
per-head query and sigmoid gate from ``q_proj``, zero-centered Q/K RMSNorm,
partial RoPE, then softmax attention scaled by ``1/sqrt(head_dim)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def partial_rope_cos_sin(
    positions: torch.Tensor,
    rope_theta: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin for text positions. Shape [S, rotary_dim].

    Qwen3.8's mRoPE sections are applied to three identical grids for text, so
    the interleave is a no-op and this is partial RoPE on ``rotary_dim`` dims.
    """
    half = rotary_dim // 2
    inv = 1.0 / (
        rope_theta
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float, device=positions.device) / rotary_dim)
    )
    freqs = positions.float().unsqueeze(-1) * inv[:half]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_partial_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k are [B, H, S, D]. cos, sin are [S, rotary_dim]."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotary = cos.shape[-1]

    def _rotate(x):
        rot, rest = x[..., :rotary], x[..., rotary:]
        x1 = rot[..., : rotary // 2]
        x2 = rot[..., rotary // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return torch.cat((rot * cos + rotated * sin, rest), dim=-1)

    return _rotate(q), _rotate(k)


def write_paged_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Write [T, Kv, D] into caches shaped [blocks, Kv, block_size, D].

    Negative slots are left untouched. The write is a tensor scatter so it stays
    inside a traced graph.
    """
    slots = slot_mapping.to(dtype=torch.long)
    valid = slots >= 0
    safe = torch.where(valid, slots, torch.zeros_like(slots))
    block = torch.div(safe, block_size, rounding_mode="floor")
    pos = torch.remainder(safe, block_size)
    written_k = k_cache[block, :, pos, :]
    written_v = v_cache[block, :, pos, :]
    mask = valid.view(-1, 1, 1)
    k_cache[block, :, pos, :] = torch.where(mask, k, written_k)
    v_cache[block, :, pos, :] = torch.where(mask, v, written_v)


def read_paged_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    n_tokens: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the first n_tokens of one request. block_table is [n_blocks]."""
    index = torch.arange(n_tokens, device=k_cache.device)
    block = block_table[torch.div(index, block_size, rounding_mode="floor")]
    pos = torch.remainder(index, block_size)
    return k_cache[block, :, pos, :], v_cache[block, :, pos, :]


def gated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    n_past: int,
) -> torch.Tensor:
    """Causal attention. q is [H, S, D], k and v are [Hkv, T, D], T = n_past + S.

    gate is [S, H*D] and is applied after the heads are merged.
    """
    heads, seq, dim = q.shape
    kv_heads = k.shape[0]
    if heads % kv_heads != 0:
        raise ValueError(f"{heads} query heads are not divisible by {kv_heads} KV heads")
    groups = heads // kv_heads
    if groups != 1:
        k = k.repeat_interleave(groups, dim=0)
        v = v.repeat_interleave(groups, dim=0)
    scale = dim ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    key_index = torch.arange(n_past + seq, device=q.device)
    query_index = n_past + torch.arange(seq, device=q.device)
    causal = key_index.unsqueeze(0) <= query_index.unsqueeze(1)
    scores = scores.masked_fill(~causal.unsqueeze(0), torch.finfo(torch.float32).min)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float()).to(q.dtype)
    merged = out.transpose(0, 1).reshape(seq, heads * dim)
    return merged * torch.sigmoid(gate)


class Qwen38Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        rope_theta: float,
        eps: float,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        if num_heads % tp_size or num_kv_heads % tp_size:
            raise ValueError(
                f"TP={tp_size} does not divide heads {num_heads}/{num_kv_heads}"
            )
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.eps = eps
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        q_out = self.num_heads * head_dim * 2
        kv_out = self.num_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(self.num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.Parameter(torch.zeros(head_dim))
        self.k_norm = nn.Parameter(torch.zeros(head_dim))

    def _norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + weight.float())).type_as(x)

    def project(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """hidden [S, H] -> q [Hq, S, D], k, v [Hkv, S, D], gate [S, Hq*D]."""
        seq = hidden.shape[0]
        qg = self.q_proj(hidden).view(seq, self.num_heads, self.head_dim * 2)
        query, gate = qg.split(self.head_dim, dim=-1)
        query = self._norm(query, self.q_norm)
        key = self._norm(
            self.k_proj(hidden).view(seq, self.num_kv_heads, self.head_dim), self.k_norm
        )
        value = self.v_proj(hidden).view(seq, self.num_kv_heads, self.head_dim)
        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)
        return q, k, v, gate.reshape(seq, -1)

    def forward_span(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
        n_past: int,
    ) -> torch.Tensor:
        """One request. hidden [S, H]. Caches are paged [blocks, Kv, block, D]."""
        q, k, v, gate = self.project(hidden)
        cos, sin = partial_rope_cos_sin(positions, self.rope_theta, self.rotary_dim)
        q, k = apply_partial_rope(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q, k = q.squeeze(0), k.squeeze(0)
        write_paged_kv(
            k.transpose(0, 1), v.transpose(0, 1), k_cache, v_cache, slot_mapping, block_size
        )
        n_tokens = n_past + hidden.shape[0]
        k_all, v_all = read_paged_kv(k_cache, v_cache, block_table, n_tokens, block_size)
        out = gated_attention(
            q, k_all.transpose(0, 1), v_all.transpose(0, 1), gate, n_past
        )
        return self.o_proj(out)
