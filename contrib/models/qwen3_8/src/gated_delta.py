# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet for Qwen3.5 / Qwen3.8 text layers.

Transcribed from the Transformers eager reference
(``torch_recurrent_gated_delta_rule`` and ``Qwen3_5GatedDeltaNet``). The
chunked kernel is not used: prefill and decode share this token loop so the
AOT graph has one recurrence. Query and key are L2-normalized, then the query
is scaled by ``1/sqrt(head_k_dim)``. The recurrent state is float32.

Qwen3.5 splits the fused Qwen3-Next projections into ``in_proj_qkv``,
``in_proj_z``, ``in_proj_b``, and ``in_proj_a``. Norm weights are zero-centered:
the scale is ``(1 + weight)``, and ``weight`` is initialized at 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .static_fp8 import linear as _linear


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One token at a time. Shapes match the Transformers eager reference.

    query, key:   [B, S, num_k_heads, head_k_dim]
    value:        [B, S, num_v_heads, head_v_dim]
    g, beta:      [B, S, num_v_heads]   (g is log-space decay, entries <= 0)
    initial_state:[B, num_v_heads, head_k_dim, head_v_dim]
    returns out:  [B, S, num_v_heads, head_v_dim]
    """
    initial_dtype = query.dtype
    batch, seq_len, num_k_heads, k_dim = key.shape
    num_v_heads, v_dim = value.shape[-2:]
    if num_k_heads != num_v_heads:
        repeat = num_v_heads // num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    query, key, value, beta, decay = [
        t.transpose(1, 2).to(torch.float32).contiguous()
        for t in (query, key, value, beta, g)
    ]
    if use_qk_l2norm:
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
    query = query / (query.shape[-1] ** 0.5)

    if initial_state is None:
        state = torch.zeros(
            batch, num_v_heads, k_dim, v_dim, dtype=value.dtype, device=value.device
        )
    else:
        state = initial_state.to(dtype=value.dtype)

    if query.device.type not in ("cpu", "meta") and k_dim == 128 and v_dim == 128:
        out, state = _recurrence_on_neuron(query, key, value, decay, beta, state)
    else:
        out = torch.zeros_like(value)
        for i in range(seq_len):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            state = state * decay[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    if not output_final_state:
        return out, None
    return out, state


def _recurrence_on_neuron(query, key, value, decay, beta, state):
    """Run the token loop as one NKI call per batch row.

    ``query`` and ``key`` are ``[B, heads, S, K]``. ``value`` is
    ``[B, heads, S, V]``. ``decay`` is log-space ``g``. ``state`` is
    ``[B, heads, K, V]``. The returned ``out`` matches ``value``'s layout.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .gated_delta_nki import gated_delta_recurrence

    kernel = wrap_nki(gated_delta_recurrence)
    batch, _heads, seq, _k_dim = query.shape
    decay = decay.exp()
    outs = []
    states = []
    for row in range(batch):
        length = torch.tensor([[seq]], dtype=torch.int32, device=query.device)
        out_row, state_row = kernel(
            query[row].transpose(1, 2).contiguous(),
            key[row].transpose(1, 2).contiguous(),
            value[row].transpose(1, 2).contiguous(),
            decay[row].contiguous(),
            beta[row].contiguous(),
            state[row].contiguous(),
            length,
        )
        outs.append(out_row.transpose(1, 2))
        states.append(state_row)
    return torch.stack(outs, dim=0), torch.stack(states, dim=0)


def causal_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Depthwise causal conv. hidden_states and weight are [B, C, T] and [C, K]."""
    channels, _kernel = weight.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight.unsqueeze(1),
        bias,
        padding=padding,
        groups=channels,
    )[:, :, : hidden_states.shape[-1]]
    if activation == "silu":
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"unsupported conv activation {activation}")
    return out.to(hidden_states.dtype)


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode step. conv_state is [B, C, K-1]. Returns (out, new_state)."""
    channels = weight.shape[0]
    seq_len = hidden_states.shape[-1]
    window = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    new_state = window[:, :, -conv_state.shape[-1] :].to(conv_state.dtype)
    out = F.conv1d(window, weight.unsqueeze(1), bias, padding=0, groups=channels)
    out = out[:, :, -seq_len:]
    if activation == "silu":
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"unsupported conv activation {activation}")
    return out.to(hidden_states.dtype), new_state


class ZeroCenteredRMSNorm(nn.Module):
    """Qwen3-Next / Qwen3.5 RMSNorm. Weight is a residual on 1, init at 0."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """RMSNorm, then SiLU gate. Used on the DeltaNet value output."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        y = hidden_states.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = self.weight * y.to(dtype)
        y = y * F.silu(gate.float())
        return y.to(dtype)


class Qwen38GatedDeltaNet(nn.Module):
    """One Qwen3.5 linear-attention layer, without the HF Cache object.

    State is passed in and returned. ``conv_state`` is ``[B, conv_dim, K-1]``.
    ``recurrent_state`` is ``[B, num_v_heads, head_k, head_v]`` float32.
    A missing state means the sequence starts at zero (prefill).
    """

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(num_v_heads).uniform_(0.01, 16)))
        self.norm = GatedRMSNorm(head_v_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """hidden_states is [B, S, H]. Returns (out, new_conv_state, new_recurrent_state)."""
        batch, seq_len, _ = hidden_states.shape
        projected = _linear(self.in_proj_qkv, hidden_states).transpose(1, 2)
        z = _linear(self.in_proj_z, hidden_states).view(batch, seq_len, -1, self.head_v_dim)
        b = _linear(self.in_proj_b, hidden_states)
        a = _linear(self.in_proj_a, hidden_states)
        weight = self.conv1d.weight.squeeze(1)
        tail = self.conv_kernel_size - 1

        if conv_state is not None and seq_len == 1:
            mixed, new_conv = causal_conv1d_update(
                projected, conv_state, weight, activation="silu"
            )
        elif conv_state is not None:
            window = torch.cat([conv_state, projected], dim=-1)
            mixed = F.conv1d(
                window.to(weight.dtype),
                weight.unsqueeze(1),
                padding=0,
                groups=self.conv_dim,
            )[:, :, -seq_len:]
            mixed = F.silu(mixed).to(projected.dtype)
            new_conv = window[:, :, -tail:]
        else:
            mixed = causal_conv1d(projected, weight, activation="silu")
            if seq_len >= tail:
                new_conv = projected[:, :, -tail:]
            else:
                new_conv = F.pad(projected, (tail - seq_len, 0))[:, :, -tail:]

        mixed = mixed.transpose(1, 2)
        query, key, value = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.view(batch, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        core, new_state = recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm=True,
        )
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        core = core.view(batch, seq_len, -1)
        return _linear(self.out_proj, core), new_conv, new_state
