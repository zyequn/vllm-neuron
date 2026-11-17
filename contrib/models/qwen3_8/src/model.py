# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B text decoder for the vLLM-Neuron runner.

Prefix caching is off. Full-attention layers use the paged KV cache. Gated
DeltaNet layers use ``DeltaSlotTable``: one fixed state per live request,
sharded on value heads. The runner identifies a request by its first KV block
id. A request that leaves the batch releases its slot.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig

from .attention import Qwen38Attention
from .config import Qwen38TextConfig
from .gated_delta import Qwen38GatedDeltaNet, ZeroCenteredRMSNorm
from .slots import DeltaSlotTable
from .weights import load_text_weights


def _first_block_id(block_row: torch.Tensor) -> int:
    for value in block_row.tolist():
        block = int(value)
        if block > 0:
            return block
    raise ValueError("request block table has no allocated block")


class Qwen38MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, tp_size: int, tp_rank: int):
        super().__init__()
        if intermediate % tp_size:
            raise ValueError(f"TP={tp_size} does not divide intermediate {intermediate}")
        shard = intermediate // tp_size
        self.gate_proj = nn.Linear(hidden, shard, bias=False)
        self.up_proj = nn.Linear(hidden, shard, bias=False)
        self.down_proj = nn.Linear(shard, hidden, bias=False)
        self.tp_rank = tp_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen38DecoderLayer(nn.Module):
    def __init__(self, config: Qwen38TextConfig, layer_idx: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = config.is_linear_layer(layer_idx)
        self.input_layernorm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        if self.is_linear:
            heads_k = config.linear_num_key_heads // tp_size
            heads_v = config.linear_num_value_heads // tp_size
            self.mixer = Qwen38GatedDeltaNet(
                config.hidden_size,
                heads_k,
                heads_v,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                config.linear_conv_kernel_dim,
                config.rms_norm_eps,
            )
            self.attn = None
        else:
            self.mixer = None
            self.attn = Qwen38Attention(
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rotary_dim,
                config.rope_theta,
                config.rms_norm_eps,
                tp_size,
                tp_rank,
            )
        self.mlp = Qwen38MLP(
            config.hidden_size, config.intermediate_size, tp_size, tp_rank
        )

    def forward_linear(
        self,
        hidden: torch.Tensor,
        conv_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
    ):
        normed = self.input_layernorm(hidden)
        mixed, new_conv, new_state = self.mixer(
            normed.unsqueeze(0), conv_state, recurrent_state
        )
        hidden = hidden + mixed.squeeze(0)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, new_conv, new_state

    def forward_attn(self, hidden, positions, k_cache, v_cache, block_table, slots, block_size, n_past):
        normed = self.input_layernorm(hidden)
        mixed = self.attn.forward_span(
            normed, positions, k_cache, v_cache, block_table, slots, block_size, n_past
        )
        hidden = hidden + mixed
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class Qwen38ForCausalLM(nn.Module):
    """Text-only model. Vision and MTP weights are not loaded."""

    prefix_caching = False

    def __init__(
        self,
        config: Qwen38TextConfig,
        max_num_seqs: int = 4,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen38DecoderLayer(config, i, tp_size, tp_rank)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        linear_layers = [layer for layer in self.layers if layer.is_linear]
        spec = linear_layers[0].mixer if linear_layers else None
        self.delta_slots = DeltaSlotTable(
            num_slots=max_num_seqs,
            num_layers=len(linear_layers),
            num_v_heads=spec.num_v_heads if spec else 1,
            head_k_dim=spec.head_k_dim if spec else 1,
            head_v_dim=spec.head_v_dim if spec else 1,
            conv_dim=spec.conv_dim if spec else 1,
            conv_tail=(spec.conv_kernel_size - 1) if spec else 1,
        )
        self._linear_index = {}
        linear_row = 0
        for layer in self.layers:
            if layer.is_linear:
                self._linear_index[layer.layer_idx] = linear_row
                linear_row += 1
        self._attn_names = [
            f"layers.{layer.layer_idx}.self_attn"
            for layer in self.layers
            if not layer.is_linear
        ]
        self.block_size = 16
        self._blocks_per_seq = 32

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict,
        neuron_config: NeuronConfig | None = None,
        max_num_seqs: int = 4,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> Qwen38ForCausalLM:
        config = Qwen38TextConfig.from_configs(hf_config, neuron_config)
        seqs = max_num_seqs
        if neuron_config is not None and getattr(neuron_config, "max_num_seqs", None):
            seqs = int(neuron_config.max_num_seqs)
        return cls(config, max_num_seqs=seqs, tp_size=tp_size, tp_rank=tp_rank)

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.delta_slots.conv = self.delta_slots.conv.to(device=device)
        self.delta_slots.recurrent = self.delta_slots.recurrent.to(device=device, dtype=torch.float32)
        if hasattr(self, "_private_kv"):
            self._private_kv = {
                idx: (k.to(device), v.to(device)) for idx, (k, v) in self._private_kv.items()
            }
        return module

    def get_kv_spec(self) -> KVSpec:
        attn = next(layer.attn for layer in self.layers if layer.attn is not None)
        return KVSpec(
            layers=[
                LayerSpec(
                    name=name,
                    num_kv_heads=attn.num_kv_heads,
                    head_size=attn.head_dim,
                    dtype=torch.bfloat16,
                )
                for name in self._attn_names
            ]
        )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        self._kv = kv_caches

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        """Runner entry. ``cache_dir`` is accepted because the Neuron worker passes it."""
        del cache_dir
        load_text_weights(self, checkpoint_path)
        self.to(device)

    def _groups(self, n_tokens: int, positions: torch.Tensor, attn_metadata, req_ids, context_len):
        """Split a flat token batch into (start, end, key, reset) groups.

        ``context_len`` is the python length of the sequence, including tokens
        already consumed. Passing it avoids reading a position value back to
        the host, which would cut an XLA graph before the KV write.
        """
        if req_ids is not None:
            if len(req_ids) == n_tokens:
                spans = [(i, i + 1, req_ids[i], int(positions[i]) == 0) for i in range(n_tokens)]
                return spans
            if len(req_ids) == 1:
                starts = context_len == n_tokens if context_len is not None else int(positions[0]) == 0
                return [(0, n_tokens, req_ids[0], starts)]
            if n_tokens % len(req_ids):
                raise ValueError(
                    f"{n_tokens} tokens do not divide across {len(req_ids)} requests"
                )
            width = n_tokens // len(req_ids)
            return [
                (i * width, (i + 1) * width, req_ids[i], int(positions[i * width]) == 0)
                for i in range(len(req_ids))
            ]
        meta = self._layer_meta(attn_metadata)
        block_table = meta["block_table_tensor"]
        n_req = block_table.shape[0]
        if n_req == 1:
            key = _first_block_id(block_table[0])
            starts = context_len == n_tokens if context_len is not None else int(positions[0]) == 0
            return [(0, n_tokens, key, starts)]
        if n_tokens == n_req:
            return [
                (i, i + 1, _first_block_id(block_table[i]), int(positions[i]) == 0)
                for i in range(n_req)
            ]
        if n_tokens % n_req:
            raise ValueError(
                f"cannot split {n_tokens} tokens across {n_req} block-table rows"
            )
        width = n_tokens // n_req
        return [
            (
                i * width,
                (i + 1) * width,
                _first_block_id(block_table[i]),
                int(positions[i * width]) == 0,
            )
            for i in range(n_req)
        ]

    def _layer_meta(self, attn_metadata) -> dict:
        if not attn_metadata:
            raise ValueError("forward requires req_ids or attn_metadata")
        name = self._attn_names[0]
        if name in attn_metadata:
            return attn_metadata[name]
        return next(iter(attn_metadata.values()))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        req_ids: list | None = None,
        context_len: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("input_ids must be a flat token list [T]")
        positions = positions.reshape(-1)
        groups = self._groups(
            input_ids.shape[0], positions, attn_metadata, req_ids, context_len
        )
        self.delta_slots.retain_only({key for _, _, key, _ in groups})
        hidden = self.embed_tokens(input_ids)
        outputs = []
        for start, end, key, reset in groups:
            outputs.append(
                self._forward_span(
                    hidden[start:end],
                    positions[start:end],
                    key,
                    reset,
                    attn_metadata,
                    context_len,
                )
            )
        hidden = torch.cat(outputs, dim=0)
        logits = self.lm_head(self.norm(hidden))
        if sampling_positions is not None:
            logits = torch.index_select(logits, 0, sampling_positions)
        return logits

    def _forward_span(self, hidden, positions, key, reset, attn_metadata, context_len):
        slot = self.delta_slots.ensure(key, reset=reset)
        if reset:
            self._zero_private_kv(slot)
        if context_len is None:
            n_past = int(positions[0])
            n_total = int(positions[-1]) + 1
        else:
            n_total = int(context_len)
            n_past = n_total - positions.shape[0]
        meta = self._layer_meta(attn_metadata) if attn_metadata else None
        for layer in self.layers:
            if layer.is_linear:
                index = self._linear_index[layer.layer_idx]
                conv = None if reset and n_past == 0 else self.delta_slots.conv[index, slot]
                state = None if reset and n_past == 0 else self.delta_slots.recurrent[index, slot]
                if conv is not None:
                    conv = conv.unsqueeze(0)
                    state = state.unsqueeze(0)
                hidden, new_conv, new_state = layer.forward_linear(hidden, conv, state)
                self.delta_slots.conv[index, slot] = new_conv.squeeze(0)
                self.delta_slots.recurrent[index, slot] = new_state.squeeze(0)
            else:
                k_cache, v_cache, block_table, slots, block_size = self._attn_view(
                    layer, meta, key, positions, n_total
                )
                hidden = layer.forward_attn(
                    hidden, positions, k_cache, v_cache, block_table, slots, block_size, n_past
                )
        return hidden

    def _attn_view(self, layer, meta, key, positions, n_total):
        name = f"layers.{layer.layer_idx}.self_attn"
        if meta is not None and hasattr(self, "_kv") and name in self._kv:
            block_table = meta["block_table_tensor"]
            row = 0
            for i in range(block_table.shape[0]):
                if _first_block_id(block_table[i]) == key:
                    row = i
                    break
            n = positions.shape[0]
            slot_mapping = meta["slot_mapping"]
            if slot_mapping.shape[0] == block_table.shape[0]:
                slots = slot_mapping[row : row + 1]
            else:
                width = slot_mapping.shape[0] // block_table.shape[0]
                slots = slot_mapping[row * width : row * width + n]
            return (
                self._kv[name][0],
                self._kv[name][1],
                block_table[row],
                slots,
                int(meta["block_size"]),
            )
        return self._private_cache(layer, key, positions, n_total)

    def _zero_private_kv(self, slot: int) -> None:
        if not hasattr(self, "_private_kv"):
            return
        per = self._blocks_per_seq
        start = slot * per
        for k_cache, v_cache in self._private_kv.values():
            k_cache[start : start + per].zero_()
            v_cache[start : start + per].zero_()

    def _private_cache(self, layer, key, positions, n_total):
        """Paged cache used when the runner has not bound one. One block range per slot."""
        slot = self.delta_slots._index[key]
        attn = layer.attn
        block = self.block_size
        per = self._blocks_per_seq
        if not hasattr(self, "_private_kv"):
            self._private_kv = {}
        if layer.layer_idx not in self._private_kv:
            shape = (
                self.delta_slots.num_slots * per,
                attn.num_kv_heads,
                block,
                attn.head_dim,
            )
            device = next(self.parameters()).device
            self._private_kv[layer.layer_idx] = (
                torch.zeros(shape, dtype=torch.float32, device=device),
                torch.zeros(shape, dtype=torch.float32, device=device),
            )
        n_needed = int(n_total)
        n_table = (n_needed + block - 1) // block
        if n_table > per:
            raise ValueError(
                f"sequence length {n_needed} exceeds the private cache ({per * block} tokens)"
            )
        base = slot * per
        k_cache, v_cache = self._private_kv[layer.layer_idx]
        device = k_cache.device
        table = torch.arange(base, base + n_table, device=device)
        offsets = positions.to(device=device, dtype=torch.long)
        slots = (base * block + offsets).to(dtype=torch.int32)
        return k_cache, v_cache, table, slots, block
