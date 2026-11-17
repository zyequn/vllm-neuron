# SPDX-License-Identifier: Apache-2.0
"""Config adapter for Qwen3.8-27B (``Qwen3_5ForConditionalGeneration``).

Hugging Face nests the language-model fields under ``text_config``. This class
flattens that dict. Vision fields are ignored: this package serves text only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen38TextConfig:
    """Language-model geometry from the Qwen3.8-27B checkpoint."""

    vocab_size: int = 248320
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    attn_output_gate: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10_000_000.0

    # Gated DeltaNet. 16 QK heads, 48 V heads, both head dims 128, conv width 4.
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # 3 linear_attention layers, then 1 full_attention, repeated.
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4

    neuron_config: NeuronConfig | None = None

    @classmethod
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None = None):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
        elif isinstance(hf_config, dict):
            config_dict = dict(hf_config)
        else:
            raise TypeError(f"unsupported config type {type(hf_config).__name__}")

        text = config_dict.get("text_config") or {}
        if not isinstance(text, dict):
            text = text.to_dict() if hasattr(text, "to_dict") else dict(text)
        merged = {**config_dict, **text}

        rope = merged.get("rope_parameters") or {}
        if isinstance(rope, dict) and "rope_theta" in rope:
            merged["rope_theta"] = rope["rope_theta"]
        if isinstance(rope, dict) and "partial_rotary_factor" in rope:
            merged["partial_rotary_factor"] = rope["partial_rotary_factor"]

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in field_names}
        if neuron_config is not None:
            filtered["neuron_config"] = neuron_config

        import os

        override = os.environ.get("QWEN38_NUM_LAYERS")
        if override is not None:
            n = int(override)
            filtered["num_hidden_layers"] = n
            if "layer_types" in filtered:
                filtered["layer_types"] = list(filtered["layer_types"])[:n]

        cfg = cls(**filtered)
        if not cfg.layer_types:
            cfg.layer_types = [
                "linear_attention" if (i + 1) % cfg.full_attention_interval else "full_attention"
                for i in range(cfg.num_hidden_layers)
            ]
        if len(cfg.layer_types) != cfg.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(cfg.layer_types)} entries, "
                f"num_hidden_layers is {cfg.num_hidden_layers}"
            )
        return cfg

    def is_linear_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "linear_attention"

    @property
    def rotary_dim(self) -> int:
        """Partial RoPE width on full-attention layers. 256 * 0.25 = 64."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_conv_dim(self) -> int:
        """Channels through the depthwise conv: Q + K + V."""
        return (
            self.linear_key_head_dim * self.linear_num_key_heads * 2
            + self.linear_value_head_dim * self.linear_num_value_heads
        )
