# SPDX-License-Identifier: Apache-2.0
"""vLLM ModelRegistry entry for Qwen3.8-27B.

Registered as ``Qwen3_5ForConditionalGeneration``, which is the architecture
string in the checkpoint. The class does not declare vLLM's mamba or hybrid
interface: that would allocate a GPU hybrid cache the Neuron runner cannot use.
Gated DeltaNet state is a fixed per-sequence buffer (see state.py). Full
attention uses the ordinary paged KV spec.

The Neuron runner builds the implementation with ``from_configs``. The other
methods exist so vLLM's text-generation protocol check passes in the API process.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5ForConditionalGeneration(nn.Module):
    is_text_generation_model = True

    def __init__(
        self,
        hf_config: PretrainedConfig | None = None,
        neuron_config: NeuronConfig | None = None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # The worker builds the text stack with from_configs. Building it here
        # would allocate the 27B parameters in the API process.
        del hf_config, neuron_config, vllm_config, prefix
        self._model = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("vllm-neuron embeds inside the compiled forward")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, *args, **kwargs):
        return self._model(input_ids, positions, *args, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("vllm-neuron runs the LM head inside forward()")

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config=None,
    ) -> nn.Module:
        if neuron_config is None:
            neuron_config = text_neuron_config
        cls._validate_config(hf_config, neuron_config)
        from .model import Qwen38ForCausalLM

        return Qwen38ForCausalLM.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(cls, hf_config, neuron_config: NeuronConfig | None) -> None:
        quantization = getattr(neuron_config, "quantization", None) if neuron_config else None
        if quantization not in (None, "bf16", "fp8"):
            raise ValueError(
                f"Qwen3.8 contrib model supports bf16 and static fp8, got quantization={quantization!r}"
            )
        archs = list(getattr(hf_config, "architectures", None) or [])
        if isinstance(hf_config, dict):
            archs = list(hf_config.get("architectures") or [])
        if archs and "Qwen3_5ForConditionalGeneration" not in archs:
            raise ValueError(
                f"expected Qwen3_5ForConditionalGeneration, got {archs}"
            )
