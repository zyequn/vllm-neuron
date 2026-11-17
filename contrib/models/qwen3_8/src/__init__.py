# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B text decoder for vLLM Neuron (contrib).

The checkpoint architecture is ``Qwen3_5ForConditionalGeneration``. This package
registers that name with vLLM. Vision and multi-token prediction are not
implemented. The text stack is hybrid: Gated DeltaNet layers plus gated
full attention with head_dim 256.

``pip install -e contrib/models/qwen3_8`` adds a ``vllm.general_plugins`` entry
point so ``register()`` runs in every vLLM process.
"""

import logging

ARCHS = ("Qwen3_5ForConditionalGeneration",)
IMPL = "qwen3_8_neuron.factory:Qwen3_5ForConditionalGeneration"


class _DropOverwriteWarning(logging.Filter):
    def filter(self, record):
        return ARCHS[0] not in record.getMessage()


def _extend_neuron_model_list() -> None:
    from vllm_neuron.model import registry

    if getattr(registry.get_models, "_qwen3_8", False):
        return
    original = registry.get_models

    def get_models():
        from .factory import Qwen3_5ForConditionalGeneration

        models = [m for m in original() if m[0] not in ARCHS]
        return models + [(arch, Qwen3_5ForConditionalGeneration) for arch in ARCHS]

    get_models._qwen3_8 = True  # type: ignore[attr-defined]
    registry.get_models = get_models


def register() -> None:
    """Make Qwen3.8 available to vLLM Neuron. Safe to call more than once."""
    from vllm import ModelRegistry

    log = logging.getLogger("vllm.model_executor.models.registry")
    quiet = _DropOverwriteWarning()
    log.addFilter(quiet)
    try:
        for arch in ARCHS:
            ModelRegistry.register_model(arch, IMPL)
    finally:
        log.removeFilter(quiet)
    _extend_neuron_model_list()


__all__ = ["ARCHS", "register"]
