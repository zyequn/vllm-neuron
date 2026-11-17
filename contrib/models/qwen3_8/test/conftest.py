# SPDX-License-Identifier: Apache-2.0
"""Load ``qwen3_8_neuron`` from src/ without installing the plugin."""

import importlib.util
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
_spec = importlib.util.spec_from_file_location(
    "qwen3_8_neuron",
    _SRC / "__init__.py",
    submodule_search_locations=[str(_SRC)],
)
_module = importlib.util.module_from_spec(_spec)
sys.modules["qwen3_8_neuron"] = _module
_spec.loader.exec_module(_module)
