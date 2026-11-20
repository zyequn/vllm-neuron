# SPDX-License-Identifier: Apache-2.0
"""Compile-time partition anchor for the native Neuron backend.

The forward is a device ``clone`` so FX → Torch-MLIR legalizes without a custom op.
``barrier_stablehlo_pass`` rewrites the matching StableHLO ops into
``stablehlo.optimization_barrier`` using this file's line in the debug location.
"""

from __future__ import annotations

import torch

# Line used by stablehlo legalization to find barrier anchor sites in MLIR.
_BARRIER_FILE = "optimization_barrier.py"


def _register_op() -> None:
    """No-op kept for register(); barrier uses ``aten.clone`` only."""


def optimization_barrier(x: torch.Tensor) -> torch.Tensor:
    """No-op. ``stablehlo.optimization_barrier`` did not split neuronx-cc graphs."""
    return x