# SPDX-License-Identifier: Apache-2.0
"""Insert ``stablehlo.optimization_barrier`` at Qwen3.8 partition anchors."""

from __future__ import annotations

import logging

from libtorch_neuronx_lite._compiler.torch_mlir.dialects.stablehlo._stablehlo_ops_gen import (
    OptimizationBarrierOp,
)
from libtorch_neuronx_lite._compiler.torch_mlir.ir import InsertionPoint

_LOG = logging.getLogger(__name__)


def _loc_text(operation) -> str:
    root = operation.operation if hasattr(operation, "operation") else operation
    try:
        return str(root.location)
    except Exception:
        return ""


def _root(operation):
    return operation.operation if hasattr(operation, "operation") else operation


def _op_results(operation):
    return list(_root(operation).results)


def _insert_barrier_after(operation, value, *, loc) -> None:
    """Insert ``stablehlo.optimization_barrier`` immediately after ``operation``."""
    root = _root(operation)
    block = root.block
    ops = list(block.operations)
    idx = ops.index(root)
    insert_before = ops[idx + 1] if idx + 1 < len(ops) else None
    if insert_before is not None:
        with InsertionPoint(insert_before):
            barrier = OptimizationBarrierOp([value], results=[value.type], loc=loc)
    else:
        with InsertionPoint(block):
            barrier = OptimizationBarrierOp([value], results=[value.type], loc=loc)
    barrier_out = barrier.results[0]
    barrier_op = barrier.operation
    for use in list(value.uses):
        if use.owner == barrier_op:
            continue
        use.owner.operands[use.operand_number] = barrier_out


def legalize_optimization_barriers(module) -> None:
    """Insert stablehlo optimization barriers at traced anchor sites."""
    count = 0
    to_wrap: list = []

    with module.context:

        def walk_block(block):
            for operation in list(block.operations):
                loc = _loc_text(operation)
                if operation.name.startswith("stablehlo.") and (
                    "optimization_barrier.py" in loc or "write_paged_kv" in loc
                ):
                    to_wrap.append(operation)
                for region in operation.regions:
                    for inner in region.blocks:
                        walk_block(inner)

        for operation in module.body.operations:
            for region in operation.regions:
                for block in region.blocks:
                    walk_block(block)

        for operation in to_wrap:
            results = _op_results(operation)
            if not results:
                continue
            _insert_barrier_after(operation, results[0], loc=operation.location)
            count += 1

        if count:
            _LOG.info("Qwen3.8 mlir optimization_barrier inserted: %d", count)