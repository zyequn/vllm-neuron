# SPDX-License-Identifier: Apache-2.0
"""Hook native FX → StableHLO lowering to emit partition barriers."""

from __future__ import annotations

_INSTALLED = False


def install_native_barrier_lowering() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from libtorch_neuronx_lite._compiler.neuron_dynamo_backend.fx import fx_transform
    from libtorch_neuronx_lite._compiler.torch_mlir.compiler_utils import (
        run_pipeline_with_repro_report,
    )

    from qwen3_8_neuron.barrier_stablehlo_pass import legalize_optimization_barriers

    _orig_convert = fx_transform.convert_fx_to_stablehlo
    _orig_run = run_pipeline_with_repro_report

    def convert_fx_to_stablehlo(gm, *args, **kwargs):
        module, io_spec, cast_spec = _orig_convert(gm, *args, **kwargs)
        legalize_optimization_barriers(module)
        return module, io_spec, cast_spec

    def _run_with_barrier_pass(module, pipeline, repro_message, **kwargs):
        result = _orig_run(module, pipeline, repro_message, **kwargs)
        if "torch-backend-to-stablehlo-backend-pipeline" in pipeline:
            legalize_optimization_barriers(module)
        return result

    fx_transform.convert_fx_to_stablehlo = convert_fx_to_stablehlo
    import libtorch_neuronx_lite._compiler.torch_mlir.compiler_utils as compiler_utils

    compiler_utils.run_pipeline_with_repro_report = _run_with_barrier_pass
    fx_transform.run_pipeline_with_repro_report = _run_with_barrier_pass
    _INSTALLED = True