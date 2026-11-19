# SPDX-License-Identifier: Apache-2.0
"""Per-tensor static FP8 for Qwen3.8 projections.

Weights stay ``float8_e4m3fn`` in HBM. A matmul dequantizes one tile at a
time and multiplies in fp32, which is the same product as
``(scale_w * W_fp8) @ x`` without a second copy of the matrix. On Trn2 the
stored bytes are rescaled into the legacy E4M3 range (max 240) and the
dequant scale is multiplied by the inverse, matching the Llama static-FP8
loader.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_FP8 = torch.float8_e4m3fn
_OCP_MAX = 448.0
_LEGACY_MAX = 240.0
_WEIGHT_DOWNSCALE = _LEGACY_MAX / _OCP_MAX
_SCALE_COMPENSATION = _OCP_MAX / _LEGACY_MAX
_TILE = 128


def needs_trn2_downscale() -> bool:
    """True when the tensor engine's FP8 max is 240 rather than 448."""
    try:
        from libtorch_neuronx_lite.compile.platform import get_platform_target
    except Exception:
        return False
    target = get_platform_target()
    return target == "trn2" or str(target).startswith("trn2-")


def _row_step(rows: int, width: int) -> int:
    """Rows whose FP32 workspace stays under 32 MiB."""
    if width <= 0:
        return max(rows, 1)
    return max(1, (32 * 1024 * 1024) // (width * 4))


def quantize_per_tensor(
    weight: torch.Tensor, fp8_max: float = _OCP_MAX
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(fp8_weight, dequant_scale)`` with ``fp8 * scale ~= weight``.

    ``dequant_scale`` has shape ``[1, 1]`` and dtype fp32. The FP32
    workspace is one row-chunk, so a replicated embedding does not
    allocate a second full-size copy.
    """
    source = weight.detach()
    rows = source.shape[0]
    width = source[0].numel() if rows else 1
    step = _row_step(rows, width)
    amax = source.new_zeros(())
    for start in range(0, rows, step):
        amax = torch.maximum(amax, source[start : start + step].abs().amax())
    amax = amax.float().clamp(min=1e-12)
    scale = (amax / fp8_max).reshape(1, 1)
    quantized = torch.empty(source.shape, dtype=_FP8)
    for start in range(0, rows, step):
        slab = source[start : start + step].float()
        quantized[start : start + step] = (slab / scale).clamp(-fp8_max, fp8_max).to(_FP8)
    return quantized, scale


def downscale_for_trn2(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map an OCP FP8 tensor into the legacy 240-range and compensate the scale."""
    scaled = torch.empty(weight.shape, dtype=_FP8)
    rows = weight.shape[0]
    width = weight[0].numel() if rows else 1
    step = _row_step(rows, width)
    for start in range(0, rows, step):
        slab = weight[start : start + step].float()
        scaled[start : start + step] = (slab * _WEIGHT_DOWNSCALE).clamp(
            -_LEGACY_MAX, _LEGACY_MAX
        ).to(_FP8)
    return scaled, scale.float() * _SCALE_COMPENSATION


def install_quantized_weight(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Replace a 2D parameter with static FP8 bytes plus a dequant scale."""
    quantized, scale = quantize_per_tensor(tensor)
    if needs_trn2_downscale():
        quantized, scale = downscale_for_trn2(quantized, scale)
    parent = module
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    parent.register_parameter(parts[-1], nn.Parameter(quantized, requires_grad=False))
    parent.register_buffer("weight_scale", scale.float().reshape(1, 1))


def is_fp8_weight(weight: torch.Tensor) -> bool:
    return weight.dtype == _FP8


def activation_fp8_max() -> float:
    return _LEGACY_MAX if needs_trn2_downscale() else _OCP_MAX


def requantize_activation(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """Quantize ``x`` to FP8 with ``input_scale`` and dequantize back to fp32."""
    scale = input_scale.float().reshape(())
    fp8_max = activation_fp8_max()
    return (x.float() / scale).clamp(-fp8_max, fp8_max).to(_FP8).float() * scale


def reference_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """BF16/FP32 matmul of the dequantized FP8 weight. This is the numeric definition."""
    out_dtype = x.dtype if x.dtype != _FP8 else torch.bfloat16
    values = x.float()
    if input_scale is not None:
        values = requantize_activation(values, input_scale)
    dequant = weight.float() * weight_scale.float()
    return F.linear(values, dequant).to(dtype=out_dtype)


def linear(layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """``F.linear`` for BF16 weights, static FP8 for quantized ones."""
    weight = layer.weight
    if not is_fp8_weight(weight):
        return F.linear(x.to(dtype=weight.dtype), weight, layer.bias)
    input_scale = getattr(layer, "input_scale", None)
    return static_fp8_linear(x, weight, layer.weight_scale, input_scale)


def embed(table: nn.Embedding, input_ids: torch.Tensor) -> torch.Tensor:
    """Gather rows. FP8 tables are dequantized after the gather, not before it."""
    weight = table.weight
    if not is_fp8_weight(weight):
        return table(input_ids)
    rows = weight.index_select(0, input_ids.reshape(-1))
    values = rows.float() * table.weight_scale.float().reshape(())
    return values.view(*input_ids.shape, -1).to(dtype=torch.bfloat16)


def static_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project ``x`` by a static-FP8 weight.

    On CPU the dequantized matmul is the reference. On Neuron the weight
    tensor passed into the kernel stays FP8; only the tile in SBUF is widened.
    """
    out_dtype = x.dtype if x.dtype != _FP8 else torch.bfloat16
    if weight.device.type in ("cpu", "meta"):
        return reference_linear(x, weight, weight_scale, input_scale)
    values = x if x.dtype != _FP8 else x.float()
    if input_scale is not None:
        values = requantize_activation(values, input_scale)
    # The resident weight stays FP8. Widening it is the matmul's weight
    # operand, which is the same dot the bf16 model already compiles.
    scale = weight_scale.detach().float().reshape(())
    widened = (weight.float() * scale).to(dtype=values.dtype)
    return F.linear(values, widened).to(dtype=out_dtype)


def _linear_on_neuron(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    flat = x.reshape(-1, x.shape[-1]).contiguous()
    if flat.shape[-1] % _TILE or weight.shape[-1] % _TILE:
        raise ValueError(
            f"static FP8 K must be a multiple of {_TILE}, got x {tuple(x.shape)} "
            f"weight {tuple(weight.shape)}"
        )
    kernel = wrap_nki(static_fp8_gemm)
    scale = weight_scale.detach().float().reshape(1, 1).contiguous()
    produced = kernel(flat, weight.contiguous(), scale)
    return produced.transpose(0, 1).reshape(*x.shape[:-1], weight.shape[0])


import nki


def _fill_scale(src, dst) -> None:
    """Broadcast a ``[1, 1]`` scale across every partition of ``dst``."""
    import nki.isa as nisa

    n_parts = dst.shape[0]
    if n_parts == 1:
        nisa.tensor_copy(dst=dst, src=src)
        return
    shuffle_mask = [0] * 32
    for group in range((n_parts + 31) // 32):
        start = group * 32
        width = min(32, n_parts - start)
        nisa.nc_stream_shuffle(
            src=src, dst=dst[start : start + width, 0:1], shuffle_mask=shuffle_mask
        )


@nki.jit
def static_fp8_gemm(x, weight, weight_scale):
    """FP8 weight times a fp32 activation.

    ``x`` is ``[M, K]`` fp32, ``weight`` is ``[N, K]`` FP8, ``weight_scale``
    is ``[1, 1]`` fp32. ``K`` is a multiple of 128. Every tensor-engine tile
    is allocated at its true shape and addressed with static bounds, so a
    short ``M`` or ``N`` is a dense tile. The result is ``[N, M]`` fp32,
    equal to ``(scale * weight) @ x.T``.
    """
    import nki.isa as nisa
    import nki.language as nl

    m_rows, k_dim = x.shape
    n_rows = weight.shape[0]
    tile = _TILE
    out = nl.ndarray((n_rows, m_rows), dtype=nl.float32, buffer=nl.shared_hbm)
    scale_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_sb, src=weight_scale[0:1, 0:1])
    k_tiles = k_dim // tile

    def _emit(n_off, n_sz, m_off, m_sz):
        acc = nl.ndarray((n_sz, m_sz), dtype=nl.float32, buffer=nl.psum)
        for k in range(k_tiles):
            k_off = k * tile
            x_sb = nl.ndarray((m_sz, tile), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=x_sb,
                src=x[m_off : m_off + m_sz, k_off : k_off + tile],
            )
            x_ps = nl.ndarray((tile, m_sz), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=x_ps, data=x_sb)
            x_t = nl.ndarray((tile, m_sz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x_t, src=x_ps)

            w_fp8 = nl.ndarray((n_sz, tile), dtype=weight.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=w_fp8,
                src=weight[n_off : n_off + n_sz, k_off : k_off + tile],
            )
            w_f32 = nl.ndarray((n_sz, tile), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_f32, src=w_fp8)
            w_ps = nl.ndarray((tile, n_sz), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=w_ps, data=w_f32)
            w_t = nl.ndarray((tile, n_sz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_t, src=w_ps)
            nisa.nc_matmul(
                dst=acc, stationary=w_t, moving=x_t, accumulate=(k > 0)
            )

        acc_sb = nl.ndarray((n_sz, m_sz), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=acc_sb, src=acc)
        scale_vec = nl.ndarray((n_sz, 1), dtype=nl.float32, buffer=nl.sbuf)
        _fill_scale(scale_sb, scale_vec)
        scaled = nl.ndarray((n_sz, m_sz), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=scaled, data=acc_sb, op0=nl.multiply, operand0=scale_vec
        )
        nisa.dma_copy(
            dst=out[n_off : n_off + n_sz, m_off : m_off + m_sz],
            src=scaled,
        )

    m_full, n_full = m_rows // tile, n_rows // tile
    m_rem, n_rem = m_rows % tile, n_rows % tile
    for n in range(n_full):
        n_off = n * tile
        for m in range(m_full):
            _emit(n_off, tile, m * tile, tile)
        if m_rem:
            _emit(n_off, tile, m_full * tile, m_rem)
    if n_rem:
        n_off = n_full * tile
        for m in range(m_full):
            _emit(n_off, n_rem, m * tile, tile)
        if m_rem:
            _emit(n_off, n_rem, m_full * tile, m_rem)
    return out
