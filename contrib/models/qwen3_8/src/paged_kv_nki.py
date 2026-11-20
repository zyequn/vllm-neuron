# SPDX-License-Identifier: Apache-2.0
"""Paged KV write as one NKI DMA store.

Modular flow builds a macro around each HBM store and asserts that the macro
root is that store. An HLO scatter (``cache[block, head, pos] = row``) is
fused into the FP8 matmul that produced the row, so the root is a dot and
MacroGeneration aborts (``Expected Store as root``). A NKI custom call is a
hard partition: the matmul stays in HLO, and the only HBM write inside the
kernel is ``nisa.dma_copy`` with a vector offset.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

_TILE = 128


@nki.jit
def paged_kv_store(rows, indices, cache):
    """Scatter ``rows[i]`` into ``cache[indices[i]]``.

    ``rows`` is ``[R, D]``, ``indices`` is ``[R, 1]`` int32, ``cache`` is
    ``[N, D]``. ``R`` is a multiple of 128 (the caller repeats the last row).
    The kernel returns ``cache`` so the compiler aliases the write to the
    caller's buffer.
    """
    row_count, width = rows.shape
    for start in range(0, row_count, _TILE):
        src = nl.ndarray((_TILE, width), dtype=rows.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=src, src=rows[start : start + _TILE, 0:width])
        slot = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=slot, src=indices[start : start + _TILE, 0:1])
        nisa.dma_copy(
            dst=cache.ap(
                pattern=[[width, _TILE], [1, width]],
                offset=0,
                vector_offset=slot,
                indirect_dim=0,
            ),
            src=src,
        )
    return cache
