# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet recurrence as one NKI kernel.

The PyTorch token loop is unrolled by torch.compile, so a 1536-token prefill
becomes tens of thousands of graph nodes. This kernel keeps that loop in one
``nl.fori_loop``. The compiled model sees a single custom call per layer.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl


def _kernel_assert(cond: bool, message: str) -> None:
    if not cond:
        raise Exception(f"[NCC_INKI016] Kernel validation exception: {message}")


def _broadcast_partitions(src, dst) -> None:
    """Copy partition 0 of ``src`` onto every partition of ``dst``."""
    n_parts = dst.shape[0]
    shuffle_mask = [0] * 32
    for group in range((n_parts + 31) // 32):
        width = min(32, n_parts - group * 32)
        nisa.nc_stream_shuffle(
            src=src[0:1, 0 : src.shape[1]],
            dst=dst[group * 32 : group * 32 + width, 0 : dst.shape[1]],
            shuffle_mask=shuffle_mask,
        )


@nki.jit
def gated_delta_recurrence(query, key, value, decay, beta, state, seq_len):
    """One shared recurrence body for every token.

    Args:
        query: [heads, K, seq] float32. K is 128.
        key: [heads, K, seq] float32.
        value: [heads, V, seq] float32. V is 128.
        decay: [heads, seq] float32. Already ``exp(g)``.
        beta: [heads, seq] float32.
        state: [heads, K, V] float32. Initial recurrent state.
        seq_len: [1, 1] int32. Runtime trip count so the loop is not unrolled.

    Returns:
        out: [heads, V, seq] float32.
        new_state: [heads, K, V] float32.
    """
    heads, k_dim, _seq = query.shape
    v_dim = value.shape[1]
    _kernel_assert(k_dim == 128 and v_dim == 128, "head dims must be 128")
    _kernel_assert(key.shape[0] == heads and value.shape[0] == heads, "head count mismatch")
    _kernel_assert(state.shape == (heads, k_dim, v_dim), "state shape must be [heads, 128, 128]")

    out = nl.ndarray((heads, v_dim, query.shape[2]), dtype=nl.float32, buffer=nl.shared_hbm)
    new_state = nl.ndarray((heads, k_dim, v_dim), dtype=nl.float32, buffer=nl.shared_hbm)

    seq_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=seq_sb, src=seq_len)
    trip = nisa.register_alloc()
    nisa.register_load(dst=trip, src=seq_sb)

    carried = []
    for head in range(heads):
        tile = nl.ndarray((k_dim, v_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile, src=state[head, 0:k_dim, 0:v_dim])
        carried.append(tile)

    k_col = nl.ndarray((k_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    q_col = nl.ndarray((k_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    v_col = nl.ndarray((v_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    v_row = nl.ndarray((1, v_dim), dtype=nl.float32, buffer=nl.sbuf)
    decay_p0 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    decay_vec = nl.ndarray((k_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    beta_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    scaled = nl.ndarray((k_dim, v_dim), dtype=nl.float32, buffer=nl.sbuf)
    kv_row = nl.ndarray((1, v_dim), dtype=nl.float32, buffer=nl.sbuf)
    diff_row = nl.ndarray((1, v_dim), dtype=nl.float32, buffer=nl.sbuf)
    delta_row = nl.ndarray((1, v_dim), dtype=nl.float32, buffer=nl.sbuf)
    k_row = nl.ndarray((1, k_dim), dtype=nl.float32, buffer=nl.sbuf)
    outer = nl.ndarray((k_dim, v_dim), dtype=nl.float32, buffer=nl.sbuf)
    out_row = nl.ndarray((1, v_dim), dtype=nl.float32, buffer=nl.sbuf)
    out_col = nl.ndarray((v_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
    kv_ps = nl.ndarray((1, v_dim), dtype=nl.float32, buffer=nl.psum)
    k_row_ps = nl.ndarray((1, k_dim), dtype=nl.float32, buffer=nl.psum)
    outer_ps = nl.ndarray((k_dim, v_dim), dtype=nl.float32, buffer=nl.psum)
    out_ps = nl.ndarray((1, v_dim), dtype=nl.float32, buffer=nl.psum)
    v_row_ps = nl.ndarray((1, v_dim), dtype=nl.float32, buffer=nl.psum)
    out_col_ps = nl.ndarray((v_dim, 1), dtype=nl.float32, buffer=nl.psum)

    def body(token):
        for head in range(heads):
            nisa.dma_copy(
                dst=k_col,
                src=key[head, 0:k_dim, nl.ds(token, 1)],
                dge_mode=nisa.dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=q_col,
                src=query[head, 0:k_dim, nl.ds(token, 1)],
                dge_mode=nisa.dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=v_col,
                src=value[head, 0:v_dim, nl.ds(token, 1)],
                dge_mode=nisa.dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=decay_p0,
                src=decay[head : head + 1, nl.ds(token, 1)],
                dge_mode=nisa.dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=beta_sb,
                src=beta[head : head + 1, nl.ds(token, 1)],
                dge_mode=nisa.dge_mode.hwdge,
            )
            _broadcast_partitions(decay_p0, decay_vec)
            nisa.nc_transpose(dst=v_row_ps, data=v_col)
            nisa.tensor_copy(dst=v_row, src=v_row_ps)

            nisa.tensor_scalar(
                dst=scaled, data=carried[head], op0=nl.multiply, operand0=decay_vec
            )
            nisa.nc_matmul(dst=kv_ps, stationary=k_col, moving=scaled, accumulate=False)
            nisa.tensor_copy(dst=kv_row, src=kv_ps)
            nisa.tensor_tensor(dst=diff_row, data1=v_row, data2=kv_row, op=nl.subtract)
            nisa.tensor_scalar(
                dst=delta_row, data=diff_row, op0=nl.multiply, operand0=beta_sb
            )
            nisa.nc_transpose(dst=k_row_ps, data=k_col)
            nisa.tensor_copy(dst=k_row, src=k_row_ps)
            nisa.nc_matmul(dst=outer_ps, stationary=k_row, moving=delta_row, accumulate=False)
            nisa.tensor_copy(dst=outer, src=outer_ps)
            nisa.tensor_tensor(dst=carried[head], data1=scaled, data2=outer, op=nl.add)

            nisa.nc_matmul(
                dst=out_ps, stationary=q_col, moving=carried[head], accumulate=False
            )
            nisa.tensor_copy(dst=out_row, src=out_ps)
            nisa.nc_transpose(dst=out_col_ps, data=out_row)
            nisa.tensor_copy(dst=out_col, src=out_col_ps)
            nisa.dma_copy(
                dst=out[head, 0:v_dim, nl.ds(token, 1)],
                src=out_col,
                dge_mode=nisa.dge_mode.hwdge,
            )

    nl.fori_loop(0, trip, body)
    for head in range(heads):
        nisa.dma_copy(dst=new_state[head, 0:k_dim, 0:v_dim], src=carried[head])
    return out, new_state
