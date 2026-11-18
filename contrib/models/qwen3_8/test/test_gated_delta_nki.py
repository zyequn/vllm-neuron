# SPDX-License-Identifier: Apache-2.0
"""CPU-simulator check of the NKI Gated DeltaNet recurrence."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import nki
from gated_delta_nki import gated_delta_recurrence


def reference(query, key, value, decay, beta, state):
    """query/key [H, K, S], value [H, V, S], decay/beta [H, S], state [H, K, V]."""
    heads, _k, seq = query.shape
    out = torch.empty(heads, value.shape[1], seq, dtype=torch.float32)
    carried = state.clone()
    for token in range(seq):
        carried = carried * decay[:, token].view(heads, 1, 1)
        kv = (carried * key[:, :, token].unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, token] - kv) * beta[:, token].unsqueeze(-1)
        carried = carried + key[:, :, token].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, token] = (carried * query[:, :, token].unsqueeze(-1)).sum(dim=-2)
    return out, carried


def check(heads: int, seq: int, seed: int) -> None:
    torch.manual_seed(seed)
    k_dim = v_dim = 128
    query = torch.randn(heads, k_dim, seq)
    key = torch.randn(heads, k_dim, seq)
    value = torch.randn(heads, v_dim, seq)
    decay = torch.rand(heads, seq) * 0.5 + 0.5
    beta = torch.rand(heads, seq)
    state = torch.randn(heads, k_dim, v_dim) * 0.01
    expected_out, expected_state = reference(query, key, value, decay, beta, state)
    seq_len = np.array([[seq]], dtype=np.int32)
    got_out, got_state = nki.simulate(gated_delta_recurrence)(
        query.numpy(),
        key.numpy(),
        value.numpy(),
        decay.numpy(),
        beta.numpy(),
        state.numpy(),
        seq_len,
    )
    got_out = torch.from_numpy(np.asarray(got_out))
    got_state = torch.from_numpy(np.asarray(got_state))
    out_rel = (got_out - expected_out).norm() / expected_out.norm()
    state_rel = (got_state - expected_state).norm() / expected_state.norm()
    print(
        f"heads={heads} seq={seq} out_rel={out_rel.item():.3e} state_rel={state_rel.item():.3e}"
    )
    if out_rel > 1e-4 or state_rel > 1e-4:
        raise SystemExit(f"mismatch out_rel={out_rel} state_rel={state_rel}")


if __name__ == "__main__":
    check(heads=2, seq=4, seed=0)
    check(heads=4, seq=7, seed=1)
    print("OK")
