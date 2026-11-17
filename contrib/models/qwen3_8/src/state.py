# SPDX-License-Identifier: Apache-2.0
"""Fixed per-sequence state for Gated DeltaNet layers.

Standard vLLM keeps this state in a hybrid KV manager next to the growing
full-attention cache, and runs the recurrence with Flash Linear Attention
(Triton). Neither exists on Trainium.

This port does not declare vLLM's mamba / hybrid interface. The Neuron runner
allocates paged KV only for ``full_attention`` layers (head_dim 256, which is
also above the NKI attention cap of 128, so that attention is PyTorch).

Each ``linear_attention`` layer keeps two fixed buffers, indexed by sequence
slot (``max_num_seqs``), not by token:

  conv_state:      [max_num_seqs, conv_dim, kernel_size - 1]
  recurrent_state: [max_num_seqs, num_v_heads, head_k_dim, head_v_dim]  fp32

For Qwen3.8-27B that recurrent matrix is 48 * 128 * 128 float32, about
3.1 MiB per sequence per linear layer. There are 48 linear layers, so about
150 MiB per sequence, plus a short conv tail. The state does not grow with
context length. A new sequence starts from zeros.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatedDeltaStateSpec:
    """Shapes of one linear layer's state, excluding the sequence-slot axis."""

    conv_dim: int
    conv_tail: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int

    @property
    def recurrent_numel(self) -> int:
        return self.num_v_heads * self.head_k_dim * self.head_v_dim

    @property
    def conv_numel(self) -> int:
        return self.conv_dim * self.conv_tail
