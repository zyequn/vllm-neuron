# SPDX-License-Identifier: Apache-2.0
"""Request slots for Gated DeltaNet state.

Prefix caching is off. A slot is owned by one live request and holds the
recurrent matrix and the convolution tail. The owner key is the request id
when the caller has one, or the request's first KV block id when the Neuron
runner only provides a block table. A new owner zeros the slot. An owner that
is absent from the current batch is released, so a finished request cannot
leak into the next one that reuses its blocks.
"""

from __future__ import annotations

import torch


class DeltaSlotTable:
    def __init__(
        self,
        num_slots: int,
        num_layers: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_dim: int,
        conv_tail: int,
        device: torch.device | None = None,
    ):
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots
        self.owner: list[object | None] = [None] * num_slots
        self._index: dict[object, int] = {}
        self.conv = torch.zeros(
            num_layers, num_slots, conv_dim, conv_tail, device=device
        )
        self.recurrent = torch.zeros(
            num_layers,
            num_slots,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            dtype=torch.float32,
            device=device,
        )

    def ensure(self, key: object, reset: bool) -> int:
        """Return the slot for key. reset=True zeros it (a new sequence)."""
        slot = self._index.get(key)
        if slot is None:
            slot = self._allocate(key)
        elif reset:
            self._zero(slot)
        return slot

    def release(self, key: object) -> None:
        slot = self._index.pop(key, None)
        if slot is None:
            return
        self.owner[slot] = None
        self._zero(slot)

    def retain_only(self, live_keys: set) -> None:
        """Drop owners that are not in this batch."""
        for key in list(self._index):
            if key not in live_keys:
                self.release(key)

    def _allocate(self, key: object) -> int:
        for slot, owner in enumerate(self.owner):
            if owner is None:
                self.owner[slot] = key
                self._index[key] = slot
                self._zero(slot)
                return slot
        raise RuntimeError(
            f"Gated DeltaNet slot table is full ({self.num_slots} live requests)"
        )

    def _zero(self, slot: int) -> None:
        self.conv[:, slot].zero_()
        self.recurrent[:, slot].zero_()
