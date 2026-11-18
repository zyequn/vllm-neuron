# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the text runner. Prefix caching stays off."""

import torch
from types import SimpleNamespace

from qwen3_8_neuron.attention import (
    apply_partial_rope,
    gated_attention,
    partial_rope_cos_sin,
    read_paged_kv,
    write_paged_kv,
)
from qwen3_8_neuron.config import Qwen38TextConfig
from qwen3_8_neuron.gated_delta import Qwen38GatedDeltaNet
from qwen3_8_neuron.model import Qwen38ForCausalLM
from qwen3_8_neuron.slots import DeltaSlotTable


def _tiny_config() -> dict:
    layers = ["linear_attention", "full_attention"]
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "partial_rotary_factor": 0.5,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        "rms_norm_eps": 1e-6,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "layer_types": layers,
        "tie_word_embeddings": False,
    }


def test_delta_matches_transformers_and_chunks():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    cfg = SimpleNamespace(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    torch.manual_seed(0)
    hf = Qwen3_5GatedDeltaNet(cfg, 0).eval()
    ours = Qwen38GatedDeltaNet(32, 2, 4, 8, 8, 4).eval()
    ours.load_state_dict(hf.state_dict())
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        ref = hf(x)
        full, conv, state = ours(x)
        first, conv1, state1 = ours(x[:, :3])
        second, _, _ = ours(x[:, 3:], conv1, state1)
    assert torch.allclose(full, ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(torch.cat([first, second], dim=1), full, atol=1e-4, rtol=1e-4)
    assert conv.shape[-1] == 3
    assert state.shape == (1, 4, 8, 8)


def test_attention_matches_manual_and_pages():
    torch.manual_seed(1)
    attn_h, kv_h, dim, rotary = 4, 2, 8, 4
    seq = 5
    q = torch.randn(attn_h, seq, dim)
    k = torch.randn(kv_h, seq, dim)
    v = torch.randn(kv_h, seq, dim)
    gate = torch.randn(seq, attn_h * dim)
    out = gated_attention(q, k, v, gate, n_past=0)
    # Rebuild one head group by hand: query 0 and 1 share KV head 0.
    scale = dim ** -0.5
    scores = torch.matmul(q[0].float(), k[0].float().T) * scale
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    scores = scores.masked_fill(~causal, torch.finfo(torch.float32).min)
    manual = torch.softmax(scores, dim=-1) @ v[0].float()
    merged = out.view(seq, attn_h, dim)[:, 0, :]
    assert torch.allclose(merged, manual * torch.sigmoid(gate).view(seq, attn_h, dim)[:, 0, :], atol=1e-5)

    positions = torch.arange(seq)
    cos, sin = partial_rope_cos_sin(positions, 10000.0, rotary)
    q_rot, k_rot = apply_partial_rope(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    assert q_rot.shape[-1] == dim
    assert torch.allclose(q_rot[..., rotary:], q[..., rotary:])

    block = 4
    k_cache = torch.zeros(3, kv_h, block, dim)
    v_cache = torch.zeros(3, kv_h, block, dim)
    slots = torch.arange(seq)
    write_paged_kv(k.transpose(0, 1), v.transpose(0, 1), k_cache, v_cache, slots, block)
    table = torch.tensor([0, 1])
    k_back, v_back = read_paged_kv(k_cache, v_cache, table, seq, block)
    assert torch.allclose(k_back, k.transpose(0, 1))
    assert torch.allclose(v_back, v.transpose(0, 1))


def test_chunked_prefill_matches_full_and_prefix_cache_is_off():
    torch.manual_seed(2)
    model = Qwen38ForCausalLM(Qwen38TextConfig.from_configs(_tiny_config()), max_num_seqs=2)
    model.eval()
    ids = torch.randint(0, 64, (6,))
    with torch.no_grad():
        full = model(ids, torch.arange(6), req_ids=["req-a"])
        first = model(ids[:3], torch.arange(3), req_ids=["req-a"])
        second = model(ids[3:], torch.arange(3, 6), req_ids=["req-a"])
    assert model.prefix_caching is False
    assert torch.allclose(first, full[:3], atol=1e-4, rtol=1e-4)
    assert torch.allclose(second, full[3:], atol=1e-4, rtol=1e-4)


def test_slot_is_zeroed_when_reused_and_absent_requests_are_dropped():
    table = DeltaSlotTable(2, 1, 2, 4, 4, 8, 3)
    a = table.ensure("a", reset=True)
    table.recurrent[0, a, 0, 0, 0] = 5
    table.retain_only(set())
    assert table.owner == [None, None]
    b = table.ensure("b", reset=True)
    assert torch.all(table.recurrent[0, b] == 0)
    assert torch.all(table.conv[0, b] == 0)


def test_runner_metadata_uses_block_id_and_releases_finished_requests():
    torch.manual_seed(3)
    model = Qwen38ForCausalLM(Qwen38TextConfig.from_configs(_tiny_config()), max_num_seqs=2)
    model.eval()
    block = 16
    # One request, tokens at positions 0..2 written into block 3.
    ids = torch.randint(0, 64, (3,))
    positions = torch.arange(3)
    meta = {
        "layers.1.self_attn": {
            "block_table_tensor": torch.tensor([[3, 0]]),
            "slot_mapping": torch.tensor([3 * block + i for i in range(3)]),
            "block_size": block,
        }
    }
    spec = model.get_kv_spec()
    caches = {}
    for layer in spec.layers:
        shape = (8, layer.num_kv_heads, block, layer.head_size)
        caches[layer.name] = [torch.zeros(shape), torch.zeros(shape)]
    model.bind_kv_cache(caches)
    with torch.no_grad():
        model(ids, positions, attn_metadata=meta)
    assert 3 in model.delta_slots._index
    # A later batch that does not mention block 3 frees it.
    other = torch.randint(0, 64, (1,))
    meta_b = {
        "layers.1.self_attn": {
            "block_table_tensor": torch.tensor([[4, 0]]),
            "slot_mapping": torch.tensor([4 * block]),
            "block_size": block,
        }
    }
    with torch.no_grad():
        model(other, torch.tensor([0]), attn_metadata=meta_b)
    assert 3 not in model.delta_slots._index
    assert 4 in model.delta_slots._index


def test_layer_reduces_both_shards_before_the_residual():
    import qwen3_8_neuron.model as model_mod

    model = Qwen38ForCausalLM(Qwen38TextConfig.from_configs(_tiny_config()), tp_size=2, tp_rank=0)
    seen = []
    original = model_mod.reduce_replicas

    def _mark(hidden, tp_size):
        seen.append(tp_size)
        return hidden

    model_mod.reduce_replicas = _mark
    try:
        with torch.no_grad():
            model.layers[0].forward_linear(torch.randn(2, 32), None, None)
    finally:
        model_mod.reduce_replicas = original
    assert seen == [2, 2]


def test_value_head_shard_sums_to_full_delta():
    torch.manual_seed(4)
    full = Qwen38GatedDeltaNet(32, 2, 4, 8, 8, 4).eval()
    left = Qwen38GatedDeltaNet(32, 1, 2, 8, 8, 4).eval()
    right = Qwen38GatedDeltaNet(32, 1, 2, 8, 8, 4).eval()

    # Q | K | V channel layout. key_dim = 2*8 = 16, value_dim = 4*8 = 32, conv = 64.
    def shard_rows(weight, rank):
        # weight [conv_dim, hidden] or conv [conv_dim, 1, K]
        q, k, v = weight[:16], weight[16:32], weight[32:]
        return torch.cat(
            [q[rank * 8 : (rank + 1) * 8], k[rank * 8 : (rank + 1) * 8], v[rank * 16 : (rank + 1) * 16]],
            dim=0,
        )

    with torch.no_grad():
        def put(module, name, value):
            module.get_parameter(name).copy_(value)

        for name, param in full.named_parameters():
            if name in ("A_log", "dt_bias", "in_proj_a.weight", "in_proj_b.weight"):
                put(left, name, param[0:2])
                put(right, name, param[2:4])
            elif name in ("in_proj_qkv.weight", "conv1d.weight"):
                put(left, name, shard_rows(param, 0))
                put(right, name, shard_rows(param, 1))
            elif name == "in_proj_z.weight":
                put(left, name, param[0:16])
                put(right, name, param[16:32])
            elif name == "out_proj.weight":
                put(left, name, param[:, 0:16])
                put(right, name, param[:, 16:32])
            elif name == "norm.weight":
                put(left, name, param)
                put(right, name, param)
        x = torch.randn(1, 3, 32)
        ref, _, _ = full(x)
        a, _, _ = left(x)
        b, _, _ = right(x)
    assert torch.allclose(a + b, ref, atol=1e-4, rtol=1e-4)
