# SPDX-License-Identifier: Apache-2.0
"""Load a real Transformers Qwen3.5 checkpoint and compare logits.

The checkpoint is a tiny ``Qwen3_5ForCausalLM`` written to safetensors, including
the ``model.language_model.`` prefix used by Qwen3.8-27B. TP=1 must match
Transformers. TP=2 must cover every weight exactly once, and the sharded
DeltaNet and MLP outputs must sum to the unsharded layer.
"""

import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from qwen3_8_neuron.model import Qwen38ForCausalLM


def _hf_config():
    from transformers import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=False,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
        },
    )


def _save_checkpoint(state, directory: Path) -> None:
    prefixed = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = "model.language_model." + key[len("model.") :]
        prefixed[key] = value.contiguous()
    prefixed["model.visual.dummy"] = torch.zeros(1)
    prefixed["mtp.dummy"] = torch.zeros(1)
    save_file(prefixed, directory / "model.safetensors")


def test_loaded_checkpoint_matches_transformers_logits():
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    cfg = _hf_config()
    hf = Qwen3_5ForCausalLM(cfg).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        _save_checkpoint(hf.state_dict(), path)
        ours = Qwen38ForCausalLM.from_configs(cfg.to_dict(), tp_size=1).eval()
        ours.load_weights(str(path), torch.device("cpu"), None)

        ids = torch.randint(0, cfg.vocab_size, (6,))
        with torch.no_grad():
            ref = hf(ids.unsqueeze(0), use_cache=False).logits[0]
            got = ours(ids, torch.arange(6), req_ids=["req"])
            nxt = int(ref[-1].argmax())
            ref_step = hf(
                torch.tensor([[nxt]]),
                past_key_values=hf(ids.unsqueeze(0), use_cache=True).past_key_values,
                use_cache=True,
            ).logits[0, 0]
            got_step = ours(
                torch.tensor([nxt]),
                torch.tensor([6]),
                req_ids=["req"],
            )[0]
    assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
    assert int(got[-1].argmax()) == nxt
    assert torch.allclose(got_step, ref_step, atol=1e-4, rtol=1e-4)


def test_tp_shards_cover_the_checkpoint_and_sum():
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(1)
    cfg = _hf_config()
    hf = Qwen3_5ForCausalLM(cfg).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        _save_checkpoint(hf.state_dict(), path)
        full = Qwen38ForCausalLM.from_configs(cfg.to_dict(), tp_size=1).eval()
        full.load_weights(str(path), torch.device("cpu"), None)
        ranks = []
        for rank in range(2):
            model = Qwen38ForCausalLM.from_configs(cfg.to_dict(), tp_size=2, tp_rank=rank).eval()
            model.load_weights(str(path), torch.device("cpu"), None)
            ranks.append(model)

        def cat(name, dim):
            return torch.cat([r.get_parameter(name).detach() for r in ranks], dim=dim)

        def reassemble_qkv(name):
            parts = {"q": [], "k": [], "v": []}
            local_k = cfg.linear_num_key_heads // 2 * cfg.linear_key_head_dim
            local_v = cfg.linear_num_value_heads // 2 * cfg.linear_value_head_dim
            for model in ranks:
                weight = model.get_parameter(name).detach()
                parts["q"].append(weight[:local_k])
                parts["k"].append(weight[local_k : 2 * local_k])
                parts["v"].append(weight[2 * local_k : 2 * local_k + local_v])
            return torch.cat([torch.cat(parts[name], 0) for name in ("q", "k", "v")], 0)

        assert torch.equal(reassemble_qkv("layers.0.mixer.in_proj_qkv.weight"), full.get_parameter("layers.0.mixer.in_proj_qkv.weight"))
        assert torch.equal(reassemble_qkv("layers.0.mixer.conv1d.weight"), full.get_parameter("layers.0.mixer.conv1d.weight"))
        assert torch.equal(cat("layers.0.mixer.A_log", 0), full.get_parameter("layers.0.mixer.A_log"))
        assert torch.equal(cat("layers.0.mixer.out_proj.weight", 1), full.get_parameter("layers.0.mixer.out_proj.weight"))
        assert torch.equal(cat("layers.0.mlp.gate_proj.weight", 0), full.get_parameter("layers.0.mlp.gate_proj.weight"))
        assert torch.equal(cat("layers.0.mlp.down_proj.weight", 1), full.get_parameter("layers.0.mlp.down_proj.weight"))
        assert torch.equal(cat("layers.1.attn.q_proj.weight", 0), full.get_parameter("layers.1.attn.q_proj.weight"))
        assert torch.equal(cat("layers.1.attn.k_proj.weight", 0), full.get_parameter("layers.1.attn.k_proj.weight"))
        assert torch.equal(cat("layers.1.attn.o_proj.weight", 1), full.get_parameter("layers.1.attn.o_proj.weight"))
        assert torch.equal(ranks[0].get_parameter("embed_tokens.weight"), full.get_parameter("embed_tokens.weight"))
        assert torch.equal(ranks[0].get_parameter("layers.1.attn.q_norm"), full.get_parameter("layers.1.attn.q_norm"))

        x = torch.randn(1, 4, cfg.hidden_size)
        with torch.no_grad():
            mixed = ranks[0].layers[0].mixer(x)[0] + ranks[1].layers[0].mixer(x)[0]
            ref = full.layers[0].mixer(x)[0]
            hidden = torch.randn(3, cfg.hidden_size)
            mlp = ranks[0].layers[0].mlp(hidden) + ranks[1].layers[0].mlp(hidden)
            mlp_ref = full.layers[0].mlp(hidden)
        assert torch.allclose(mixed, ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(mlp, mlp_ref, atol=1e-5, rtol=1e-5)
