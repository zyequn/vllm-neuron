# SPDX-License-Identifier: Apache-2.0
"""Load a Qwen3.5 / Qwen3.8 text checkpoint into the runner.

Vision and MTP tensors are skipped. Layer projections are sharded for the
local tensor-parallel rank. Embeddings, norms, and the LM head stay replicated
so each rank can form logits without an extra gather.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .config import Qwen38TextConfig


def _checkpoint_files(path: str | os.PathLike) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    files = sorted(root.glob("*.safetensors"))
    if not files:
        files = sorted(root.glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"no checkpoint tensors in {root}")
    return files


def _read_tensors(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for file in _checkpoint_files(path):
        if file.suffix == ".safetensors":
            from safetensors.torch import load_file

            tensors.update(load_file(str(file)))
        else:
            tensors.update(torch.load(file, map_location="cpu", weights_only=True))
    return tensors


def _strip(key: str) -> str | None:
    if key.startswith("visual.") or key.startswith("model.visual.") or ".mtp." in key:
        return None
    for prefix in ("model.language_model.", "model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _shard(weight: torch.Tensor, rank: int, world: int, pieces: int) -> torch.Tensor:
    if world == 1:
        return weight
    if pieces % world:
        raise ValueError(f"cannot shard {pieces} across TP={world}")
    width = pieces // world
    return weight[rank * width : (rank + 1) * width]


def _shard_qkv(
    weight: torch.Tensor,
    rank: int,
    world: int,
    num_k: int,
    num_v: int,
    head_k: int,
    head_v: int,
) -> torch.Tensor:
    """Q | K | V along dim 0. The conv kernel axis, if present, stays last."""
    key = num_k * head_k
    value = num_v * head_v
    q, k, v = weight[:key], weight[key : 2 * key], weight[2 * key : 2 * key + value]
    q = _shard(q.reshape(num_k, head_k, *q.shape[1:]), rank, world, num_k).reshape(-1, *q.shape[1:])
    k = _shard(k.reshape(num_k, head_k, *k.shape[1:]), rank, world, num_k).reshape(-1, *k.shape[1:])
    v = _shard(v.reshape(num_v, head_v, *v.shape[1:]), rank, world, num_v).reshape(-1, *v.shape[1:])
    return torch.cat([q, k, v], dim=0)


def _shard_out(weight: torch.Tensor, rank: int, world: int, num_v: int, head_v: int) -> torch.Tensor:
    """out_proj is [hidden, value_dim]. Shard the value-head axis, not hidden."""
    hidden = weight.shape[0]
    value = weight[:, : num_v * head_v].reshape(hidden, num_v, head_v)
    if world > 1:
        if num_v % world:
            raise ValueError(f"cannot shard {num_v} value heads across TP={world}")
        width = num_v // world
        value = value.narrow(1, rank * width, width)
    return value.reshape(hidden, -1)


def _assign(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    param = module.get_parameter(name)
    if param.shape != tensor.shape:
        raise ValueError(f"{name}: checkpoint {tuple(tensor.shape)} != parameter {tuple(param.shape)}")
    with torch.no_grad():
        param.copy_(tensor.to(dtype=param.dtype))


def load_text_weights(model, path: str | os.PathLike) -> list[str]:
    """Copy text weights into ``model``. Returns the parameter names that were set."""
    config: Qwen38TextConfig = model.config
    rank, world = model.tp_rank, model.tp_size
    raw = _read_tensors(path)
    wanted: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        stripped = _strip(key)
        if stripped is None or stripped.startswith("visual"):
            continue
        wanted[stripped] = tensor

    loaded: list[str] = []

    def take(src: str, dest: str, tensor: torch.Tensor) -> None:
        _assign(model, dest, tensor)
        loaded.append(dest)

    if "embed_tokens.weight" not in wanted or "lm_head.weight" not in wanted:
        raise KeyError("checkpoint is missing embed_tokens.weight or lm_head.weight")
    take("embed_tokens.weight", "embed_tokens.weight", wanted["embed_tokens.weight"])
    take("lm_head.weight", "lm_head.weight", wanted["lm_head.weight"])
    take("norm.weight", "norm.weight", wanted["norm.weight"])

    num_k = config.linear_num_key_heads
    num_v = config.linear_num_value_heads
    head_k = config.linear_key_head_dim
    head_v = config.linear_value_head_dim
    heads = config.num_attention_heads
    kv = config.num_key_value_heads
    head = config.head_dim

    for layer in model.layers:
        i = layer.layer_idx
        prefix = f"layers.{i}."
        take(prefix + "input_layernorm.weight", f"layers.{i}.input_layernorm.weight", wanted[prefix + "input_layernorm.weight"])
        take(
            prefix + "post_attention_layernorm.weight",
            f"layers.{i}.post_attention_layernorm.weight",
            wanted[prefix + "post_attention_layernorm.weight"],
        )
        gate = wanted[prefix + "mlp.gate_proj.weight"]
        up = wanted[prefix + "mlp.up_proj.weight"]
        down = wanted[prefix + "mlp.down_proj.weight"]
        take(prefix + "mlp.gate_proj.weight", f"layers.{i}.mlp.gate_proj.weight", _shard(gate, rank, world, gate.shape[0]))
        take(prefix + "mlp.up_proj.weight", f"layers.{i}.mlp.up_proj.weight", _shard(up, rank, world, up.shape[0]))
        inter = down.shape[1]
        local = inter // world
        take(
            prefix + "mlp.down_proj.weight",
            f"layers.{i}.mlp.down_proj.weight",
            down[:, rank * local : (rank + 1) * local],
        )
        if layer.is_linear:
            src = prefix + "linear_attn."
            dest = f"layers.{i}.mixer."
            take(src + "in_proj_qkv.weight", dest + "in_proj_qkv.weight", _shard_qkv(wanted[src + "in_proj_qkv.weight"], rank, world, num_k, num_v, head_k, head_v))
            z = wanted[src + "in_proj_z.weight"]
            take(src + "in_proj_z.weight", dest + "in_proj_z.weight", _shard(z.reshape(num_v, head_v, -1), rank, world, num_v).reshape(-1, z.shape[-1]))
            take(src + "in_proj_a.weight", dest + "in_proj_a.weight", _shard(wanted[src + "in_proj_a.weight"], rank, world, num_v))
            take(src + "in_proj_b.weight", dest + "in_proj_b.weight", _shard(wanted[src + "in_proj_b.weight"], rank, world, num_v))
            take(src + "conv1d.weight", dest + "conv1d.weight", _shard_qkv(wanted[src + "conv1d.weight"], rank, world, num_k, num_v, head_k, head_v))
            take(src + "A_log", dest + "A_log", _shard(wanted[src + "A_log"], rank, world, num_v))
            take(src + "dt_bias", dest + "dt_bias", _shard(wanted[src + "dt_bias"], rank, world, num_v))
            take(src + "norm.weight", dest + "norm.weight", wanted[src + "norm.weight"])
            take(src + "out_proj.weight", dest + "out_proj.weight", _shard_out(wanted[src + "out_proj.weight"], rank, world, num_v, head_v))
        else:
            src = prefix + "self_attn."
            dest = f"layers.{i}.attn."
            q = wanted[src + "q_proj.weight"]
            q = q.reshape(heads, 2 * head, -1)
            q = _shard(q, rank, world, heads).reshape(-1, q.shape[-1])
            take(src + "q_proj.weight", dest + "q_proj.weight", q)
            k = wanted[src + "k_proj.weight"].reshape(kv, head, -1)
            v = wanted[src + "v_proj.weight"].reshape(kv, head, -1)
            take(src + "k_proj.weight", dest + "k_proj.weight", _shard(k, rank, world, kv).reshape(-1, k.shape[-1]))
            take(src + "v_proj.weight", dest + "v_proj.weight", _shard(v, rank, world, kv).reshape(-1, v.shape[-1]))
            o = wanted[src + "o_proj.weight"]
            o = o.reshape(o.shape[0], heads, head)
            o = _shard(o.transpose(0, 1), rank, world, heads).transpose(0, 1).reshape(o.shape[0], -1)
            take(src + "o_proj.weight", dest + "o_proj.weight", o)
            take(src + "q_norm.weight", dest + "q_norm", wanted[src + "q_norm.weight"])
            take(src + "k_norm.weight", dest + "k_norm", wanted[src + "k_norm.weight"])
    missing = [name for name, _ in model.named_parameters() if name not in loaded]
    if missing:
        raise KeyError("checkpoint did not fill: " + ", ".join(missing))
    return loaded


def config_from_checkpoint(path: str | os.PathLike) -> dict:
    root = Path(path)
    config_path = root / "config.json" if root.is_dir() else None
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"no config.json next to {path}")
    with open(config_path) as handle:
        return json.load(handle)
