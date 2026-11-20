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


def resolve_checkpoint(path: str | os.PathLike, cache_dir: str | None) -> Path:
    """Return a local checkpoint directory. A repo id is downloaded first."""
    root = Path(path)
    if root.is_dir() or root.is_file():
        return root
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(path), cache_dir=cache_dir))


def checkpoint_dtype(path: str | os.PathLike) -> torch.dtype:
    """Dtype of the first tensor in the checkpoint, without reading every shard."""
    file = _checkpoint_files(path)[0]
    if file.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(file), framework="pt") as handle:
            view = handle.get_slice(next(iter(handle.keys())))
            dtype = getattr(view, "dtype", None)
            if dtype is not None:
                return dtype
    blob = torch.load(file, map_location="cpu", weights_only=True)
    return next(iter(blob.values())).dtype


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
    if (
        key.startswith("visual.")
        or key.startswith("model.visual.")
        or key.startswith("mtp.")
        or ".mtp." in key
    ):
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
    # 2D projection and embedding matrices. Conv kernels stay BF16; the
    # depthwise conv reads them directly and the FP8 kernel only covers linear.
    if getattr(module, "quantize_fp8", False) and tensor.ndim == 2 and name.endswith("weight"):
        from .static_fp8 import install_quantized_weight

        install_quantized_weight(module, name, tensor)
        return
    with torch.no_grad():
        param.copy_(tensor.to(dtype=param.dtype))


def _iter_tensors(path: str | os.PathLike):
    """Yield checkpoint tensors one at a time so a 27B load does not keep every shard."""
    for file in _checkpoint_files(path):
        if file.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    yield key, handle.get_tensor(key)
        else:
            blob = torch.load(file, map_location="cpu", weights_only=True)
            yield from blob.items()


def _place(model, key: str, tensor: torch.Tensor) -> str | None:
    """Shard one text tensor into ``model``. Returns the parameter name, or None to skip."""
    config: Qwen38TextConfig = model.config
    rank, world = model.tp_rank, model.tp_size
    if key in ("embed_tokens.weight", "lm_head.weight", "norm.weight"):
        _assign(model, key, tensor)
        return key
    parts = key.split(".")
    if len(parts) < 4 or parts[0] != "layers":
        return None
    layer = int(parts[1])
    if layer >= config.num_hidden_layers:
        return None
    kind = parts[2]
    name = ".".join(parts[3:])
    num_k = config.linear_num_key_heads
    num_v = config.linear_num_value_heads
    head_k = config.linear_key_head_dim
    head_v = config.linear_value_head_dim
    if kind in ("input_layernorm", "post_attention_layernorm") and name == "weight":
        dest = f"layers.{layer}.{kind}.weight"
        _assign(model, dest, tensor)
        return dest
    if kind == "mlp":
        dest = f"layers.{layer}.mlp.{name}"
        if name in ("gate_proj.weight", "up_proj.weight"):
            tensor = _shard(tensor, rank, world, tensor.shape[0])
        elif name == "down_proj.weight":
            local = tensor.shape[1] // world
            tensor = tensor[:, rank * local : (rank + 1) * local]
        else:
            return None
        _assign(model, dest, tensor)
        return dest
    if kind == "linear_attn":
        dest = f"layers.{layer}.mixer.{name}"
        if name in ("in_proj_qkv.weight", "conv1d.weight"):
            tensor = _shard_qkv(tensor, rank, world, num_k, num_v, head_k, head_v)
        elif name == "in_proj_z.weight":
            tensor = _shard(tensor.reshape(num_v, head_v, -1), rank, world, num_v).reshape(-1, tensor.shape[-1])
        elif name in ("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias"):
            tensor = _shard(tensor, rank, world, num_v)
        elif name == "out_proj.weight":
            tensor = _shard_out(tensor, rank, world, num_v, head_v)
        elif name != "norm.weight":
            return None
        _assign(model, dest, tensor)
        return dest
    if kind == "self_attn":
        heads = config.num_attention_heads
        kv = config.num_key_value_heads
        head = config.head_dim
        if name == "q_proj.weight":
            tensor = _shard(tensor.reshape(heads, 2 * head, -1), rank, world, heads).reshape(-1, tensor.shape[-1])
            dest = f"layers.{layer}.attn.q_proj.weight"
        elif name == "k_proj.weight":
            tensor = _shard(tensor.reshape(kv, head, -1), rank, world, kv).reshape(-1, tensor.shape[-1])
            dest = f"layers.{layer}.attn.k_proj.weight"
        elif name == "v_proj.weight":
            tensor = _shard(tensor.reshape(kv, head, -1), rank, world, kv).reshape(-1, tensor.shape[-1])
            dest = f"layers.{layer}.attn.v_proj.weight"
        elif name == "o_proj.weight":
            hidden = tensor.shape[0]
            tensor = _shard(tensor.reshape(hidden, heads, head).transpose(0, 1), rank, world, heads)
            tensor = tensor.transpose(0, 1).reshape(hidden, -1)
            dest = f"layers.{layer}.attn.o_proj.weight"
        elif name == "q_norm.weight":
            dest = f"layers.{layer}.attn.q_norm"
        elif name == "k_norm.weight":
            dest = f"layers.{layer}.attn.k_norm"
        else:
            return None
        _assign(model, dest, tensor)
        return dest
    return None


def load_text_weights(model, path: str | os.PathLike) -> list[str]:
    """Copy text weights into ``model``. Returns the parameter names that were set."""
    loaded: list[str] = []
    for key, tensor in _iter_tensors(path):
        stripped = _strip(key)
        if stripped is None or stripped.startswith("visual"):
            continue
        dest = _place(model, stripped, tensor)
        del tensor
        if dest is not None:
            loaded.append(dest)
    missing = [name for name, _ in model.named_parameters() if name not in loaded]
    if missing:
        raise KeyError("checkpoint did not fill: " + ", ".join(missing))
    if getattr(model, "quantize_fp8", False):
        fp8_bytes = 0
        other_bytes = 0
        for param in model.parameters():
            nbytes = param.numel() * param.element_size()
            if param.dtype == torch.float8_e4m3fn:
                fp8_bytes += nbytes
            else:
                other_bytes += nbytes
        print(
            f"static FP8 weights: {fp8_bytes / (1024 ** 3):.2f} GiB fp8, "
            f"{other_bytes / (1024 ** 3):.2f} GiB other parameters",
            flush=True,
        )
    return loaded


def config_from_checkpoint(path: str | os.PathLike) -> dict:
    root = Path(path)
    config_path = root / "config.json" if root.is_dir() else None
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"no config.json next to {path}")
    with open(config_path) as handle:
        return json.load(handle)
