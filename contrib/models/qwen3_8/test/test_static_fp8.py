# SPDX-License-Identifier: Apache-2.0
"""Numeric error and byte count of Qwen3.8 static FP8 weights."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from static_fp8 import (  # noqa: E402
    downscale_for_trn2,
    quantize_per_tensor,
    reference_linear,
    static_fp8_gemm,
)


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.float().norm().clamp(min=1e-12)
    return float((got.float() - ref.float()).norm() / denom)


def _max_rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.float().abs().clamp(min=1e-6)
    return float(((got.float() - ref.float()).abs() / denom).max())


def test_weight_roundtrip_and_linear_error():
    torch.manual_seed(0)
    weight = torch.randn(256, 128, dtype=torch.bfloat16)
    activation = torch.randn(32, 128, dtype=torch.bfloat16)
    quantized, scale = quantize_per_tensor(weight)
    restored = quantized.float() * scale
    weight_rel = _rel(restored, weight.float())
    got = reference_linear(activation, quantized, scale)
    ref = torch.nn.functional.linear(activation.float(), weight.float())
    linear_rel = _rel(got, ref)
    print(
        f"weight_rel={weight_rel:.4e} weight_max_rel={_max_rel(restored, weight.float()):.4e} "
        f"linear_rel={linear_rel:.4e} linear_max_abs={(got.float() - ref).abs().max().item():.4e} "
        f"linear_rms={ref.float().std().item():.4e}"
    )
    # E4M3 has a 3-bit mantissa, so a few percent of L2 error is the format.
    # A dropped or doubled scale would land near 1.
    if weight_rel > 5e-2 or linear_rel > 5e-2:
        raise SystemExit(f"FP8 error too large: weight {weight_rel} linear {linear_rel}")

    down_w, down_s = downscale_for_trn2(quantized, scale)
    down_rel = _rel(down_w.float() * down_s, weight.float())
    print(f"trn2_downscale_rel={down_rel:.4e}")
    if down_rel > 5e-2:
        raise SystemExit(f"Trn2 downscale moved the weight by {down_rel}")


def _check_kernel(m: int, n: int, k: int, seed: int) -> None:
    import nki

    torch.manual_seed(seed)
    weight = torch.randn(n, k)
    activation = torch.randn(m, k)
    quantized, scale = quantize_per_tensor(weight)
    expected = reference_linear(activation, quantized, scale).float()
    got = nki.simulate(static_fp8_gemm)(activation.float(), quantized, scale)
    got = got.transpose(0, 1)
    rel = _rel(got, expected)
    print(f"kernel m={m} n={n} k={k} rel={rel:.4e}")
    if rel > 5e-2:
        raise SystemExit(f"kernel mismatch rel={rel}")


def test_kernel_matches_reference():
    _check_kernel(128, 128, 128, 1)
    _check_kernel(128, 256, 256, 2)
    _check_kernel(256, 128, 256, 3)
    _check_kernel(1, 128, 256, 4)
    _check_kernel(1, 12, 128, 5)
    _check_kernel(32, 12, 256, 6)


def test_model_logits_and_bytes():
    from qwen3_8_neuron.config import Qwen38TextConfig
    from qwen3_8_neuron.model import Qwen38ForCausalLM
    from vllm_neuron.model.neuron_config import NeuronConfig

    torch.manual_seed(2)
    cfg = dict(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    plain = Qwen38ForCausalLM(Qwen38TextConfig.from_configs(cfg)).bfloat16().eval()
    fp8_model = Qwen38ForCausalLM(
        Qwen38TextConfig.from_configs(cfg, NeuronConfig(quantization="fp8"))
    ).eval()
    bf16_bytes = 0
    fp8_bytes = 0
    with torch.no_grad():
        for name, param in plain.named_parameters():
            src = torch.randn(param.shape, dtype=torch.bfloat16)
            param.copy_(src)
            target = fp8_model.get_parameter(name)
            if target.ndim == 2 and name.endswith(".weight"):
                from static_fp8 import install_quantized_weight

                install_quantized_weight(fp8_model, name, src)
            else:
                target.copy_(src.to(dtype=target.dtype))
    for param in plain.parameters():
        bf16_bytes += param.numel() * param.element_size()
    for param in fp8_model.parameters():
        fp8_bytes += param.numel() * param.element_size()
    ids = torch.randint(0, 128, (8,))
    with torch.no_grad():
        ref = plain(ids, torch.arange(8), req_ids=["req"]).float()
        got = fp8_model(ids, torch.arange(8), req_ids=["req"]).float()
    rel = _rel(got, ref)
    twod_bf16 = 0
    twod_fp8 = 0
    for name, param in plain.named_parameters():
        if param.ndim == 2 and name.endswith(".weight"):
            twod_bf16 += param.numel() * param.element_size()
    for name, param in fp8_model.named_parameters():
        if param.ndim == 2 and name.endswith(".weight"):
            twod_fp8 += param.numel() * param.element_size()
    agree = int((got.argmax(-1) == ref.argmax(-1)).sum())
    print(
        f"logit_rel={rel:.4e} argmax_agree={agree}/{got.shape[0]} "
        f"bytes_bf16={bf16_bytes} bytes_fp8={fp8_bytes} "
        f"matrix_bytes_bf16={twod_bf16} matrix_bytes_fp8={twod_fp8} "
        f"matrix_ratio={twod_fp8 / twod_bf16:.3f}"
    )
    if twod_fp8 * 2 > twod_bf16:
        raise SystemExit("FP8 matrices did not halve")
    if rel > 2.5e-1:
        raise SystemExit(f"logit error too large: {rel}")


if __name__ == "__main__":
    test_weight_roundtrip_and_linear_error()
    test_model_logits_and_bytes()
    test_kernel_matches_reference()
    print("OK")
