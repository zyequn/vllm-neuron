# SPDX-License-Identifier: Apache-2.0
"""Compile Qwen3.8 pieces on Neuron at seq_len 64.

The 64-layer serve graph spends a long time inside one neuronx-cc job. These
tests compile the same ops as separate graphs, then a 2-layer model, so a
failure points at one piece instead of the whole stack.

Ladder, smallest first:

1. BF16 linear. Proves the native backend and compiler flags work.
2. Static FP8 GEMM at M=64. This is the remainder tile a seq-64 projection hits.
3. Gated DeltaNet layer. The recurrence is one NKI call, not an unrolled loop.
4. One full-attention layer, including the paged KV read and write.
5. Two layers (DeltaNet then full attention), embed, and LM head.
6. Eight DeltaNet layers in one graph, without ``modular-flow-mac-threshold``.
   With that flag, two or more NKI kernels become a NEFF NRT cannot load.
7. The real ``Qwen38ForCausalLM`` prefill, seq 64 and seq 1536, with a bound
   paged cache. DeltaNet state has to stay a module buffer after ``.to()``.
8. Eight layers (six DeltaNet kernels) at seq 64 and seq 1536, flag off.
9. All 64 layers at seq 1536, flag off. Forty-eight DeltaNet kernels, one graph.

Run on Trainium:

    pytest contrib/models/qwen3_8/test/test_compile_subgraphs.py -v -s --timeout=1800
"""

from __future__ import annotations

import copy
import os
import time

# One physical core, pinned before the Neuron runtime initializes. A NEFF that
# contains a collective will not load unless the root comm id is set. The serve
# path uses logical-nc-config 2; these tests first ask whether the op compiles.
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "1"
os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:61234")

import pytest
import torch
from torch import nn

pytest.importorskip("libtorch_neuronx_lite")

from qwen3_8_neuron.attention import Qwen38Attention
from qwen3_8_neuron.config import Qwen38TextConfig
from qwen3_8_neuron.gated_delta import Qwen38GatedDeltaNet
from qwen3_8_neuron.model import Qwen38DecoderLayer, Qwen38ForCausalLM, Qwen38MLP
from qwen3_8_neuron.static_fp8 import (
    downscale_for_trn2,
    needs_trn2_downscale,
    quantize_per_tensor,
    static_fp8_linear,
)

SEQ = 64
HIDDEN = 128
INTERMEDIATE = 256
HEADS = 2
KV_HEADS = 2
HEAD_DIM = 128
ROTARY = 32
BLOCK = 16
N_BLOCKS = SEQ // BLOCK

# Same flags the runner passes for an explicit -O1 compile. The platform helper
# appends --logical-nc-config and, on trn2, the legacy FP8 cast.
COMPILER_ARGS = [
    "--auto-cast=none",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
]

pytestmark = pytest.mark.skipif(
    not os.path.exists("/dev/neuron0"),
    reason="Neuron device is not present",
)


def _layer_types(n_layers: int) -> list[str]:
    """Qwen3.8 repeats three DeltaNet layers and one full-attention layer.

    The 2-layer ladder keeps one of each so the smallest graph stays one
    DeltaNet kernel plus one attention layer.
    """
    if n_layers == 2:
        return ["linear_attention", "full_attention"]
    return [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(n_layers)
    ]


def _small_config(n_layers: int = 2) -> Qwen38TextConfig:
    return Qwen38TextConfig.from_configs(
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "vocab_size": 128,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": n_layers,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "partial_rotary_factor": ROTARY / HEAD_DIM,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": ROTARY / HEAD_DIM,
            },
            "rms_norm_eps": 1e-6,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": _layer_types(n_layers),
            "tie_word_embeddings": False,
        }
    )


def _init(module: nn.Module, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    for parameter in module.parameters():
        if parameter.dtype.is_floating_point:
            parameter.data.normal_(mean=0.0, std=0.02)
    return module


def _move_args(args, device):
    moved = []
    for value in args:
        if torch.is_tensor(value):
            moved.append(value.to(device))
        else:
            moved.append(value)
    return tuple(moved)


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.float().norm().clamp(min=1e-6)
    return float((got.float().cpu() - ref.float().cpu()).norm() / denom)


def _compile_forward(module: nn.Module, args, compiler_args=None):
    """Compile ``module`` and run one forward. Returns neuron outputs on CPU."""
    from vllm_neuron.envs import get_compile_backend_name

    device = torch.device("neuron:0")
    module = module.to(device).eval()
    compiled = torch.compile(
        module,
        backend=get_compile_backend_name(),
        fullgraph=True,
        options={
            "alias_meta_to_neuron": True,
            "compiler_args": list(compiler_args or COMPILER_ARGS),
        },
    )
    neuron_args = _move_args(args, device)
    started = time.perf_counter()
    with torch.no_grad():
        output = compiled(*neuron_args)
    elapsed = time.perf_counter() - started
    print(f"compile+run {type(module).__name__}: {elapsed:.1f}s")
    return _to_cpu(output), elapsed


def _assert_close(name: str, got, ref, limit: float) -> None:
    pairs = list(zip(_leaves(got), _leaves(ref), strict=True))
    assert pairs, name
    for index, (left, right) in enumerate(pairs):
        assert left.shape == right.shape, (name, index, left.shape, right.shape)
        assert torch.isfinite(left).all(), name
        error = _rel(left, right)
        print(f"{name}[{index}] rel={error:.4e} shape={tuple(left.shape)}")
        assert error < limit, f"{name}[{index}] rel {error} >= {limit}"


class _Bf16Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class _Fp8Gemm(nn.Module):
    """One static-FP8 projection. M=64 exercises the short-tile path."""

    def __init__(self, n_rows: int, k_dim: int):
        super().__init__()
        weight = torch.randn(n_rows, k_dim)
        quantized, scale = quantize_per_tensor(weight)
        if needs_trn2_downscale():
            quantized, scale = downscale_for_trn2(quantized, scale)
        self.weight = nn.Parameter(quantized, requires_grad=False)
        self.register_buffer("weight_scale", scale.float().reshape(1, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return static_fp8_linear(hidden, self.weight, self.weight_scale)


class _DeltaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Qwen38GatedDeltaNet(HIDDEN, 1, 2, 128, 128, 4)

    def forward(self, hidden: torch.Tensor):
        # [S, H] -> [1, S, H], matching a prefill that starts from a zero state.
        out, new_conv, new_state = self.layer(hidden.unsqueeze(0), None, None)
        return out.squeeze(0), new_conv, new_state


class _AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Qwen38Attention(
            HIDDEN, HEADS, KV_HEADS, HEAD_DIM, ROTARY, 10000.0, 1e-6
        )

    def forward(self, hidden, positions, k_cache, v_cache, block_table, slots):
        out = self.attn.forward_span(
            hidden, positions, k_cache, v_cache, block_table, slots, BLOCK, 0
        )
        return out, k_cache, v_cache


class _TwoLayerPrefill(nn.Module):
    """Embed, one DeltaNet layer, one full-attention layer, LM head. Seq is 64."""

    def __init__(self):
        super().__init__()
        config = _small_config()
        self.model = Qwen38ForCausalLM(config, max_num_seqs=1, tp_size=1, tp_rank=0)

    def forward(self, input_ids, positions, k_cache, v_cache, block_table, slots):
        hidden = self.model.embed_tokens(input_ids)
        hidden, new_conv, new_state = self.model.layers[0].forward_linear(
            hidden, None, None
        )
        hidden = self.model.layers[1].forward_attn(
            hidden,
            positions,
            k_cache,
            v_cache,
            block_table,
            slots,
            BLOCK,
            0,
        )
        logits = self.model.lm_head(self.model.norm(hidden))
        return logits, new_conv, new_state, k_cache, v_cache


def _attention_inputs(dtype=torch.bfloat16):
    torch.manual_seed(4)
    hidden = torch.randn(SEQ, HIDDEN, dtype=dtype)
    positions = torch.arange(SEQ)
    k_cache = torch.zeros(N_BLOCKS, KV_HEADS, BLOCK, HEAD_DIM, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.arange(N_BLOCKS)
    slots = torch.arange(SEQ, dtype=torch.int32)
    return hidden, positions, k_cache, v_cache, block_table, slots


def _two_layer_inputs():
    torch.manual_seed(5)
    input_ids = torch.randint(0, 128, (SEQ,))
    positions = torch.arange(SEQ)
    k_cache = torch.zeros(N_BLOCKS, KV_HEADS, BLOCK, HEAD_DIM, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.arange(N_BLOCKS)
    slots = torch.arange(SEQ, dtype=torch.int32)
    return input_ids, positions, k_cache, v_cache, block_table, slots


def _cpu_forward(module: nn.Module, args):
    cloned = copy.deepcopy(module).eval()
    with torch.no_grad():
        return cloned(*args)


def test_compile_bf16_linear_seq64():
    module = _init(_Bf16Linear(), 1).to(torch.bfloat16)
    hidden = torch.randn(SEQ, HIDDEN, dtype=torch.bfloat16)
    ref = _cpu_forward(module, (hidden,))
    got, _ = _compile_forward(module, (hidden,))
    _assert_close("bf16_linear", got, ref, 2e-2)


def test_compile_fp8_gemm_seq64():
    module = _Fp8Gemm(INTERMEDIATE, HIDDEN).eval()
    hidden = torch.randn(SEQ, HIDDEN)
    ref = _cpu_forward(module, (hidden,))
    got, _ = _compile_forward(module, (hidden,))
    _assert_close("fp8_gemm", got, ref, 5e-2)


def test_compile_gated_delta_seq64():
    module = _init(_DeltaLayer(), 2).to(torch.bfloat16)
    hidden = torch.randn(SEQ, HIDDEN, dtype=torch.bfloat16)
    ref = _cpu_forward(module, (hidden,))
    got, _ = _compile_forward(module, (hidden,))
    # The Neuron path is the NKI recurrence. A few percent is the kernel tolerance.
    _assert_close("gated_delta", got, ref, 5e-2)


def test_compile_attention_seq64():
    module = _init(_AttentionLayer(), 3).to(torch.bfloat16)
    args = _attention_inputs()
    ref = _cpu_forward(module, args)
    got, _ = _compile_forward(module, args)
    _assert_close("attention", got, ref, 2e-2)


def test_compile_two_layer_seq64():
    module = _init(_TwoLayerPrefill(), 6).to(torch.bfloat16)
    args = _two_layer_inputs()
    ref = _cpu_forward(module, args)
    got, _ = _compile_forward(module, args)
    _assert_close("two_layer", got, ref, 5e-2)


def test_compile_full_width_mlp_seq64():
    """One MLP at the 27B widths. Separates tensor size from layer count."""
    module = _init(
        Qwen38MLP(5120, 17408, tp_size=1, tp_rank=0), 7
    ).to(torch.bfloat16)
    hidden = torch.randn(SEQ, 5120, dtype=torch.bfloat16)
    ref = _cpu_forward(module, (hidden,))
    got, _ = _compile_forward(module, (hidden,))
    _assert_close("full_width_mlp", got, ref, 2e-2)


def test_compile_full_width_delta_seq64():
    """DeltaNet at 16 QK heads and 48 V heads. The NKI body is unrolled per head."""
    module = _init(Qwen38GatedDeltaNet(5120, 16, 48, 128, 128, 4), 8).to(torch.bfloat16)

    class _Wrap(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, hidden):
            out, new_conv, new_state = self.layer(hidden.unsqueeze(0), None, None)
            return out.squeeze(0), new_conv, new_state

    wrapped = _Wrap(module)
    hidden = torch.randn(SEQ, 5120, dtype=torch.bfloat16)
    ref = _cpu_forward(wrapped, (hidden,))
    got, _ = _compile_forward(wrapped, (hidden,))
    _assert_close("full_width_delta", got, ref, 5e-2)


def test_compile_full_width_attention_seq64():
    """Full attention at 24 query heads, 4 KV heads, head dim 256."""
    module = _init(
        Qwen38Attention(5120, 24, 4, 256, 64, 10_000_000.0, 1e-6), 9
    ).to(torch.bfloat16)

    class _Wrap(nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.attn = attn

        def forward(self, hidden, positions, k_cache, v_cache, block_table, slots):
            out = self.attn.forward_span(
                hidden, positions, k_cache, v_cache, block_table, slots, BLOCK, 0
            )
            return out, k_cache, v_cache

    wrapped = _Wrap(module)
    torch.manual_seed(10)
    hidden = torch.randn(SEQ, 5120, dtype=torch.bfloat16)
    positions = torch.arange(SEQ)
    k_cache = torch.zeros(N_BLOCKS, 4, BLOCK, 256, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.arange(N_BLOCKS)
    slots = torch.arange(SEQ, dtype=torch.int32)
    args = (hidden, positions, k_cache, v_cache, block_table, slots)
    ref = _cpu_forward(wrapped, args)
    got, _ = _compile_forward(wrapped, args)
    _assert_close("full_width_attention", got, ref, 2e-2)


class _DeltaStack(nn.Module):
    """Several DeltaNet layers in one graph. Used to see how compile time grows."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [Qwen38GatedDeltaNet(HIDDEN, 1, 2, 128, 128, 4) for _ in range(n_layers)]
        )

    def forward(self, hidden: torch.Tensor):
        states = []
        for layer in self.layers:
            hidden, _, state = layer(hidden.unsqueeze(0), None, None)
            hidden = hidden.squeeze(0)
            states.append(state)
        return hidden, states


# The runner passes --modular-flow-mac-threshold=10. One NKI kernel still
# loads. Two or more in the same graph are split, and NRT then refuses the
# NEFF: "Unable to find output tensor intermediate0_sg0000". Leaving the
# threshold off keeps a single subgraph, which loads.
_NO_SPLIT_ARGS = [
    "--auto-cast=none",
    "-O1",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
]


def test_compile_eight_delta_layers_seq64():
    module = _init(_DeltaStack(8), 11).to(torch.bfloat16)
    hidden = torch.randn(SEQ, HIDDEN, dtype=torch.bfloat16)
    ref = _cpu_forward(module, (hidden,))
    got, _ = _compile_forward(module, (hidden,), _NO_SPLIT_ARGS)
    _assert_close("eight_delta", got, ref, 5e-2)


class _ServePrefill(nn.Module):
    """The real forward, with the metadata shape ``vllm serve`` traces.

    The private cache holds 512 tokens. A 1536-token prefill uses a bound
    paged cache, which is what the runner does. Each full-attention layer
    gets its own cache. One metadata record is shared, matching a single
    request.
    """

    def __init__(self, seq: int, n_layers: int = 2):
        super().__init__()
        self.seq = seq
        self.model = Qwen38ForCausalLM(
            _small_config(n_layers), max_num_seqs=1, tp_size=1, tp_rank=0
        )
        blocks = seq // BLOCK
        self.attn_names = list(self.model._attn_names)
        for index, name in enumerate(self.attn_names):
            # The runner key is ``layers.N.self_attn``. The module attribute is ``attn``.
            layer_idx = int(name.split(".")[1])
            attn = self.model.layers[layer_idx].attn
            cache = torch.zeros(blocks, attn.num_kv_heads, BLOCK, attn.head_dim)
            self.register_buffer(f"k_cache_{index}", cache)
            self.register_buffer(f"v_cache_{index}", torch.zeros_like(cache))
        self.register_buffer("block_table", torch.arange(blocks).view(1, -1))
        self.register_buffer("slots", torch.arange(seq, dtype=torch.int32))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        self.model.bind_kv_cache(
            {
                name: [
                    getattr(self, f"k_cache_{index}"),
                    getattr(self, f"v_cache_{index}"),
                ]
                for index, name in enumerate(self.attn_names)
            }
        )
        meta = {
            "block_table_tensor": self.block_table,
            "slot_mapping": self.slots,
            "block_size": BLOCK,
            "max_query_len": self.seq,
            "decode_token_threshold": 1,
        }
        return self.model(
            input_ids, positions, attn_metadata={self.attn_names[0]: meta}
        )


def test_compile_serve_prefill_seq64():
    """Same real forward as the 1536 case, at a length already known to compile."""
    seq = 64
    module = _init(_ServePrefill(seq), 15).to(torch.bfloat16)
    torch.manual_seed(16)
    input_ids = torch.randint(0, 128, (seq,))
    positions = torch.arange(seq)
    ref = _cpu_forward(module, (input_ids, positions))
    got, _ = _compile_forward(module, (input_ids, positions), _NO_SPLIT_ARGS)
    _assert_close("serve_prefill_64", got, ref, 5e-2)


def test_compile_serve_prefill_seq1536():
    """Two real layers at seq 1536, one graph, no modular split."""
    seq = 1536
    module = _init(_ServePrefill(seq), 13).to(torch.bfloat16)
    torch.manual_seed(14)
    input_ids = torch.randint(0, 128, (seq,))
    positions = torch.arange(seq)
    ref = _cpu_forward(module, (input_ids, positions))
    got, _ = _compile_forward(module, (input_ids, positions), _NO_SPLIT_ARGS)
    _assert_close("serve_prefill_1536", got, ref, 5e-2)


def _check_prefill(seq: int, n_layers: int, seed: int, label: str) -> None:
    """Real prefill, flag off. Several DeltaNet kernels share one graph."""
    module = _init(_ServePrefill(seq, n_layers), seed).to(torch.bfloat16)
    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, 128, (seq,))
    positions = torch.arange(seq)
    ref = _cpu_forward(module, (input_ids, positions))
    got, _ = _compile_forward(module, (input_ids, positions), _NO_SPLIT_ARGS)
    _assert_close(label, got, ref, 5e-2)


def test_compile_eight_layer_prefill_seq64():
    """Eight layers, six DeltaNet kernels, seq 64, no modular split."""
    _check_prefill(64, 8, 21, "eight_layer_64")


def test_compile_eight_layer_prefill_seq1536():
    """Same eight-layer graph at seq 1536."""
    _check_prefill(1536, 8, 22, "eight_layer_1536")


def test_compile_full_model_prefill_seq1536():
    """All 64 layers at seq 1536. Forty-eight DeltaNet kernels, flag off."""
    _check_prefill(1536, 64, 23, "full_64_1536")


def test_decoder_layer_types_match_two_layer_config():
    """The 2-layer config is one DeltaNet layer and one full-attention layer."""
    config = _small_config()
    linear = Qwen38DecoderLayer(config, 0, tp_size=1, tp_rank=0)
    full = Qwen38DecoderLayer(config, 1, tp_size=1, tp_rank=0)
    assert linear.is_linear and linear.attn is None
    assert not full.is_linear and full.mixer is None
    assert config.rotary_dim == ROTARY
