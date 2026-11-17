# Contrib model: Qwen3.8-27B (text decoder, in progress)

Checkpoint: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Architecture
string: `Qwen3_5ForConditionalGeneration`. This is not `Qwen3ForCausalLM`.

The language model repeats 16 times: three Gated DeltaNet layers, then one gated
full-attention layer (24 query heads, 4 KV heads, head dim 256, partial RoPE of
64 dims, sigmoid output gate). Vision and the MTP head are not in this slice.

## What is implemented

- Config adapter (`src/config.py`) for the text geometry.
- Gated DeltaNet (`src/gated_delta.py`). Prefill and a later chunk share one
  recurrence. Checked against Transformers `Qwen3_5GatedDeltaNet` on CPU.
- One fixed state per live request (`src/slots.py`). Prefix caching is off.
  The runner key is the request id or, from attention metadata, the first KV
  block id. A request that leaves the batch releases and zeros its slot.
  Value heads are sharded for tensor parallel.
- Text runner (`src/model.py`): embedding, hybrid layers, paged KV for the
  full-attention layers, LM head. `from_configs` builds it.
- Checkpoint load (`src/weights.py`). `load_weights(path, device, cache_dir)`
  is the Neuron worker entry. It reads the Qwen3.8 text keys
  (`model.language_model.*`, `lm_head.weight`), skips `model.visual.*` and
  `mtp.*`, and shards projections for the local tensor-parallel rank.
  Embeddings, norms, and the LM head stay replicated.
- Plugin registration (`src/__init__.py`) so vLLM resolves
  `Qwen3_5ForConditionalGeneration` to this package.

Tensor-parallel outputs of attention, DeltaNet, and the MLP are partial sums.
There is no allreduce yet, so a multi-rank forward is not a full model.

## What this is not

This port targets the vLLM Neuron 0.29 plugin (`torch_neuronx` and
`vllm_neuron.functional`). It does not go through NxD Inference.

Full attention cannot use the in-tree NKI attention kernels: those cap head dim
at 128. The full-attention layers are PyTorch over the paged KV cache.

On this `trn2.3xlarge` (4 cores, 96 GB, logical NeuronCore config 2) BF16 weights
need tensor parallel 4. Tensor parallel 2 puts about 27 GB on a 24 GB core.

## Checks

```bash
pip install -e contrib/models/qwen3_8
python contrib/models/qwen3_8/test/test_gated_delta_ref.py
```

`test/test_runner.py` compares a tiny text model with Transformers and checks
chunked prefill, slot reuse, and the value-head shard. `test/test_load_weights.py`
saves a real `Qwen3_5ForCausalLM` checkpoint (with the `model.language_model.`
prefix), reloads it, and checks greedy prefill and one decode step against
Transformers. The Qwen3.8-27B weight index uses those same text key names.
The 27B tensors themselves are not loaded in that test.

A 64-layer tiny model (TP=4 shapes on one rank, prefix caching off, one
sequence) compiles and runs on this instance: prefill then decode keep the
same DeltaNet slot, and the prefill NEFF contains 64 paged-KV scatters covering
the 16 full-attention layers. `vllm serve` of the 27B checkpoint is not started.
Rank outputs still need an allreduce, and that short compile used tiny weights,
not the BF16 checkpoint.
