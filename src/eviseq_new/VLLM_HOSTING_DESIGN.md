# Hosting AFMR with vLLM 0.25.1

## Decision

Use an out-of-tree vLLM plugin first, targeting vLLM 0.25.1 and its V2 model
runner. Keep the AFMR checkpoint semantics intact: encoder, bridge, decoder
cross-attention, source bias, and the grounded-copy probability mixture all
remain active. Do not silently fall back to decoder-only generation when any
of these pieces is unavailable.

The plugin should be config-driven. It should read the resolved AFMR config
and an exported checkpoint bundle rather than hard-code one encoder/decoder
pair. A decoder-family registry is still required because Qwen, Llama, and
other decoder families have different vLLM layer implementations. Initial
adapters should be added for the families actually used in experiments; an
unknown family must fail with a clear compatibility error.

“General” here means one AFMR serving package with interchangeable, validated
backbone adapters, not one universal implementation for every Transformers
model. The existing encoder contract already covers many HF encoders; model
families with non-standard outputs (such as EmbeddingGemma-2 or Qwen3.5) need
their corresponding encoder adapter. The decoder must be a causal model whose
vLLM layer implementation can be extended at the trained cross-attention
insertion point. Diffusion decoders need a separate generation adapter and
should not be treated as causal decoders by default.

T5Gemma 2 is a separate serving target, not an AFMR decoder-family adapter.
It is already an encoder-decoder model with its own weight layout and decoding
path. The repository therefore provides a separate launcher for its dedicated
vLLM plugin rather than routing it through AFMR's causal-decoder implementation.
The upstream plugin was developed against a vLLM nightly commit, so compatibility
with the user's vLLM 0.25.1 installation still needs a server-side smoke test.

## vLLM integration points verified in 0.25.1

- vLLM supports registering an external architecture through the
  `vllm.general_plugins` entry point and `ModelRegistry.register_model`.
- Encoder-decoder models use the V2 runner's `EncoderDecoderModelState` and
  paged cross-attention KV cache. A model can provide `get_model_state_cls()`
  to supply request-specific state handling.
- A multimodal processor can carry source text separately from the decoder
  prompt. This lets the plugin tokenize source and decoder prompts with their
  own tokenizers.
- The BART plugin is a useful reference for the vLLM 0.25.x encoder-decoder
  plugin pattern, but its model implementation cannot be reused as AFMR:
  AFMR has a separate pretrained encoder, a prompt-conditioned bridge, source
  bias, and grounded copy.

## Preserving the trained computation

The encoder adapter should load the configured encoder locally, return the
final token states and requested depth taps, and preserve the source content
mask. The shared AFMR bridge then consumes those states together with the
decoder prompt embeddings and generation budget. The decoder adapter should
insert the trained cross-attention at the same point in each decoder block as
the training implementation.

The source bias is an additive term in cross-attention scores. It can be
represented exactly in a paged attention key by augmenting each per-head key
and query with one coordinate: with the original head size `d` and scale
`1/sqrt(d)`, append `sqrt(d) * bias` to each key and `1` to each query. Then
the added dot-product term is exactly `bias`. The vLLM adapter pads the
augmented head width to the next multiple of eight for backend alignment; all
extra coordinates except the bias coordinate are zero, and the value/output
width remains `d`. The chosen GPU backend still needs a smoke test for that
padded head size.

Grounded copy needs its own per-request state: copy keys, aligned decoder token
IDs, validity mask, and source bias. A custom model state should create this
state when the encoder input is processed, pass it to each decode step in the
current request order, and release it in `remove_request`. The model's
`compute_logits` must apply the same normalized generation/copy mixture as
`GroundedCopyHead.mix_logits` before vLLM sampling. If request ordering,
chunked prefill, or sampling cannot be mapped correctly, the plugin must fail
closed rather than emit logits from the generation-only branch.

## Adapter boundary

Keep the AFMR bridge and copy implementation shared. Put family-specific code
behind small adapters for:

1. encoder loading, hidden-state taps, and tokenizer behavior;
2. decoder block construction, self-attention cache, and checkpoint weight
   mapping;
3. tokenizer/prompt processing for the encoder and decoder.

The first production targets should be Qwen3 and Llama 3 decoders, with encoder
adapters for the encoder families already present in this project (PPLX, Qwen
embedding/Qwen3.5, and EmbeddingGemma-2). New families should be added with a
parity test before being marked supported. A config that combines an
unimplemented family or incompatible hidden-state contract should be rejected
at startup.

## Required validation before product use

1. **Weight parity:** every trainable AFMR checkpoint tensor is mapped, with
   no missing or unexpected keys except explicitly documented buffers.
2. **Logit parity:** compare next-token logits against the existing PyTorch
   AFMR implementation on fixed examples, including copy-enabled examples.
3. **Decode parity:** greedy token IDs match on a tiny fixture and a real
   checkpoint, subject only to documented dtype/backend tolerances.
4. **Batch-state safety:** test mixed source lengths, requests entering and
   finishing in different orders, chunked prefill, and state cleanup/reuse.
5. **Quality parity:** run the same test inputs and generation settings through
   both runtimes; compare ROUGE and the existing source-alignment metric.
6. **Performance:** report warm and cold load time, time to first token,
   inter-token latency, aggregate tokens/s at several concurrent request
   counts, and peak VRAM. Compare the same checkpoint, GPU, dtype, and prompt
   lengths.

## Current implementation status

An initial Qwen3 AFMR adapter is now implemented in `seam_vllm/`. It registers
`SEAMForQwen3`, uses the vLLM encoder-decoder `ModelState`, and exposes the
in-process `SEAMLLM(config_path, checkpoint_path).summarize(sources)` API.
That adapter keeps the HF encoder and shared AFMR bridge, uses vLLM paged
cross-attention for the trained decoder cross route, and applies grounded-copy
mixture logits before sampling. It currently rejects tensor/pipeline parallel
sizes above one and requires eager execution for request-side copy state.

This implementation has not been run against an installed vLLM runtime or a
GPU in this Mac checkout: the `bienkieu_env` environment is absent. Checkpoint
loading, encoder/copy batching, logit parity, greedy decode parity, and
throughput therefore still require a server-side smoke test. The plugin does
not currently register Llama, Gemma, Nemotron, diffusion, or other decoder
families; each needs its own decoder-block adapter and parity check.

T5Gemma 2 remains a separate serving target. Its install/serve scripts are
`scripts/install_t5gemma2_vllm_plugin.sh` and
`scripts/serve_t5gemma2_vllm.sh`. The community plugin was developed against a
vLLM nightly commit, so compatibility with vLLM 0.25.1 is unverified.

## References

- [vLLM model registration](https://github.com/vllm-project/vllm/blob/v0.25.1/docs/contributing/model/registration.md)
- [vLLM plugin system](https://github.com/vllm-project/vllm/blob/v0.25.1/docs/design/plugin_system.md)
- [vLLM encoder-decoder model state](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu/model_states/encoder_decoder.py)
- [vLLM paged cross-attention](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/layers/attention/cross_attention.py)
- [vLLM BART plugin reference](https://github.com/vllm-project/bart-plugin/tree/v0.6.1)
- [T5Gemma 2 vLLM plugin](https://github.com/d0rj/t5gemma2-vllm-plugin)
