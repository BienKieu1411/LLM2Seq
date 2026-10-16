# Calling a local SEAM checkpoint through vLLM

The repository now includes an in-process Python facade over vLLM. The intended
call is close to using a regular model library:

```python
from seam_vllm import SEAMLLM

model = SEAMLLM(
    config_path="/path/to/resolved_config.yaml",
    checkpoint_path="/path/to/last.pt",
    max_num_seqs=4,
    gpu_memory_utilization=0.90,
)
summary = model.summarize("First source document ...")
summaries = model(
    ["First source document ...", "Second source document ..."],
    use_tqdm=True,
)
```

The run config supplies token limits, temperature, top-p, and top-k defaults.
AFMR-only decoding constraints are not inherited: no-repeat-ngram is omitted,
and repetition penalty defaults to the neutral value `1.0`. No custom sampler
is added to emulate these constraints. Training configs and the normal AFMR
evaluator remain unchanged.

```python
summary = model(
    "Source document ...",
    max_new_tokens=512,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
)
```

Native vLLM `repetition_penalty` can still be passed explicitly; it includes
prompt tokens, unlike AFMR's chat-prompt penalty. These defaults change decoding
relative to the original AFMR recipes. Use matching constraints in both runtimes
before comparing generation parity.

`generate()` also accepts the normal vLLM `SamplingParams` as its second
argument, or a sequence with one entry per source:

```python
from vllm import SamplingParams

params = SamplingParams(max_tokens=256, temperature=0.7, top_p=0.9, top_k=40, seed=42)
outputs = model.generate(["Source A ...", "Source B ..."], params)
print(outputs[0].outputs[0].text)
```

Explicit `SamplingParams` replaces the run's sampler defaults. Keyword calls
accept both `max_tokens`/`min_tokens` and AFMR's
`max_new_tokens`/`min_new_tokens`, as well as native options such as `seed`,
`stop`, `stop_token_ids`, and output `logprobs`. Pass either a `SamplingParams`
object or keyword options, rather than combining them. An explicit positive
`temperature` enables sampling unless `do_sample=False` is also specified.
`temperature=0` is greedy. Explicit beam search or nonzero
`no_repeat_ngram_size` requests are rejected because this adapter does not
provide them.

Each request's finite `max_tokens` is also sent to the bridge as its requested
output budget. It participates in vLLM's multimodal input hash, so the same
source with different budgets cannot share a mismatched encoder result. Use
`max_input_tokens=N` to reduce the source token cap for a call; `N` cannot exceed
the source limit used when loading the engine. Increasing that limit requires
loading an engine with the corresponding config. `generate()` supports native
multiple completions; `summarize()` requires `n=1` and `detokenize=True` to avoid
silently dropping outputs. Prompt log-probabilities remain unsupported.

`generate()` returns vLLM `RequestOutput` objects when token IDs, timings, or
log-probabilities are needed. `summarize()` and calling `model(...)` directly
return a string for one source and a list of strings for a batch. The wrapper
reads the encoder, decoder, prompt, precision, and generation settings from the resolved
config, and loads only local checkpoint paths. It enables offline HF loading
and does not download backbone weights.

## Install in the existing vLLM environment

On the server, activate the environment that already contains vLLM 0.25.1 and
run:

```bash
bash src/eviseq_new/scripts/install_seam_vllm_plugin.sh
```

The install uses `pip install --no-deps --no-build-isolation -e ...`; it does
not replace vLLM, PyTorch, Transformers, CUDA, or download model weights. Set
`SEAM_VLLM_PYTHON` if the active environment's interpreter is not `python3`.

The facade requires version `0.25.1` and selects the V2 GPU model runner.
An explicit `VLLM_USE_V2_MODEL_RUNNER=0`, an incompatible runner fallback, or a
nonstandard loader such as `load_format="dummy"` is rejected.

## Initial adapter boundary

This first serving adapter supports **Qwen3 causal decoders only**, with
`tensor_parallel_size=1` and `pipeline_parallel_size=1`. It uses the configured
HF encoder and the shared AFMR bridge, and restores the trained Qwen3 decoder,
cross-attention, and grounded-copy weights from the AFMR `.pt` checkpoint. The
source-bias term is carried in the paged cross-attention key; grounded-copy
probabilities are mixed into logits before vLLM samples. Missing alignment or
misaligned request state raises an error instead of falling back to
decoder-only generation.

The adapter sets vLLM `dtype` from AFMR's CUDA evaluation precision
(`model.compute_dtype`, falling back to `model.dtype`) and rejects a mismatch.
The bridge and grounded-copy modules stay FP32, matching AFMR evaluation.
It requires `enforce_eager=True` because vLLM 0.25.1's
request-side copy state is Python-managed. It disables prefix caching because
the adapter has not verified restoring that state on cache hits. The adapter
also rejects speculative decoding, quantized model weights, and non-default
KV-cache dtypes. The source-bias signal is encoded in cross-attention keys, so
changing the cache dtype would change that signal. The facade and direct model
configuration must use the same local decoder for both model and tokenizer,
`tokenizer_mode="auto"`, matching revisions, and initialized tokenization.
Grounded copy also requires the fast decoder tokenizer for offset alignment.
The bridge is conditioned on the exact decoder prompt supplied for each request.
The processor checks that the decoder IDs equal the IDs sent to the bridge,
including request-level prompt overrides. Without overrides it uses the trained
config prompt. The
checkpoint loader checks for all expected Qwen3 tensors and verifies that vLLM
reports every base decoder parameter as loaded. Empty processor-cache batches
remain empty instead of being treated as an empty source. vLLM 0.25.1 normally
disables the multimodal processor cache for encoder-decoder models; this guard
also protects configurations that explicitly turn it on.
Scheduled source inputs represented only by a vLLM cache reference are rejected
because this adapter does not yet restore encoder outputs and copy alignment as
one atomic request state. Paged attention and continuous batching remain
available; CUDA graph capture is disabled.
Other decoder families (Llama, Gemma, Nemotron, or diffusion models), tensor
parallelism and speculative decoding have not been implemented by this adapter.
OpenAI-compatible HTTP completions and single-source chat requests are available
through the SEAM middleware described in [SEAM_HTTP.md](SEAM_HTTP.md). Use the same `SEAMLLM` interface
only after an adapter for the chosen family has been added and parity-tested.

## Encoder and decode execution

The configured HF encoder produces token states and the selected depth taps;
it is not replaced by a pooled sentence embedding. Encoder, bridge, and copy
preparation run once per scheduled source. Decoder steps reuse self-attention
and cross-attention K/V through vLLM's encoder-decoder runner.

While the ordered active requests remain identical, the adapter reuses their
packed immutable copy state. It also caches unique source-token destinations
and aggregates copy probability only over those destinations, avoiding dense
vocabulary-sized copy-mass and log-source buffers. The generator still computes
full-vocabulary logits and their normalizer. Request removal clears both caches;
reordering, replacement, and vocabulary changes invalidate the relevant entries.

Without a distinct value anchor, encoder payloads store one memory tensor and
the bias (`H+1` columns); anchored payloads retain separate key/value memories
(`2H+1`). The shared-memory case also reuses its normalization result.

The source prior uses two compensated padded key coordinates to reduce BF16
encoding error. For a 128-wide head, the padded width remains 136. CPU tests
cover the bias representation and additive-attention approximation; they do
not establish CUDA kernel parity. Copy-logit tests match the original mixture
exactly on CPU, including repeated source IDs and fully masked source rows.
CUDA duplicate-mass accumulation and end-to-end generated-token parity still
need validation on the serving GPU. These changes are not a measured throughput
claim. Research, precedents, and the server validation plan are in
[the optimization report](research/seam_vllm_inference_optimization/2026-10-16_validation.md).

## HTTP serving

See [SEAM_HTTP.md](SEAM_HTTP.md) for the native `vllm serve` launcher,
OpenAI SDK/curl examples, streaming, and request-level encoder/decoder prompts
and budgets. The HTTP path uses AsyncLLM rather than a synchronous Python
inference queue.

## Bare vLLM API

The facade calls the ordinary vLLM `LLM.generate()` API internally. For
integration code that already owns an `LLM` instance, install the plugin and
construct it with the same model override fields:

```python
from vllm import LLM, SamplingParams

engine = LLM(
    model="/local/path/to/Qwen3-decoder",
    dtype="bfloat16",  # use model.compute_dtype from resolved_config.yaml
    hf_overrides={
        "architectures": ["SEAMForQwen3"],
        "is_encoder_decoder": True,
        "seam_config_path": "/local/path/to/resolved_config.yaml",
        "seam_checkpoint_path": "/local/path/to/last.pt",
        "max_source_positions": 8192,  # match data.max_source_length
    },
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    enforce_eager=True,
    enable_prefix_caching=False,
    generation_config="vllm",
    kv_cache_dtype="auto",
)
outputs = engine.generate(
    [{
        "prompt_token_ids": decoder_prompt_ids,
        "multi_modal_data": {"text": source},
        "mm_processor_kwargs": {"seam_output_budget": 512},
    }],
    SamplingParams(max_tokens=512, temperature=0.0),
)
```

For this direct vLLM path, `prompt_token_ids` defaults to the configured decoder
prompt. For a custom decoder prompt, pass those same IDs as
`mm_processor_kwargs["seam_decoder_prompt_ids"]` so SEAM conditions the bridge
on the exact decoder input. `SEAMLLM` and the HTTP adapter do this automatically.
For direct-engine calls, set `VLLM_USE_V2_MODEL_RUNNER=1` before creating the
engine and keep `seam_output_budget` equal to the request's `max_tokens`.

T5Gemma 2 remains a separate encoder-decoder model and should use its own
vLLM plugin path. It is not routed through `SEAMForQwen3`.

## Validation status

The 2026-10-16 audit repaired per-call parameter forwarding, request-specific
bridge budgets, decoder-start-token insertion, profiling prompts, source
detokenization, full parameter-load reporting, and runner/loader guards. It
also removed a duplicate source-tokenization/copy-alignment pass, uses vLLM
RMSNorm kernels for the cross route, and avoids repacking single-request copy
states. Checkpoint loading memory-maps the project's standard torch zip
checkpoint so optimizer tensor storage need not be eagerly loaded into RAM.

These are implementation improvements, not measured throughput gains. Base
decoder weights are still loaded before the trained checkpoint is restored;
multi-request copy-state packing still runs on each decoder step. CUDA graphs,
prefix caching, and TP/PP remain disabled or unsupported as described above.

The adapter is statically checked against vLLM 0.25.1 and AFMR's local model
code. This Mac checkout has no vLLM runtime or GPU, so it has not passed a
server-side model load, kernel-level logit-parity test, greedy-decode test, or
throughput test. Run those checks before using it in a product; the first target
is one local Qwen3 AFMR checkpoint with grounded copy enabled. Audit sources and
findings are recorded under
`src/eviseq_new/research/seam_vllm_0_25_1_audit/`.
