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
The bridge is conditioned on the configured decoder prompt, so the request's
prompt token IDs must match it exactly; the processor rejects a mismatch. The
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
parallelism, speculative decoding, and OpenAI-compatible HTTP requests have
not been implemented by this first adapter. Use the same `SEAMLLM` interface
only after an adapter for the chosen family has been added and parity-tested.

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
    },
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    enforce_eager=True,
    enable_prefix_caching=False,
    generation_config="vllm",
    kv_cache_dtype="auto",
)
outputs = engine.generate(
    [{"prompt_token_ids": decoder_prompt_ids, "multi_modal_data": {"text": source}}],
    SamplingParams(max_tokens=512, temperature=0.0),
)
```

For this direct vLLM path, `prompt_token_ids` must exactly match the decoder
prompt in `resolved_config.yaml`; SEAM uses those same IDs to condition the bridge.

T5Gemma 2 remains a separate encoder-decoder model and should use its own
vLLM plugin path. It is not routed through `SEAMForQwen3`.

## Validation status

The adapter is statically checked against vLLM 0.25.1 and AFMR's local model
code. This Mac checkout has no vLLM runtime or GPU, so it has not passed a
server-side model load, kernel-level logit-parity test, greedy-decode test, or
throughput test. Run those checks before using it in a product; the first target
is one local Qwen3 AFMR checkpoint with grounded copy enabled. Audit sources and
findings are recorded under
`src/eviseq_new/research/seam_vllm_0_25_1_audit/`.
