# SEAM GRPO and vLLM: implemented synchronization and verification

Date: 2026-10-17. User authorized improving the custom vLLM engine for GRPO and
checking both single-GPU and two-GPU training. This report supersedes the earlier
native-only decision's online-weight-transfer limitation. No commit was made.

## Implemented boundary

TRL 1.16.0 retains the GRPO optimizer loop/objective. `rollout_backend: seam_vllm`
selects the custom full-architecture callback. The stock TRL causal-only vLLM
model constructor is deliberately not invoked. The implementation creates one
SEAM vLLM 0.25.1 TP=1/PP=1 engine per training rank on that rank's GPU, using
`external_launcher` and `VLLM_ENABLE_V1_MULTIPROCESSING=0`. This follows the
[tagged TRL colocate implementation](https://github.com/huggingface/trl/blob/v1.16.0/trl/generation/vllm_generation.py)
and [vLLM's in-process external-launcher executor](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/executor/uniproc_executor.py).
Native rollout remains selectable and supports the base architecture's decoder
families. The custom vLLM path is currently Qwen3-only, unquantized, with cross
attention in every layer. It is not a general Llama/Gemma vLLM adapter.

Each source's G rows are supplied by TRL's existing repeated sampler; the vLLM
callback returns one completion per row with n=1. Groups stay local to a rank.
The actor's parameters remain unchanged until all accumulation microbatches
finish. DDP synchronizes gradients over encoder, bridge, decoder/cross-attention
and copy before the optimizer step. Each rank then updates its inference copy
before the next version's rollout. No decoder-only transfer is used.

## Weight transaction

`SEAMLLM.sync_weights(native_model, policy_version=k)` serializes generation and
sync with a local lock and requires a drained engine. The transfer uses public
[`LLM.collective_rpc`](https://docs.vllm.ai/en/v0.25.1/api/vllm/entrypoints/llm/)
with an in-process executor, so CUDA tensor references are copied directly to
the local inference tensors. No checkpoint file or CPU weight snapshot is
created per update. This is colocated GPU copying, not NCCL or CUDA-IPC transport
to a separate HTTP server.

Before writing weights, `ActorWeightUpdate` validates the complete native
parameter manifest against runtime destinations. It maps encoder/bridge/copy
parameters directly, translates decoder names, packs native Q/K/V into vLLM's
QKV rows and gate/up into merged rows, checks every destination slice, preserves
tied aliases and handles padded vocabulary rows. Packing applies only to the
vLLM decoder, never to an embedding encoder's ordinary Q/K/V projections.

All source parameters are sent, including frozen ones, so the inference policy
is a complete snapshot. Config-derived nonpersistent RoPE buffers are not actor
parameters; encoder and decoder are the same configured architectures. Dropout
is disabled in the actor and inference copy. No separate reference weights are
sent: optional KL uses a frozen full SFT actor.

Version numbers must increase. Shapes, duplicates, omissions and nonfinite
destination values are checked. The version is committed only after all
parameters arrive and CUDA copies complete. Failure leaves the engine blocked;
it does not roll back a partially written GPU model. Rebuild on failure.
Ordinary `load_weights` restores the SFT checkpoint only during initialization
and rejects further bare HF reloads, preventing the former static-checkpoint
overwrite during online updates.

## State invalidation

Before transferring a new actor version, the engine invalidates prefix/KV
cache ownership and multimodal preprocessing state. The worker retires the last
completed request IDs, which V2 can retain until its next scheduler step, and
clears encoder outputs, completed/pending/request copy state and copy mixer/batch
caches. New requests rebuild all encoder/bridge/copy and self/cross-KV state.
Physical KV pages need not be zeroed: ownership is reset and new source/prefill
writes populate the pages before decoding. No in-flight request is continued
across a version change. The exact worker/cache layout was checked against
[V2 model runner](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu/model_runner.py),
[encoder-decoder state](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu/model_states/encoder_decoder.py),
and [encoder cache](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu/mm/encoder_cache.py).

## Likelihood and preprocessing

Candidates are sampled from the final generate/copy logits at the configured
temperature, with top-p=1, top-k disabled and neutral repetition/length filters.
The engine is configured with `logprobs_mode=processed_logprobs` and returns the
sampled token's probability after temperature. IDs, EOS and log-probability
lengths are checked; text is not decoded/re-tokenized before training. Current
likelihoods are recomputed through the full native graph with autograd.

Two corrections were made during the audit: sampling sources were being sent
through an extra detokenization in the serving processor, so rollout now uses a
separate config with detokenize disabled after the dataset's processing; AMP
selection now consistently follows compute_dtype with the model dtype fallback,
including explicit FP16 support, matching the inference configuration.

Every scored microbatch logs native-vs-sampling log-probability mean/max error.
The default guard rejects mean >0.05 or max >0.50 log units before backward.
Ranks reduce/gather these statistics before deciding to fail, so they agree on
a parity rejection. These thresholds detect substantive mismatches; they do not
prove bitwise GPU equivalence. Saved actual behavior probabilities remain the
denominator of the policy ratio. Unexplained failures should be diagnosed rather
than hidden by increasing tolerance.

## Evidence and limits

Final local verification: **238 passed, 3 skipped, 7 warnings in 58.72 seconds**.
Torch fixtures verify real parameter copies with packed decoder layout, tied and
untied weights, copy enabled/disabled and padded vocabulary. Actual TRL runs
verify two rollout versions and a mismatch rejection before optimizer update.
Actual two-process CPU/Gloo verifies gradient/weight equality, accumulation and
CE 0/0.05 for native and vLLM-callback paths. The vLLM callback uses an explicitly
marked CPU transport double in those integration tests. Request/HTTP/tokenizer/
copy/encoder batching regressions are also covered. Ruff, shell and whitespace
checks passed. No pretrained models or heavy libraries were installed/downloaded.

**Still unverified:** real vLLM/CUDA executor initialization, NCCL, CUDA kernel
likelihood differences, combined VRAM and speed on one/two GPUs, and ROUGE gains.
The Mac tests cannot establish these. Default engine memory utilization 0.20 is
a pilot per-GPU reservation, not a cap on total actor+optimizer+engine memory.
Engine sleep mode, remote online sync and non-Qwen3 vLLM decoders are outside
this change. The server should first run a short fresh-output-directory pilot
(max_steps=1, small validation subset, then max_steps=2 to exercise a second
sync) before a full run. Final quality comparison still uses Perl ROUGE-1.5.5.

## Follow-up audit requested by the user

Additional concrete defects fixed:

- A relative model directory present alongside the SFT config was incorrectly
  checked only against the caller's current directory. Resolution now uses the
  SFT config, retains an unambiguous legacy CWD path and rejects ambiguity.
- A closed/recreated rollout engine could retain the old wrapper's policy
  version and skip syncing its freshly loaded SFT weights at the same trainer
  step. Recreation now always resets the sync version; a regression test closes
  and recreates at step 7 and verifies both engines receive version 7.
- Missing CUDA previously failed only when the first rollout initialized, after
  full backbone loading and possible baseline evaluation. It now fails before
  either. CPU transport doubles explicitly bypass only this hardware precheck
  in their tests.

New guards reject a missing common rendezvous port on multiple ranks and a
mixed-device/offloaded runtime before any weight copy. Tests execute the actual
initializer, stubbing only GPU engine construction, for a single rank (existing
and auto-selected port) and both ranks of the two-process layout. They verify
rank/device environment, shared port, output budget, processed probabilities
and that only the rollout config disables repeated detokenization.

Result after all fixes: **247 passed, 3 skipped, 7 warnings in 59.32 seconds**.
The complete native/TRL/two-process Gloo suites also pass. This remains CPU,
source and control-path evidence; no real CUDA/vLLM/NCCL or score claim is added.
