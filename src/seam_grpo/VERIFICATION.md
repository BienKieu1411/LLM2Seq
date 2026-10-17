# Verification: 2026-10-17

## Follow-up audit: initializer, paths and engine recreation

The same full command below now reports **247 passed, 3 skipped, 7 warnings
in 59.32 seconds**. New tests execute the actual initializer with only the CUDA
engine constructor stubbed, for one rank and both ranks of a two-rank setup.
They check per-rank environment, common rendezvous, decoder context budget,
processed log-probs and one-pass source normalization. CUDA execution is still
not part of these tests.

Fixed: relative backbone resolution against the SFT config, ambiguous legacy
CWD paths, stale version after engine recreation, and missing-CUDA failure only
after backbone loading/baseline evaluation. Extra guards reject a missing shared
multi-rank port and mixed-device/offloaded runtime weights before copying. A
same-step close/recreate regression confirms full synchronization happens again.

All earlier Gloo/DDP, real TRL, mixture-gradient and inference regression checks
remain green. Remaining CUDA/NCCL/VRAM/throughput/quality limitations are unchanged.

## Latest: colocated full-actor vLLM integration

```bash
PYTHONPATH=src/eviseq_new:src/seam_grpo:src/seam_grpo/checks \
python -m pytest -q src/seam_grpo/checks src/eviseq_new/tests/test_seam_vllm_*.py
```

Result: **238 passed, 3 skipped, 7 warnings in 58.72 seconds** on macOS CPU.
The skip conditions belong to optional inference checks. vLLM/CUDA was not
installed or executed on this Mac, and no pretrained model was downloaded.
After the final logits-update guard and absolute local-path normalization,
focused integration/mapping/inference checks also passed: **47 passed,
2 skipped in 13.42 seconds**. Ruff/format and whitespace checks were repeated.

New coverage:

- Production full-actor mapping on real torch parameter trees with vLLM's packed
  QKV/gate-up layout; copy on/off, tied/untied embeddings and padded vocabulary.
- Two committed weight versions; complete parameter/slice coverage; stale,
  incomplete, duplicate, wrong-shape and nonfinite updates are rejected.
- RPC transaction clears completed request copy/encoder state; no disk snapshot
  or CPU weight staging is used by the colocated path.
- Actual TRL training through the vLLM callback with an explicitly marked CPU
  transport double; two successive actor versions, EOS/processed-log-probability
  conversion and frozen reference invariance. Injected log-probability mismatch
  stops training before the first optimizer update.
- Actual two-process Gloo with native and vLLM-callback transport-double paths,
  CE 0/0.05 and accumulation. Gradients and weights match across ranks around
  each optimizer step. This validates distributed training/control wiring, not
  vLLM's CUDA executor or NCCL kernels.
- Regression coverage for Python/HTTP request paths, tokenization, state,
  copy mixing, encoder microbatching, serving and evaluation helpers.

Ruff, shell syntax and whitespace checks passed. The remaining server checks
are real vLLM 0.25.1 initialization on 1/2 GPUs, CUDA copy/scoring parity, memory,
throughput and final Perl ROUGE. Backend parity thresholds are runtime guards,
not evidence of identical numerics or a quality improvement.

## Earlier: native-only TRL integration

Environment: `/Users/kieugiangbien/bienkieu_env`, macOS CPU. Network model
loading disabled; only randomly initialized tiny backbones were used.

```bash
python -m pytest -q src/seam_grpo/checks
```

Result: **21 passed**, 5 warnings, 28.78 seconds. The warnings were two SWIG
dependency deprecations and three notices that TRL's rollout hook is experimental.
Installed TRL: **1.16.0**, Transformers 5.14.1, Accelerate 1.13.0, Torch 2.12.0.
The active training loop is the real `trl.GRPOTrainer` through a pinned subclass.

Covered:

- Chunked full-mixture action log-probabilities versus the original model's
  dense logits, at temperatures 1 and 0.7, with copy enabled and disabled.
- Every named parameter gradient compared against the dense reference.
- Explicit nonzero finite gradients for bridge memory/value memory/source bias
  and all grounded-copy parameters; encoder and decoder routes also checked.
- Cached rollout versus teacher-forced log-probability parity and an RL-only
  encoder weight update.
- EOS is scored and later padding is ignored, including pad-token equals EOS.
- Positive/negative advantage clipping, detached behavior probabilities,
  zero-variance groups, padding NaNs, optional reference-KL gradients, and fixed
  horizon gradient accumulation equivalence.
- Unicode Vietnamese reward behavior, weighted metrics, invalid reward configs.
- SFT weights loaded with reset step count; GRPO with and without frozen SFT KL;
  saved checkpoints loaded through the standard AFMR checkpoint loader.
- Decoder context-bound rejection and cache cleanup after encoder failure.
- Two real CPU/Gloo processes with gradient accumulation, complete local groups,
  uneven dataset length, CE weight 0 and 0.05. Gradients and all weights were
  compared across ranks immediately around each optimizer step.
- Cosine schedule including zero warmup.
- A spy on upstream `GRPOTrainer._compute_loss` confirms the library objective
  runs during training. Two optimizer updates consume rollout versions 0 and 1;
  version assertions reject stale samples, and buffers are cleared after update.
- HF actor wrapper: action likelihoods and gold CE share one DDP actor forward;
  their combined gradients match direct scoring. All frozen reference tensors
  remain bitwise unchanged after training.

The first two-process launch using `--standalone` timed out before workers
started because this Mac's hostname did not resolve. Static loopback rendezvous
resolved it without changing the host network configuration. Accelerate's CPU
barrier with CUDA-style device IDs also selected unsupported MPS synchronization
on this Mac; the lifecycle callback uses a plain Gloo barrier on CPU. Both actual
TRL distributed tests then passed. The launcher uses loopback for its single-node
default. This callback workaround does not alter CUDA synchronization.

Also passed: Ruff check/format, shell syntax checks, launcher `--help`, and
`git diff --check`. No commits or pretrained model downloads occurred. The small
TRL 1.16.0 package was installed with `--no-deps`; no dependency upgrades or heavy
libraries were installed.

Unverified: CUDA/NCCL execution, full-model memory/time, BF16 GPU likelihood
differences, and actual validation/final Perl ROUGE gains. CPU parity does not
establish these. The training reward is explicitly a proxy; use the unchanged
ROUGE-1.5.5 evaluator for the final SFT-versus-GRPO comparison.
