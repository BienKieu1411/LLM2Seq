# XOV implementation ledger

Scope: new `src/xov_bridge`, standalone package `xov`; no edits to eviseq_new.
Template: historical XOV efce6f8, with current dataset/prompt semantics retained.
Contract: ../2026-09-23_design.md, overridden by user's explicit request for stronger,
nonzero initial lexical signal. Use orthogonal up-projection gain 1 and initial gate
0.10 (previously zero output / gate 0.05); retain cap 0.20. This biases access and
opens gradients, not an assurance of better predictions. No forced minimum gate.
No model download, commit, branch operation, or GPU training.

Acceptance checks:
- Separate base keys/copy memory and active values at initialization.
- Original decoder token ordinals, clipped prefix spans, no adjacency across gaps.
- SiLU before reverse pooling; visible source only; padding/empty rows safe.
- First CE backward and optimizer step affect norm/down/conv/up/gate.
- Copy enabled/disabled integration; cached generation and compaction.
- BF16 autocast finite, residual bounded, checkpoint operator rejection/roundtrip.
- Local model/tokenizer assets only; config/tokenizer fingerprints.
- Per-rank RNG capture and restoration for same-world-size DDP resume.
- CLI tiny train/eval, two-rank CPU check; no production metric claim.

In progress: verification and bounded independent reviewer.

## Verification evidence

- New package imports and CLI smoke: tiny Qwen encoder/decoder, forward/backward and
  cached generation passed without downloads.
- First production CE update: all lexical parameters have finite nonzero gradients
  and changed weights, with copy enabled and disabled (`test_active_bridge.py`).
- CLI: two tiny epochs (warmup then full), checkpoint save and test evaluation wrote
  two prediction rows. Python diagnostic ROUGE from random fixtures is not evidence
  of task quality.
- CPU Gloo two ranks: warmup/full training passed. Resume from epoch 1 produced
  exactly equal final model weights to the uninterrupted fixture (max delta 0).
  Per-rank RNG list length 2. See `runtime_result.json`.
- Empty-alignment rank: two DDP iterations with one empty lexical route passed;
  both ranks ended with equal, updated lexical weights.
- `initial_signal.json`: synthetic width1024/rank256 median relative RMS 0.0112452,
  max 0.0166686 at init. Not a measurement from real pretrained embeddings.
- Ruff, shell syntax, and diff whitespace checks: passing before final review.
- `self-review` skill is not installed; manual mechanical review and separate
  `luna_worker` reviewer used, as allowed by fallback discipline.

## Gap decisions

| Requirement | Status / evidence |
|---|---|
| Zero-up / exact initial identity | Overridden by user; nonzero orthogonal up and gate 0.10. Keys/copy still use X. |
| Adjacency and pre-pooling activation | Implemented in xov/modeling/xov.py; parity/gap/order tests. |
| Alignment visibility/boundaries | Token ordinals, clipped spans, excluded truncated final token; fixture tests. |
| Copy memory separate | Explicit BridgeState and filtered copy kwargs, both copy modes tested. |
| Checkpoint operator contract | Explicit contract plus asset fingerprints; old contract rejected by test. |
| RNG and map_location | Rank-local CUDA RNG and all-rank collected states; resume fixture passes on CPU. |
| Production GPU/BF16/NCCL | Not executed: local Mac lacks CUDA. CPU BF16 component tests passed; no claim of GPU verification. |
| Task-level quality | Not tested; requires matched training and validation ROUGE-1.5.5. |
| No auxiliary losses / generated training candidates | Preserved; token-level CE only. |
| Baseline preservation | No source edits outside src/xov_bridge; no Git commits or downloads. |

## Final review and status

Independent luna_worker review found omitted input/execution settings in checkpoint
compatibility. Fixed by recording source prefix/limits, target limit, decoder prompts,
field/detokenization policy, tokenizer mode, attention backend and dropout. Seven
negative checkpoint tests confirm rejection. No other bridge/gradient/cache blocker
was reported. Final suite: **30 passed**, Ruff check/format passed, shell syntax passed.
CLI train/eval repeated successfully after the checkpoint-policy fix.

Status: local implementation complete. Actual pretrained GPU training and task-score
validation remain experimental work, not a verified gain. Temporary tiny checkpoints,
logs and implementation helper scripts were removed after recording results.
