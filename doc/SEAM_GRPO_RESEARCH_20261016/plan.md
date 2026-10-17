# Post-SFT GRPO for SEAM: research and implementation plan

Decision: add an isolated `src/seam_grpo` phase that loads an eviseq_new SFT checkpoint, optimizes ROUGE reward, and retains a small supervised gold-summary CE term. Existing SFT and serving files are outside the edit scope.

Hypotheses:
1. Policy action log-probabilities must include the full grounded-copy mixture; a decoder-only TRL wrapper would optimize a different distribution.
2. ROUGE-2-sensitive reward may improve overlap, but neither GRPO nor CE guarantees ROUGE gains, fluency, or factuality.
3. Length-independent normalization and optional reward-std scaling reduce specific GRPO biases; their suitability for Vietnamese summarization remains empirical.
4. Gold CE can anchor supervised behavior without a critic or external judge; its coefficient is an ablation, not an established universal constant.

Scope: audit the supplied pure_rouge.py; original GRPO and modern corrections; Vietnamese tokenization; exact trajectory/EOS masks; gradient flow; offline tiny-model tests; cosine schedule; 1-GPU and torchrun DDP. Native model rollout initially avoids stale vLLM weight synchronization.

Sources: official papers, author repositories, framework documentation. Save each source with URL, short verified quotations, credibility/recency/bias, and applicability. Mark unsupported transfer to this task as a hypothesis. Do not expose API keys during capability discovery; web access and local source inspection are sufficient.

Risks: reward hacking/length drift; group zero variance; invalid likelihood ratios with top-p truncation; dropout old/new mismatch; gradients through stale cache; EOS/prompt masking; DDP unequal batches; reference-model memory; accidental test-set reward tuning.

Stop criteria: a reusable research report, runnable separate package/config/scripts, meaningful offline tests for mixture gradient routes and the GRPO objective, and an adversarial review. GPU performance and quality claims remain pending server experiments. No models downloaded, heavy installs, or commits.

## 2026-10-17: user correction — use TRL and audit rollout weight lifecycle

The initial implementation uses a custom PyTorch loop. User now explicitly asks
for TRL, so the target is a real GRPOTrainer subclass with the library's training
loop/objective, plus architecture-specific rollout, likelihood, and gold-CE
adapters. Verify the tagged source before selecting extension hooks; importing
GRPOConfig alone does not satisfy this request.

Additional hypotheses:
- A custom rollout callback alone is insufficient unless per-token scoring also
  receives the separate source and recomputes the full bridge/copy graph.
- Native rollout sharing the trained actor requires no separate weight copy,
  but must never overlap an optimizer update and must rebuild stale caches.
- A separate inference engine must synchronize the complete effective policy,
  including encoder/bridge/cross/copy, before starting the next rollout version.
- Updating encoder/bridge changes conditional representations but does not
  invalidate importance ratios if old probabilities were saved and current
  probabilities are recomputed under the full current policy.

Capability check: TRL absent locally; base dependencies already installed.
PyPI reports TRL 1.16.0 current, Python >=3.11. Install only the small TRL package
without dependency upgrades for local verification; no torch/model downloads.

Completed: actual TRL 1.16.0 subclass and HF full-actor wrapper, source-aware
native rollout/likelihood hooks, one-forward DDP gold CE, frozen full SFT
reference, version guards and cache/buffer lifecycle. All 21 offline checks
passed, including two-process Gloo. See the current
[TRL decision](2026-10-17_trl_rollout_decision.md). GPU performance/quality
remain server experiments. No commit or pretrained-model download was made.
