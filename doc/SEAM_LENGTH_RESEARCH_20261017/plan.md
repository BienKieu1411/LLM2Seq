# Length-controlled SEAM: research contract

Date: 2026-10-17. Status: research and integration design, not an implemented training mode.

## Decision

Design a post-SFT GRPO experiment that receives a requested word count, improves length adherence, and preserves or improves ROUGE. Keep the existing `eviseq_new`, `seam_grpo`, and SEAM vLLM behavior until a separately tested integration exists.

The user confirmed that a desired word count is available at inference. Training must also expose that request to the model. A natural-language request, a numeric conditioning feature, and a hard decoder token cap are different interfaces.

## Falsifiable hypotheses

1. Matched length instructions plus content-and-length GRPO outperform unconditioned SFT on requested-length error without a validation ROUGE regression. Literature establishes feasibility elsewhere; SEAM remains untested.
2. Explicit numeric word conditioning in the bridge adds value beyond a decoder prompt under the same training, requests, and token caps. This is an architecture-specific hypothesis, not an established encoder-decoder advantage.
3. Length control can retain a useful signal even when content scores are low. Check the supplied reward branches numerically before reusing them.
4. A bounded training-length sampling correction improves underrepresented lengths without harming the seven-dataset result. Leave this disabled in the first pilot; balancing length bins is not equivalent to balancing datasets.

## Scope and sources

- Audit all six user-supplied Python files without executing their training entrypoints.
- Inspect the existing SEAM controller, collator, GRPO scorer/rollout, and vLLM request metadata.
- Use original papers, proceedings, journal text, and author repositories. Historical control mechanisms and recent RL length work answer different parts of the question.
- Verify bibliographic metadata and quoted passages by opening their primary pages. Save individual source notes and a CSV index.
- Adversarial searches: length-only reward harming content; additive cost compensation; multiplicative reward dead zones; inconsistent word units; test-length oracle and multi-reference counting.

## Limits and stop criteria

No model downloads, GPU training, server access, commits, or claims of measured SEAM gains. Stop after a reproducible script audit, a primary-source-backed design, explicit integration gaps, and a validation/acceptance plan. Independent agent audits were attempted but unavailable due to the session usage limit; findings here are the coordinator's audit, not multi-agent consensus.

Evidence diversity is limited to primary research, author code, and local implementation. There is no independent production evidence establishing that SEAM beats decoder-only length control. That stronger thesis remains insufficiently supported.

## Update: distinct encoder and decoder prompts

User steering after the initial report: exploit both prompts and the encoder-decoder separation, rather than treating length as a decoder-only instruction.

Research questions for this update:

1. Does length-aware source encoding improve content/length outcomes beyond decoder-only prompting under the same actual visible source?
2. Does the existing prompt-conditioned bridge suffice, or does it need an explicit numeric word input?
3. Can G candidates share the encoded source/bridge state at rollout while preserving full gradients during actor updates?
4. Does any benefit persist across causal and bidirectional encoder families without changing their attention masks?

Search guided summarization, source/target control, length-conditioned attention and planning. Keep source length, target request and output token caps controlled. Save this as a delta to the initial decoder-prompt-first recommendation; source-side conditioning is now an explicit primary candidate. No runtime code changes are requested by this research update.

Three `luna_worker` agents completed this update: primary-literature review, local conditioning/gradient/cache audit, and adversarial review. Their findings are consolidated in `reviews/dual_prompt_review.md`. They did not run model training or modify runtime code. This is distinct from the unavailable agent checks during the initial report.

## Update: architecture-aware length reward

User steering: exploit the encoder-decoder interface when customizing length reward, beyond supplying two prompts.

Hypotheses to check before selecting a recipe:

- A source-content reward conditioned on the requested budget can improve content at that length over a length-only penalty; full-source recall is not the right target for very short summaries.
- Stable encoder-derived evidence units can provide useful reward at acceptable cost; trainable attention, copy probability and encoder embeddings can instead create self-reward shortcuts.
- A frozen teacher's correct-vs-ablated-source likelihood can measure source dependence, but may favor copying or rare unsupported tokens and may be too expensive for the first pilot.

Keep these as hypotheses. Compare source/saliency reward papers, coverage rewards, attention interpretability counterevidence, and local scorer interfaces. Decide whether an architecture-specific reward is warranted or whether the architecture should primarily alter conditioning/gradient flow. No code or heavy model execution in this research update.

Outcome: see `2026-10-17_reward_update.md` and `reviews/reward_review.md`. Adopt MAE/B as a proposed baseline cost, retain the frozen-teacher source-swap margin behind an offline validation gate, and reject raw actor-attention reward. The source-score effectiveness hypothesis remains underdetermined; compute and teacher bias are material risks.
