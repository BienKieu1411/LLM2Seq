# Bridge contribution investigation

Date: 2026-09-18. Genre: implementation decision and falsifiable research note.

## Decision

Choose one bridge change for `src/eviseq_new` that has a specific causal route to better PubMed summarization while keeping the direct-projection ablation, H0 value anchor, grounded copy, data, and evaluation protocol fair. Code correctness is necessary but cannot establish a ROUGE gain without retraining.

## Observations and hypotheses

- Observed PubMed ROUGE-1/2/L: previous full 49.686/22.153/45.939; direct projection 49.671/22.135/45.956; subsequent all-region contextual value 49.657/22.098/45.920. The differences are too small to identify a cause from aggregate scores alone. The new prediction file has 6,657 ID/prediction/reference rows, no source or paired control outputs.
- H1: a source-only value adjustment may be redundant with a strong pretrained encoder; a predicted source-evidence prior shared by query-conditioned semantic attention and grounded copy may make the bridge useful. The prior itself is source/prompt-conditioned, not decoder-query-conditioned. Falsify by masking the prior at eval and checking unchanged outputs/loss, or by retraining and observing no paired gain.
- H2: the current zero-initialized, bounded paths train correctly but are too weak or too diffuse to affect content selection. Falsify with checkpoint activation/gradient measurements showing substantial, token-discriminating signal despite flat ROUGE.
- H3: supervision derived from gold source-summary alignment at training time could improve evidence selection without generating candidates. Falsify by controlled full-vs-no-auxiliary training and validation ROUGE/faithfulness.
- H4: reported sub-0.05 differences reflect run variance or evaluation formatting, rather than a bridge mechanism. Falsify with same-seed paired outputs and multi-seed confidence intervals.

## Search and implementation scope

Use existing AFMR project reports first; search primary summarization/encoder-decoder papers and official model reports for support and counterexamples. Prefer sources with direct ablations. Avoid other `src` architectures, model downloads, training-time generation, contrastive/R-Drop/NEFTune additions, and any weakening of w/o bridge. Implement only after the mechanism and measurement route are explicit.

## Risk register

- A stronger bridge can merely perturb fluent text, lowering all ROUGE metrics.
- Auxiliary labels can leak gold summaries at inference if wired incorrectly.
- Length/formatting and truncation can mask architecture changes.
- Another zero-initialized branch can be trainable yet dormant for the short training budget.
- An ablation gap from mismatched hyperparameters, preprocessing, or random seeds is not a contribution.

## Stop criteria

Stop literature expansion when the selected design has an explicit gradient route, at least three independent relevant sources or an insufficient-evidence label, a serious counterargument, and a controlled evaluation plan. Do not claim full > w/o until a trained checkpoint yields a reproducible paired gain on held-out data.

## Decision log

- 2026-09-18: Disabled contextual-value branch by default after the measured version lost on all three ROUGE scores. The optional local variant remains available but is not itself measured.
- 2026-09-18: Selected train-only silver phrase labels for the already-predicted `source_bias`, which is shared by semantic cross-attention and grounded copy. Preserved `H0` values and did not introduce hard source pruning or generated candidates.
- 2026-09-18: Reduced tentative auxiliary coefficient from 0.2 to 0.002 after a tiny-model gradient check found 0.2 made the focus gradient roughly three orders of magnitude larger than CE at initialization. This calibration is not a full-scale optimum.
- 2026-09-18: A third-pass ablation audit found that the PubMed full default had the auxiliary objective enabled while the direct-projection control forcibly disabled it. Set both PubMed recipe and queue defaults to CE-only. Keep evidence supervision as an explicit, separately named arm so architecture and training effects are not confounded.
