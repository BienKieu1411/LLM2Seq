# F04 — Fair three-way ablation and falsification protocol

Use one data/preprocessing/decoder protocol, one model initialization per paired seed, and fresh output/checkpoint directories for all three runs:

| Arm | Bridge graph | Evidence supervision | Purpose |
| --- | --- | --- | --- |
| A: `no_bridge` | `architecture.bridge_mode: direct_projection`; retain grounded copy | none | Isolate the no-bridge control already defined by the repository. |
| B: `afmr_ce` | current full AFMR bridge and `H0` value anchor | generation CE only | Measure the existing bridge without the proposed objective. |
| C: `afmr_silver_prior` | current full AFMR bridge, `H0` value anchor, predicted prior shared by semantic cross-attention and copy | train-only silver evidence loss plus CE | Test the incremental training method on the existing bridge. |

The primary system comparison is C versus A. B versus A identifies whether the existing bridge already moves the result; C versus B identifies the incremental effect of evidence supervision. C versus A alone does **not** isolate an architectural effect, because both graph and objective differ. Keep all three runs identical in prompt, truncation, tokenizer alignment, warm-up/full-training schedule, optimizer, decoding, and checkpoint selection. Gold-derived labels may be used only to calculate train/validation auxiliary loss and coverage; never feed them into generation or test evaluation.

Report paired ROUGE-1/2/L with bootstrap or multi-seed intervals, plus evidence-relevant diagnostics: proportion of rows with valid silver labels, prior separation on held-out labeled data, copy mass on positive versus competing occurrences, semantic attention mass, copy/LM gate, and bridge gradient norm. The contextual-value branch is off in all three arms, so a value-residual norm is irrelevant here. Extend to other project datasets only after the PubMed result is stable. A single test score or a difference below the run variance is not evidence of a contribution.

Falsify the hypothesis if any of the following holds: the shared prior has no target-dependent gradient; masking it leaves logits and source allocation unchanged; C does not exceed A on held-out paired evaluation; C improves ROUGE only through a metric-specific artifact while faithfulness or source precision worsens; or the result appears only for one seed and disappears under the pre-registered multi-seed comparison.
