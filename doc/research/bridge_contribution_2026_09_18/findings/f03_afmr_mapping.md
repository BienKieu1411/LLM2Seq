# F03 — External evidence mapped to current AFMR code

The source of truth is code, not the historical README. The full AFMR branch builds `memory` for decoder cross-attention keys, `value_memory=H0` for value content, and `source_bias` from overlapping focus windows. The grounded-copy head prepares its keys from `H0` and pools the same `source_bias` into its attention logits. Direct projection removes controller/depth/feature/focus/value anchor but keeps the encoder, decoder and copy head. `contextual_value` remains implemented as an optional experimental branch and is disabled in the updated PubMed recipe because the measured version lowered all three ROUGE metrics.

The new train-only auxiliary supervises the *existing* `source_bias`, not a separate salience head. Inference and CE training both use the predicted prior. Gold-summary phrase matches are detached labels used only in the auxiliary objective. The source value content is not replaced, no source tokens are hard dropped, and no candidates are generated.

This establishes a plausible causal path but no gain: the full system must beat direct projection in matched retraining. Ablate CE-only full versus direct as well as auxiliary full versus CE-only full to separate architecture and training effects.
