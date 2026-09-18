# F01 — What the supportive evidence actually licenses

The strongest external pattern is a **source-side, trainable evidence signal that reaches decoder selection**. TopDownFormer shows a coarse representation can be globally updated and sent back to token representations; its PubMed ablation separates cross-attentive top-down correction from no correction (S01). SEASON predicts a gold-derived salience allocation and adds it to cross-attention keys while preserving the original values; its predicted-SACA row beats its no-SACA/no-MTL row on CNN/DailyMail (S02). HierGNN shows a learned hierarchy plus graph-selection attention can change which source sentences the decoder uses, with module ablations on XSum (S03). Deutsch & Roth show why phrase/occurrence-level silver labels are useful: two occurrences of the same phrase can have different target relevance, and predicted spans must be used at inference to avoid an oracle-label gap (S04).

Together these papers support a design hypothesis for `eviseq_new`:

1. Derive a **training-only silver evidence target** from source/gold alignment at a target unit (content phrase or short target span), with multiple positives when the evidence is tied and no label when alignment is ambiguous.
2. Predict a source prior `p_i` from the source/bridge representation. At evaluation, use only `p_i`; never pass the reference, gold spans, or a gold-derived mask into the model.
3. Share `p_i` with both decoder semantic cross-attention and grounded-copy selection. Let it modify key/logit selection, while keeping the content values anchored to the unmodified `H0`. This makes the bridge's output observable by both reading paths without replacing the native source content.
4. Keep the auxiliary supervision train-only and small enough that the CE generation objective remains primary. Log evidence-label coverage, confidence, positive/negative counts, and train/eval prior statistics.

The recommendation is a **hypothesis**, not a result. S01–S03 use different hierarchy units and model families; S04 uses external QG/QA models and short news data. None of them tests AFMR's overlapping windows, bounded value residual, grounded copy, or Perl155/PubMed checkpoint. The external result that transfers is the causal route—source prior → decoder selection—not a ROUGE number.

