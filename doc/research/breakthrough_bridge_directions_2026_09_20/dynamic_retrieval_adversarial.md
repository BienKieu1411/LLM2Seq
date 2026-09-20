# Adversarial pass: dynamic exact-source bridge ideas

## What could make the proposed gains illusory?

1. A query-dependent branch can simply learn a second noisy attention path and
   dilute the already strong direct projection. This is why all proposals keep
   the original token path and require gate/mass diagnostics.
2. Sparsemax or hard top-k can omit the one token needed for a number or entity.
   Use soft fallback or differentiable interpolation; never rely on a gold
   target to choose candidates at training time.
3. Phrase/span pooling may improve ROUGE-2 by copying frequent n-grams while
   lowering semantic faithfulness. Evaluate BERTScore and AlignScore together
   with ROUGE.
4. Coverage may suppress legitimate repeated biomedical terms. The primary
   pointer-generator paper reports that coverage without its extra loss did not
   reduce repetition, so a CE-only result should be treated as uncertain.
5. Landmark and deformable attention are not summarization papers. Their
   transfer provides a mechanism precedent, not an empirical PubMed claim.
6. If the new branch is only added before the decoder and never sees decoder
   queries, calling it query-dependent would be incorrect. The implementation
   must expose the query-dependent read inside decoder cross-attention.

## Decision

Run the query-hierarchical exact read and the landmark-gated exact block read
first. Treat posterior span/edit and coverage as follow-up variants only if the
first two fail for identifiable reasons. Do not combine all four in one run;
the ablation would become uninterpretable.
