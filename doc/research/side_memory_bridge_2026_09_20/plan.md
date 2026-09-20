# Side-memory bridge research plan

## Decision

Identify bridge-only designs that add a compact, global evidence path while preserving the original source-token key/value route used by `eviseq_new` and grounded copy. The failed `afmr_query_regions`, `afmr_query_qspace`, `afmr_adaptive_topdown`, and `afmr_evidence_slots` variants are treated as negative evidence because they residualized a learned summary into the token-key route and reduced all three ROUGE scores.

## Constraints

- Keep `H0` as the exact source-value/copy anchor.
- Keep grounded copy on the original token route.
- Use token-level autoregressive CE/NLL only; no candidates, self-improvement, contrastive loss, salience loss, or auxiliary reconstruction loss in the first experiment.
- Change the bridge/interface only. Do not change the encoder, decoder checkpoint, tokenizer, data, optimizer, or generation settings.
- Preserve a direct token-attention fallback and initialize the new side path to zero influence.

## Falsifiable hypotheses

1. A separate gated cross-attention side bank improves validation ROUGE because it supplies global context without perturbing token-level matching.
2. A normalized mixture of token-bank and side-bank contexts is safer than concatenating their keys, because the side bank cannot steal token attention mass.
3. A block-global router that retrieves exact original token K/V can improve long-document coverage while retaining entity and number fidelity.
4. A hierarchical recurrent/compressive side bank can add long-range evidence, but only if the original full token route remains available; otherwise compression loss will dominate.

## Stop criteria

- Reject a direction if validation R-1 or R-L drops by >=0.10 with matched seed/protocol.
- Reject a direction if a checkpoint intervention shows the added path has no effect on logits or if its gate remains dormant after training.
- Do not select on the test set; use validation for architecture choice and run test once for the chosen variant.

