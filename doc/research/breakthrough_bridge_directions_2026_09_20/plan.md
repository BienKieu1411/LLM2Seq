# Four breakthrough bridge directions: research plan

Date: 2026-09-20

## Decision

Identify four bridge architectures worth testing after four source-key residual
families underperformed the matched `direct_projection + grounded_copy` control
on PubMed. The output must separate paper-backed mechanism precedents from local
design hypotheses; no source is allowed to imply a guaranteed ROUGE gain.

## Locked constraints

- Preserve the PPLX encoder, Qwen decoder, original token path, grounded copy,
  prompts, data, decoding, and CE-style token likelihood.
- Exclude mechanisms already tried or already rejected: source-key residuals,
  region-query injection, projected-Q region residuals, top-down regional
  write-back, evidence-slot write-back, contextual-value residuals, salience
  auxiliary loss, contrastive loss, R-Drop, NEFTune, and generated candidates.
- Prefer a candidate that can fall back to the direct-projection behavior
  without corrupting original token keys or values.
- Select and tune on validation. The repeatedly inspected PubMed test result is
  not a clean architecture-selection set.

## Falsifiable hypotheses

1. **Layer-specific depth access:** decoder layers benefit from separate,
   aligned encoder-depth K/V banks rather than one early mixture shared by all
   layers.
2. **Independent dual-path read:** a compact global path helps only when its
   attention normalization and output gate are separate from the untouched
   token path.
3. **Dynamic exact-span read:** decoder-step-dependent routing to exact source
   tokens or spans preserves lexical detail better than pooled region/slot
   write-back and may improve R-2/R-L.
4. **Conditioned read operator:** a document-conditioned low-rank modulation of
   each decoder layer's cross-attention can align PPLX and Qwen without
   rewriting or compressing the source-token memory.

Each hypothesis is rejected if a matched validation run fails to beat the
direct-projection control or if the new branch is functionally dormant.

## Sourcing strategy

- Primary NLP papers on transparent/dense encoder-depth attention.
- Primary papers on multi-source and gated cross-attention fusion.
- Primary papers on hypernetworks, layer-conditioned adapters and
  instance-dependent parameter generation.
- Primary long-document summarization papers on dynamic snippet routing and
  exact-source access.
- Opposition queries: information loss from extraction/compression, prompt
  depth limitations, multi-source fusion failures, and auxiliary-loss
  dependence.

## Risk register

- A mechanism successful in vision-language or translation may not transfer to
  PubMed summarization.
- Compact-memory methods can repeat the same information-loss failure as
  evidence slots unless the full token path remains available.
- Independent branches can still dilute or destabilize the decoder after
  fusion; zero-init gates alone do not guarantee a useful learned branch.
- Dynamic routing may require auxiliary supervision in prior work; a CE-only
  adaptation is a new hypothesis, not a reproduced result.
- Repeated test-guided iteration can inflate the apparent final result.

## Stop criteria

- At least three independent primary sources per proposed mechanism family, or
  explicitly mark the evidence as insufficient.
- One adversarial pass comparing the four ideas against the observed local
  failures.
- Rank all four by expected benefit, implementation risk, and diagnostic
  clarity; recommend no more than two first-run candidates.
