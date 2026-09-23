# Decision: a complementary role for a bridge

Date: 2026-09-23. Scope: research only; no architecture code changes.

## Question

What useful operation can a bridge H_encoder -> source memory perform that is
not merely another source selector or a linear map already expressible in
decoder cross-attention? Recommend one bounded experiment, or report insufficient
evidence. Do not manufacture an architectural contribution for a desired ablation.

## User constraints

- Keep encoder, decoder cross-attention, grounded copy, and token CE unchanged.
- No graph, parser, auxiliary label, extra generation, or preprocessing pipeline.
- Reject revisiting region query, projected-query regions, top-down, latent slots,
  depth mixture, hyper-operator, side memory, source FiLM, evidence routing, and
  delivery ledger under new names.
- Preserve positions/masks and lexical alignment. No additional model downloads.
- Base code: eviseq_new; newest failed experiment: afmr_delivery_ledger.

## Hypotheses

H1. Pure linear space alignment is algebraically redundant with trainable cross
    K/V in the unconstrained case; it is not sufficient justification on its own.
H2. A source-conditioned distribution/geometry correction can improve conditioning
    under limited training, but current normalization may already remove it.
H3. Position-preserving nonlinear local composition before attention supplies a
    useful inductive bias beyond per-token K/V, despite contextual encoder states.
H4. The strongest control already provides everything a bridge can profitably add
    under this budget; new capacity may not improve summarization at all.

## Existing work and evidence strategy

Read existing research reports and actual memory/normalization/copy routes first.
Use parallel luna workers for interface evidence and independent adversarial
review while coordinator audits the exact algebra and evaluates primary papers.
Use web search/open (available), local source, academic papers and official
implementation documents. No API credentials are needed or displayed. Research
claims should be triangulated where possible; source type concentration and
absence of exact PPLX->Qwen experiments must be explicit, not hidden by citation
counts. Primary-only sources are required for technical assertions.

Search opposition: encoder contextualization already sufficient; linear mapping
absorbed into attention; normalization erases correction; random bridge hurts;
local smoothing loses names/numbers; learnable residual unused; zero-init gradient
delay; multimodal projector results do not imply summarization gains.

## Outputs and stop criteria

Per-source notes with short verified quotes, source quality scores and caveats;
atomic findings; a decision report with falsifiers; refresh targets. Stop when
one candidate has a precise bridge-only contract, clear distinction from failed
designs, an honest evidence limit and a cheap discriminating experiment. If not,
report that rather than filling the gap with promised ROUGE gains.

## Correction from the user

XOV has NOT been experimentally tried. Its presence in Git proves an
implementation exists, not that a run failed. The coordinator initially
conflated those facts; that exclusion is withdrawn. XOV is an untested
candidate and should be assessed before inventing a substitute for it.
