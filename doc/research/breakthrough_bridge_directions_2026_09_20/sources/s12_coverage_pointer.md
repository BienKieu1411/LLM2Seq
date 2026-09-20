# See, Liu & Manning 2017 — Pointer-generator coverage

Primary record: https://www-nlp.stanford.edu/pubs/see2017get.pdf  
Venue: ACL 2017, pp. 1073–1083.

## Verbatim evidence

- Method, p. 1076: the coverage vector is “the sum of attention distributions over all previous decoder timesteps.”
- Method, p. 1076: the current attention decision is informed by a reminder of previous decisions.
- Training analysis, p. 1080: training coverage “without the loss function” was ineffective for reducing repetition.
- Training analysis, p. 1080: applying the coverage objective from the first iteration interfered with the main objective.

## Mechanism relevant to EviSeq

Maintain a differentiable cumulative cross-attention state over exact source tokens and feed a bounded novelty bias into the next cross-attention read. Restrict it to upper decoder layers, initialize the bias at zero, and keep a full-attention fallback. This is a CE-only architectural adaptation because the coverage state changes attention logits and gradients arrive through the usual token likelihood.

## Training status and limitation

The original gains use an explicit coverage loss; the paper’s own CE-only ablation is negative. This direction should therefore be ranked below exact hierarchical/landmark retrieval and evaluated only as a small gated branch. Biomedical summaries may legitimately revisit a term, so a hard repetition penalty is unsafe.
