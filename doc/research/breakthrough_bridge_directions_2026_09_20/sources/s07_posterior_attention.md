# Shankar & Sarawagi 2019 — Posterior Attention Models

Primary record: https://openreview.net/pdf?id=BkltNhC9FX  
Venue: ICLR 2019.

## Verbatim evidence

- Abstract, p. 1: “the attention propagated to the next decoding stage is a posterior attention distribution conditioned on the output.”
- Abstract, p. 1: the output probability is a “mixture of output probability for each attention.”
- Method, Sec. 2 and Fig. 1: the posterior attention is conditioned on the current output, rather than using an output-independent prior attention.

## Mechanism relevant to EviSeq

At target step `t`, maintain a distribution over exact source positions or short spans. Compute a source-conditioned token distribution for each candidate and marginalize:

`p(y_t) = sum_i p(a_t=i | y_<t) * p(y_t | a_t=i, y_<t)`.

The decoder history updates the next alignment distribution. This differs from a static bridge source bias: the evidence choice changes as the summary unfolds, which is directly relevant to ROUGE-2 and ROUGE-L continuity.

## Training status and limitation

The original model is trained by sequence likelihood through its latent marginalization, so a CE/NLL-only adaptation is plausible. It was evaluated on translation and morphological inflection, not summarization. Exact marginalization can be expensive; use a differentiable block-plus-token approximation rather than a hard top-k operator.
