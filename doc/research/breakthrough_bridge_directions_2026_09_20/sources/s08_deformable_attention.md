# Zhu et al. 2021 — Deformable DETR

Primary record: https://arxiv.org/pdf/2010.04159  
Venue: ICLR 2021.

## Verbatim evidence

- Abstract, p. 1: deformable attention “only attend[s] to a small set of key sampling points around a reference.”
- Method, p. 5, Eq. 2: sampling offsets and attention weights are predicted from the query feature.
- Method, pp. 5–6: “the 2-d normalized coordinate of the reference point ... is predicted from its object query embedding.”

## Mechanism relevant to EviSeq

Adapt the one-dimensional source sequence as follows. A decoder query predicts `K` normalized source coordinates, local offsets, and weights. Read exact source K/V at those coordinates using differentiable linear interpolation between neighboring token positions. Keep a small dense cross-attention fallback and mix it with the deformable read using a bounded gate.

This is a query-conditioned read of exact source features. It does not create a global residual and does not require changing grounded-copy values.

## Training status and limitation

The original paper uses object-detection losses, so it is not evidence for summarization or CE-only training. The transferable operation is the query-to-offset/read rule; the proposed summarization adaptation would train only with token CE. Discrete top-k indexing must be avoided because it blocks useful gradients.
