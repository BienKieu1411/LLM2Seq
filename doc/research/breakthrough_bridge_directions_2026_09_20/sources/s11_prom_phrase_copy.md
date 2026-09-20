# Ma et al. 2024 — PROM phrase-level copying

Primary record: https://aclanthology.org/2024.lrec-main.1148.pdf  
Venue: LREC-COLING 2024, pp. 13103–13119.

## Verbatim evidence

- Abstract, p. 13103: PROM “enhances attention on n-grams.”
- Abstract, p. 13103: the copying indicator is trained with “an auxiliary loss for the copying prediction.”
- Method, p. 13107, Eq. 4: the copying indicator is explicitly supervised with cross-entropy against copying labels.

## Mechanism relevant to EviSeq

The useful architectural idea is a phrase-level source route: form short contiguous n-gram/span entries from exact H0 token states and expose them as additional K/V candidates. A decoder query can select a phrase entry for multi-token continuity while the original token bank remains available for exact copy and rare details.

## Training status and limitation

PROM itself is not a CE-only architecture because its copying indicator uses an auxiliary loss and its paper also studies pre-training. Therefore only the phrase-memory interface should be transferred into the current CE-only experiment; do not claim to reproduce PROM without that supervision.
