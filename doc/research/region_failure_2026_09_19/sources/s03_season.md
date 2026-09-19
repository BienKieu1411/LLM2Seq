# S3 — SEASON

Primary source: https://aclanthology.org/2022.emnlp-main.409.pdf, Figure 2 and Section 3.3.

SEASON injects estimated sentence salience into cross-attention *keys* while keeping original encoder values. This supports a separation between selecting source content and reading its details. Its training generation uses gold salience while test uses predicted salience (Section 3.3–3.4); copying that protocol would introduce train/inference mismatch here. The paper also uses an auxiliary salience objective. The candidate here is CE-only, so SEASON's gain is not a performance forecast.
