# Mao et al. 2022 — DYLE dynamic latent extraction

Primary record: https://aclanthology.org/2022.acl-long.118.pdf  
Venue: ACL 2022, pp. 1687–1698.

## Verbatim evidence

- Abstract, p. 1687: DYLE allows “dynamic snippet-level attention weights during decoding.”
- Method, p. 1689, Eq. 3: generation probability is a mixture over snippets weighted by the generator’s decoder-history-dependent distribution.
- Method, p. 1690: the generator loss is the NLL of the gold summary.
- Training objective, p. 1690: the full model adds oracle extraction and consistency losses; top-K extraction is non-differentiable.

## Mechanism relevant to EviSeq

Use the decoder query/history to weight exact source snippets, but keep all snippets in a differentiable soft mixture rather than hard top-K. A candidate can read each selected snippet’s original token K/V and marginalize the resulting token likelihood, while the copy route continues to use H0.

## Training status and limitation

DYLE is strong precedent for decoder-history-dependent evidence selection, but its extractor requires oracle and consistency supervision. A CE-only EviSeq version is a new ablation hypothesis, not a reproduction of DYLE. The paper’s long-document gains cannot be attributed to dynamic weights alone.
