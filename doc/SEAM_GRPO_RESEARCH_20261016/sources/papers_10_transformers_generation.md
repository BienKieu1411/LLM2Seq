# Source note: Hugging Face generation sampling transformations

- **URL:** https://huggingface.co/docs/transformers/v5.5.3/en/main_classes/text_generation
- **Publisher / version:** Hugging Face Transformers official documentation, version 5.5.3 (the page notes a newer version is available).
- **Credibility:** High for this library's generation parameter semantics.
- **Recency:** Medium-high; versioned documentation was current in the recent public docs when accessed 2026-10-17.
- **Bias:** Low. API documentation; behavior is library-version-specific.

## Evidence used

The documentation says temperature changes the next-token probabilities and top-p keeps a probability-mass-dependent subset of tokens. It also notes that some logits processors break normalization and recommends renormalizing when using search algorithms that assume normalized scores. Therefore, for a mathematically on-policy rollout, store log-probabilities after the same sampling transforms used to draw tokens and recompute current probabilities with the same transform. This consistency rule is an inference from PPO's behavior-policy ratio plus the documented transformations. Starting with temperature 1 and top-p 1 avoids truncation-support mismatch; if top-p or other filters are introduced, test the transformed distribution and its normalization explicitly.

> “temperature ... modulate the next token probabilities”

## Limitations for this decision

This is an API reference, not an RL algorithm proof. Generation and training-forward code can use distinct code paths; the documentation alone cannot guarantee their log-probabilities match.
