# Source note: TRL v1.16.0 fused causal-LM scoring patch

- **Checked:** 2026-10-17.
- **Source type:** Official, tag-pinned Hugging Face TRL utility implementation.
- **URL:** [Tagged raw `utils.py`](https://raw.githubusercontent.com/huggingface/trl/v1.16.0/trl/trainer/utils.py); [GitHub source](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/utils.py).
- **Credibility:** High for what the helper patches and expects.
- **Recency:** Exact v1.16.0 release; later tags may change this path.
- **Bias:** Maintainer implementation; tests would still be needed for a custom adapter.

## Evidence used

`GRPOTrainer.__init__` calls `add_fused_lm_head` on the policy (and reference model, when present). That function wraps `model.forward`. With the default `fused_lm_head=False`, it forwards to the original `forward`; with `fused_lm_head=True`, it takes a causal-LM path that accesses the backbone and output embeddings, shifts labels, and calculates next-token log-probabilities. The base GRPO helper explicitly uses the fused path.

An architecture-specific scorer can call the wrapped model without the fused flag to reach the original forward, then compute selected decoder-token log-probabilities from the returned logits. This does not make the base helper compatible by itself: a custom helper must supply encoder inputs, decoder inputs, action alignment, and return entropy in the shapes expected by `_compute_loss`.

## Direct excerpt

> “Without it, the forward is the original one, so generation is unchanged.”
