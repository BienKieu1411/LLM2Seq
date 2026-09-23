# XOV bridge audit and bounded repair

Task: improve source-to-summary training through a correct XOV bridge; performance: matched validation CE and ROUGE-1/2/L; experience: supervised source-summary pairs with the existing single likelihood objective.

## Scope and identity

At initial inspection HEAD is fa27888 and src/xov_bridge is absent. The existing research identifies efce6f8:src/xov_bridge as historical XOV. A clarification is pending before restoring or replacing production files. Existing research and user files remain intact.

Clarification received: the user means the XOV design in research/debate. Deliver a revised research design and isolated prototype; do not restore or edit src. HEAD later became 45a20ba through an external change; the coordinator has not committed anything.

## Falsifiable hypotheses

1. Linear local composition followed by mean alignment loses some internal token-order distinctions. Test the existing many-to-one counterexample and a corrected activation placement.
2. Filtering decoder tokens can create false convolution neighbors. Test with a deliberately omitted internal token and preserve original token adjacency if confirmed.
3. Corrected values can reach the decoder while keys and copy context remain anchored, with finite CE gradients and exact initialization parity. Verify actual model wiring and checkpoint graph identity.
4. These repairs improve held-out summary quality: untested; tensor tests cannot establish this hypothesis.

## Method and sources

Reuse bridge_role_2026_09_23 findings first. Parallel luna_worker slices: historical code audit; verification of primary ConvS2S/Primer papers and opposing interpretations. Coordinator owns implementation and regression checks after identifying the intended source tree. Use the Deep Learning book chapters 8/11 for optimization/debugging principles, not claims about post-2016 Transformer designs. Web tools provide public sources; credentials are unnecessary and will not be read.

## Risks and stop criteria

- Do not conflate historical implementation with an experiment or claim a ROUGE gain.
- Distinguish aggregate pooling ambiguity from one verified collision.
- Preserve CE-only learning, visible-source boundary, base copy memory and parameter cap.
- Same-shaped checkpoints may encode different computation; changed operator needs explicit graph metadata.
- Synthetic CPU models only; no pretrained model downloads or remote training.
- Stop after actionable defects are corrected and bounded gradient/routing/checkpoint tests pass. Leave dataset-level efficacy explicitly unresolved.

## Adversarial review

Ask whether lexical input is redundant with H0/copy; whether normalization removes residual effects; whether changed masking/pooling merely trades one bias for another; whether all claimed improvements are measured at the level claimed. Literature motivation with fewer than three independent source types is insufficient evidence of task effectiveness.
