# Source note: official TRL GRPO documentation

- **URL:** https://huggingface.co/docs/trl/grpo_trainer
- **Type:** Primary framework documentation, retrieved 2026-10-17.
- **Credibility:** High for documented framework objective and parameters.
- **Recency:** Current mutable documentation; not a dependency pin for this implementation.
- **Bias:** Medium: maintained by the implementing framework, not independent validation.

Documented loss variant identifier:

> `dr_grpo`

The documentation distinguishes GRPO loss normalizers and sampling settings.
It is a reference for comparison with the papers, not evidence that a custom
bridge/copy model can be passed unchanged into that trainer. The original native-loop prototype has been superseded: the current phase
uses a pinned TRL 1.16.0 subclass and native sampling from the project's complete
distribution. See the tag-specific source notes for the current API audit. Current documentation also lists evolving variants,
which should not be silently substituted for the chosen fixed-budget loss.
