# F4: source-side and decoder-side control should be tested separately

Status: supported prior-art direction; exact SEAM dual prompts untested.

Independent primary papers: [CTRLsum](../sources/11_ctrlsum.md), [GSum](../sources/12_gsum.md), [two-sided length integration](../sources/13_two_sided_length.md), [LAAM](../sources/15_laam.md). They implement different mechanisms and do not collectively prove the proposed prompt template. Author code is implementation context, not independent evidence of gains.

Current design: [dual-role prompts](../2026-10-17_dual_prompt_update.md). Same requested words, distinct reading/writing instructions, current bridge, content+length GRPO and gold CE. Compare D/E/ED under the same visible source and declared compute.

Falsifier: ED fails to improve adherence/content over D or any apparent gain vanishes after matching source retention and training compute. Numeric bridge and architecture superiority remain underdetermined.
