# Source note: Dr. GRPO

- **URL:** https://arxiv.org/html/2503.20783
- **Authors / date / venue:** Liu et al., Sea AI Lab, NUS, and SMU; arXiv v2, 2025-10-06. Critical-perspective primary paper; code is linked from the paper.
- **Credibility:** High for the derivation and its reported experiments; peer-reviewed workshop status was not established from the accessible source, so treat the version as an arXiv paper.
- **Recency:** High as of 2026-10-17.
- **Bias:** Medium. The authors propose Dr. GRPO; the main experiment is narrow and math-focused.

## Evidence used

The paper identifies two response/group weighting effects in common GRPO: dividing by each completion length makes negative-advantage long outputs receive less penalty, and dividing rewards by each prompt group's standard deviation increases the weight of low-variance prompts. Dr. GRPO removes both factors. Its practical masked loss divides by a fixed generation-budget constant rather than each sampled response's realized length. Its reported settings include eight responses per prompt, temperature 1, top-p 1, one inner update epoch, and symmetric clipping at 0.2.

These changes are a plausible starting point for summarization, but the reported results measure math reasoning/token efficiency. They do not prove the same dynamics for summaries, where length also controls coverage and concision.

> “remove the length and std normalization terms.”

## Limitations for this decision

This paper's empirical evidence is for decoder-only math reasoning. It argues for removing per-prompt standard-deviation scaling; it does not establish the best normalization for continuous summarization rewards. A global batch scale is a separate design choice and preserves relative weighting between prompts.
