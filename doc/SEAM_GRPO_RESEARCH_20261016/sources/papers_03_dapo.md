# Source note: DAPO paper

- **URL:** https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf
- **Authors / date / venue:** Yu et al., ByteDance Seed and collaborators; NeurIPS 2025.
- **Credibility:** High. Peer-reviewed conference paper with algorithm details, ablations, code, and dataset release.
- **Recency:** High as of 2026-10-17.
- **Bias:** Medium. Authors report strong gains for their own system; all headline evaluations are math reasoning.

## Evidence used

DAPO replaces GRPO's per-response token mean with a token-level sum divided by the number of valid tokens in the response group. Thus tokens in longer completions contribute proportionally more to that group's gradient. This can help learn from useful long trajectories, but can also give verbose summaries more influence unless the reward and length budget are well specified.

DAPO retains group reward standardization, so it is not identical to Dr. GRPO. The paper also introduces decoupled clipping, dynamic sampling to avoid groups where every reward is identical, and handling for overlong generations. Its large run uses 16 samples per prompt, learning rate 1e-6, and asymmetric clip ranges 0.2/0.28; these are large-scale math settings, not direct summarization defaults.

> “longer sequences can have more influence on the overall gradient update”

## Limitations for this decision

The paper reports no summarization or encoder-decoder GRPO experiment. The observed contribution of DAPO's changes is entangled with its math data, reward, model, and training system. Its token-mean recommendation is motivated by long chain-of-thought responses, which differ from concise summaries.
