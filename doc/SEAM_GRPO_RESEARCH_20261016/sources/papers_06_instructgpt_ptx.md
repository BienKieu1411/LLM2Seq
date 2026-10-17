# Source note: InstructGPT PPO-ptx auxiliary supervised objective

- **URL:** https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf
- **Authors / date / venue:** Ouyang et al.; NeurIPS 2022.
- **Credibility:** High. Peer-reviewed primary study with public NLP regression analysis and an explicit combined objective.
- **Recency:** Medium. 2022 is older than GRPO developments but remains direct evidence for auxiliary supervised gradients during RL.
- **Bias:** Medium. The authors evaluate their own RLHF training recipe and large GPT models.

## Evidence used

InstructGPT reports regressions on several public NLP evaluations after RLHF and reports reducing them by mixing PPO gradients with a language-model likelihood term on pretraining data. This supports an auxiliary supervised objective as a capability-retention mechanism. It is not evidence that cross-entropy on summary references preserves general language ability: the paper's auxiliary distribution is broad pretraining data, while summary-reference CE preserves the summarization distribution.

> “mixing PPO updates with updates that increase the log likelihood”

## Limitations for this decision

The method is PPO/RLHF on decoder-only GPT models, not GRPO or encoder-decoder summarization. The study does not determine a CE mixing weight for the proposed task.
