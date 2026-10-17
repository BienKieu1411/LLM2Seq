# F5: source intervention can define a diagnostic, not guaranteed content reward

Status: proposed experiment, not validated.

Independent precedents support different components: [CMDP](../sources/04_cmdp.md), [source coverage](../sources/18_source_coverage.md), [CPMI](../sources/19_cpmi.md), with [ROUGESal](../sources/17_rougesal.md) for salient-content/XE mixing. None establishes frozen SEAM true-vs-wrong-source margin as a successful length-conditioned GRPO reward.

Proposed [specification](../2026-10-17_reward_update.md): content ROUGE, MAE divided by a declared word-error budget, optional bounded fixed-teacher source-discrimination margin. Do not use current actor attention as self-reward. Compare source-sensitive score against copying, wrong-number and negation controls before introducing it into policy reward.

Falsifier: the source score primarily rewards names, copying, summary length or teacher preferences; or its addition worsens validation ROUGE/length despite increasing reward. In those cases keep the baseline and do not claim the diagnostic as factuality.
