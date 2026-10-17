# F2: length gains do not establish content gains

Status: supported risk; choice of best SEAM objective unresolved.

Independent evidence: [CMDP](../sources/04_cmdp.md) separates similarity from requirement costs; [LARFT](../sources/05_larft.md) reports some general-capability regression; [reward study](../sources/07_reward_study.md) observes dataset-dependent content effects; [GR3](../sources/08_gr3.md) challenges naïve additive penalties. These are distinct tasks and are not interchangeable performance replications.

Use content-aware GRPO and report length error and ROUGE separately. Validate trade-offs instead of choosing a checkpoint solely from a sum of metrics. Fixed-cost and adaptive-dual recipes remain experiment choices, not proven solutions for this model.

Falsifier of the proposed recipe: apparent aggregate gains disappear by dataset or coincide with a significant ROUGE loss or infeasible token caps.
