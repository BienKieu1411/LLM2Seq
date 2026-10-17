# Source note: PPO clipped policy objective

- **URL:** https://arxiv.org/abs/1707.06347
- **Authors / date / venue:** Schulman et al.; arXiv 2017. Primary algorithm paper.
- **Credibility:** High for the clipped policy-gradient definition.
- **Recency:** Low as a foundational source, but the policy ratio remains the basis used by GRPO.
- **Bias:** Low-medium. Algorithm paper with general RL experiments rather than language-model training.

## Evidence used

PPO's clipped surrogate compares the current policy's probability of an action with the probability under the behavior/old policy that generated that action. For sampled decoder tokens, the rollout distribution, saved old log-probabilities, and recomputed current log-probabilities must refer to the same categorical policy. The latter sampling caveat follows from the objective; PPO itself does not discuss modern top-p generation filters.

> “clipped surrogate objective”

## Limitations for this decision

The paper is not specific to language or seq2seq. Consult GRPO papers for the group-relative estimator and generation documentation for actual temperature/top-p transformations.
