# Source 05 — Liu et al. (2020), Admin

- URL: https://arxiv.org/abs/2004.08249
- Authors/venue: Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, Jiawei Han; EMNLP 2020.
- Accessed: 2026-09-23.
- Source type: primary optimization/initialization paper.
- Transfer confidence: relevant warning about residual dependence; no direct XOV evidence.

## Verified evidence

Admin analyzes residual dependence during Transformer training. Its abstract reports that a heavy residual dependency can destabilize training by amplifying parameter perturbations, while a light dependency can restrict the model's eventual potential. The paper summarizes this as: “heavy dependency on its residual branch makes training unstable.” (9 quoted words.) It proposes an adaptive initialization intended to stabilize early training and increase residual participation later.

## Implication for XOV

The bridge gate should be treated as an optimization control, not a novelty claim. A permanently tiny cap or overly damped residual can make XOV numerically present but functionally irrelevant; an initially large residual can disturb a pretrained direct path. This supports measuring residual/value norms over early updates and evaluating whether the gate opens, rather than choosing a fixed gate from intuition.

## Constraint

Do not tune the gate against test scores. Use validation CE to select at most one small initialization/schedule variant after the exact-identity pilot, and report whether the value residual survives the memory normalization.
