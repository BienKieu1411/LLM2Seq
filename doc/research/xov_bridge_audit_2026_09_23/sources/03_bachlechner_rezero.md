# Source 03 — Bachlechner et al. (2021), ReZero

- URL: https://arxiv.org/abs/2003.04887
- Published version: https://proceedings.mlr.press/v161/bachlechner21a/bachlechner21a.pdf
- Authors/venue: Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Garrison Cottrell, Julian McAuley; UAI 2021, PMLR 161.
- Accessed: 2026-09-23.
- Source type: primary optimization/initialization paper.
- Transfer confidence: relevant to residual initialization dynamics; not evidence for XOV quality.

## Verified evidence

ReZero uses `x_{l+1} = x_l + alpha_l F(x_l)` with `alpha=0` at the start, so the initial network is an identity path. The authors explicitly note: “Initially the gradients for all parameters defining F vanish, but dynamically evolve to suitable values during initial stages of training.” (24 quoted words.) They report that zero initialization of the residual weight matters in their Transformer experiments; initializing the weight to one did not give the same convergence behavior.

## Implication for XOV

An identity-preserving residual is a defensible baseline for a pretrained bridge. However, a zero gate or zero output projection creates a cold-start route: the first backward pass can update only the parameter directly exposed to the residual output, while lower lexical layers receive no gradient through a zero downstream map. The historical XOV probe already records this pattern. A nonzero gate does not remove a zero `lexical_up` Jacobian; the two initialization choices must be analyzed separately.

## Constraint

Retain an exact identity initialization for the primary safety run and log first-step and second-step gradients for `lexical_down`, convolution, `lexical_up`, and the gate. Do not call nonzero gradient or nonzero residual evidence of a task benefit.
