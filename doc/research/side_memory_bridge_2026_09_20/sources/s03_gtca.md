# S03 — Gated Tree Cross-Attention

**Primary source:** Gao, Wang, and Ding, *Gated Tree Cross-Attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs*, ACL 2026.

URL: https://aclanthology.org/2026.acl-long.1629.pdf

## Evidence

- Page 1 introduces a “checkpoint-compatible gated cross-attention side branch” that reads cached chunk memory while leaving the backbone architecture unchanged.
- Pages 1 and 4 use a token update mask and staged training to control interference. The branch is added as a regulated update, rather than a hard replacement of the backbone state.
- Page 4 defines the update as a masked residual `H_post = H_pre + alpha_struct * mask * DeltaH`; this is useful evidence for restricting where an injected side signal can act.
- Table 4 shows removing the gate harms both syntax and broad competence in their Qwen and Llama experiments. The paper reports BLiMP gains while retaining general benchmark performance.

## Transfer to AFMR

Use the checkpoint-compatible side-path principle, but make `E` a source-evidence bank and attach it after the existing semantic cross-attention. Keep the original token K/V and copy path intact. A per-layer tanh gate can be zero-initialized; a controller or decoder-hidden mask can decide whether the side bank is useful at a target step. A short gate warmup can be tested as an optimization schedule, but the primary AFMR comparison should keep the objective CE-only.

## Caveats

GTCA relies on external parse trees, LoRA, and a three-stage MCQA adaptation recipe. Its syntax results do not prove summarization gains. The useful transferable evidence is the regulated side path and retention ablations, not the parser or training schedule itself.

