# Relational-key bridge: independent PubMed experiment

This folder has its own `relational_key` package and entrypoint. It reuses the
XOV experiment's data handling, encoder, decoder, grounded-copy head, optimizer,
and evaluation protocol, but replaces **only the bridge**. It does not import
the XOV package or change any existing experiment folder. All model loading is
local (`local_files_only=True`); the offline tests use tiny random fixtures.

## Hypothesis and architecture contract

For source encoder states (H_i), direct projection gives (X_i). A masked,
directional low-rank pair operator computes separate left and right terms:

```
L_i = SiLU(W_c RMSNorm(H_i) ⊙ W_l RMSNorm(H_{i-1}))  if both positions are content
R_i = SiLU(W_c RMSNorm(H_i) ⊙ W_r RMSNorm(H_{i+1}))  if both positions are content
D_i = W_up [L_i ; R_i]
K_i = X_i + g · cap(D_i, X_i),  0 < g <= 0.20
V_i = X_i
C_i = X_i
```

`K` is the memory from which each decoder cross-attention layer computes keys.
`V` supplies its values; `C` supplies grounded-copy context. Thus cross-attention
still selects source positions and copy still exposes source token identities;
the bridge changes the compatibility of a decoder query with a source position
when that position is part of a local ordered pair. Prefix and padding cannot
enter pair products and receive no residual. A nonzero up-projection and gate
allow CE gradients to reach every pair parameter from the first update. The
residual RMS is capped relative to the projected source, before each decoder
layer's memory normalization. Although copy **memory** is fixed, copy
probabilities may change downstream because altered attention changes the
decoder hidden states.

The encoder is already contextual; this operator does **not** recover
information proven absent from it. This is a finite-budget retrieval bias.
Local source composition has precedent in [Wu et al., ICLR 2019](https://arxiv.org/abs/1901.10430),
and encoder-decoder attention has measurable source structure in
[Manakul and Gales, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.739/).
Neither paper validates this exact bridge. A higher ROUGE score or broad novelty
claim requires an actual experiment.

## Matched run

`configs/relational_key_pubmed.yaml` preserves the current XOV PubMed model
paths, source files, prompts, decoding, and training settings. In particular,
it uses **batch 84 on one GPU, accumulation 1, three full epochs, no interface
warmup, gradient clip 3.0**. Its only config differences are the bridge and
output directory. It evaluates `last.pt`, not a selected best checkpoint.

On the server, run from the repository root on the spare GPU while the XOV
experiment uses the other GPU:

```bash
CUDA_VISIBLE_DEVICES=1 \
ROUGE155_SCRIPT="$PWD/src/rouge155/evaluate_rouge.py" \
PYROUGE_HOME_DIR=/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5 \
bash src/relational_key_bridge/scripts/run_pubmed_one_gpu.sh
```

The script writes a unique run directory under `src/relational_key_bridge/runs`,
trains, evaluates on test, and invokes the supplied ROUGE-1.5.5 script. Override
`RUN_DIR` and `EVAL_BATCH_SIZE` if needed. The last checkpoint and its own
`resolved_config.yaml` stay together.

## Falsification and controls

The bridge has a plausible role only if its residual survives cross-attention's
memory norm and key projection, changes the attention distribution for a fixed
query, and improves held-out ROUGE against a matched direct-projection run. Tiny
offline tests cover the first two properties and CE weight updates, not ROUGE.

If the score improves, run a same-parameter-budget control that keeps the pair
projections but removes adjacency (e.g. shuffled neighbors or a center-only
pair mask) before attributing the gain to source relation modeling. Compare
against the already measured direct-projection + grounded-copy result only when
its data, global batch, epochs, decoding, checkpoint, and scorer match this run.
The expected score is unknown.
