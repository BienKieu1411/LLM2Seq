# Refresh targets

- XOV remains untested at task level. Do not replace that status with synthetic probe success.
- Implement only after the research design is taken forward; no production source has been restored here.
- Measure real-tokenizer gap and many-to-one alignment frequencies on the intended training/validation corpus.
- Run actual CE gradient/weight-update, copy-state anchoring, cache, BF16 and DDP regression checks.
- Record exact backbone paths, tokenization, source budget, prompts, training updates, clipping, seeds and checkpoint rule for matched controls.
- Track residual effect after RMSNorm, W_V and attended output, not just raw residual norm.
- Capture validation ROUGE-1.5.5 and uncertainty before test evaluation. Retain a kernel-1 mechanism test only if the first pilot gains.
- Reassess additional capacity or projection changes only after measured failures; no gain is established by the cited precedents.
