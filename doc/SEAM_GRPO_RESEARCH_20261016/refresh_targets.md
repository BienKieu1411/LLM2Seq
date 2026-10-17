# Refresh targets

- Installed PyTorch/Transformers versions: repeat cached rollout versus full
  forward log-probability and gradient parity after upgrades.
- TRL and author GRPO code: check objective normalization, temperature handling,
  and any claimed new seq2seq support against the exact custom mixture.
- Pin TRL 1.16.0: custom rollout and private likelihood/buffer hooks are verified
  against this version. Re-run all 21 checks before accepting an upgrade, with
  particular attention to sampler repeats, CE accumulation, and old log-probs.
- ROUGE installation: compare the training proxy on accented Vietnamese and
  punctuation cases with the deployed Perl evaluator; retain measured mismatch.
- GRPO run: validation reward, final Perl ROUGE, CE, truncation, reward ties,
  length and actual quality. Do not promote pilot settings into evidence.
- vLLM rollout: require encoder/bridge/copy weight synchronization, request
  isolation, exact generation settings, and log-prob parity before integration.
- Reassess the CE coefficient and optional KL on validation if reward increases
  while readability or factual consistency declines.
