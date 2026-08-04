# EviSeq-KD audit checklist

Verified offline with `bienkieu_env`; no model weights were downloaded.

- [x] **Shapes and masks:** sequence/full/top-k KD reject incompatible
  `[B,T,V]`, `[B,T,K]`, masks, vocabulary IDs, duplicate IDs, and `K > V`.
  Padding and all-masked batches are finite and contribute zero gradient.
- [x] **Gradient flow:** teacher tensors are detached; gold EviSeq, pseudo CE,
  pseudo soft KD, and gold soft KD all remain connected to student parameters.
- [x] **Top-k probability mass:** cache schema v3 stores one full-vocabulary
  log normalizer per token. KD uses `K + 1` buckets (top-k plus `OTHER`) rather
  than renormalizing the student only inside top-k.
- [x] **Forward count:** gold and pseudo trajectories each use one supervised
  encoder-decoder forward; CE and KD reuse its logits.
- [x] **Teacher cache:** source hashes, unique IDs, split/model/top-k,
  tokenizer fingerprint, vocabulary mapping, schema, KD temperature, EOS, and
  token-position alignment are validated before training.
- [x] **EOS/truncation:** EOS is retained when `pad_token_id == eos_token_id`;
  a synthetic or truncation-induced EOS is excluded from soft KD.
- [x] **Paths:** dataset inputs resolve relative to the config; generated
  caches resolve relative to the launch working directory.
- [x] **Checkpoint/evaluation imports:** complete `epoch`, `best`, and `last`
  state dictionaries use the wrapper graph and the evaluation package imports
  without the separate EviSeq installation.

Regression command:

```bash
cd src/eviseq_kd
PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q tests
```
