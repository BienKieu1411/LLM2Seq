# Data

The runnable folder expects the locked split at:

- `data/wikilingua/train.jsonl` (13,999 examples)
- `data/wikilingua/validation.jsonl` (1,680 examples)
- `data/wikilingua/test.jsonl` (3,901 examples)

Every row must contain `id`, `source`, and `target`. The included copies are
the same files used by the current GenBridge/T5Gemma comparison. Training
writes their exact SHA-256 fingerprints into `data_manifest.json`.
