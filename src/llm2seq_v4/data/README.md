# Data

LLM2Seq-v4 reads its WikiLingua JSONL splits from `data/wikilingua/`. The v4
configuration records SHA-256 fingerprints for all three files so the paper
dataset cannot silently change.

CNN/DailyMail is copied into V4-owned storage and converted to the same
canonical `id/source/target` JSONL schema:

```bash
bash run.sh cnndm-prepare /absolute/path/to/cnndm
```

The source directory may contain `train.txt`, `val.txt`, and `test.txt` (the
format used by this project), or the documented `.jsonl`/`.json` aliases. Raw
copies are written to `data/raw/cnndm/`, canonical splits to `data/cnndm/`, and
`data/cnndm/manifest.json` records counts and SHA-256 fingerprints. Duplicate
IDs within a split and ID leakage between splits are rejected.

The v4 training and evaluation flows are self-contained and do not depend on
the sibling `llm2seq_v2/` directory.
