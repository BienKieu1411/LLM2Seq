# Data

LLM2Seq-v5 reads its WikiLingua JSONL splits from this directory:
`data/wikilingua/`. The v5 configuration records SHA-256 fingerprints for all
three files so the paper dataset cannot silently change.

The v5 training and evaluation flows are self-contained and do not depend on
the sibling `llm2seq_v2/` directory.

CNN/DailyMail is also self-contained. Prepare it from a local directory that
contains `train.txt`, `val.txt`, and `test.txt` (JSON object per line):

```bash
bash run.sh cnndm-prepare /absolute/path/to/cnndm
```

The command copies the raw files to `data/raw/cnndm/`, writes canonical
`id/source/target` JSONL files to `data/cnndm/`, checks duplicate and
cross-split IDs, and records counts plus SHA-256 fingerprints in
`data/cnndm/manifest.json`. It never reads the sibling T5Gemma processed-data
directory.
