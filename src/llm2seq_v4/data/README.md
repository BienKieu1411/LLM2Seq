# Data

LLM2Seq-v4 reads its WikiLingua JSONL splits from this directory:
`data/wikilingua/`. The v4 configuration records SHA-256 fingerprints for all
three files so the paper dataset cannot silently change.

The v4 training and evaluation flows are self-contained and do not depend on
the sibling `llm2seq_v2/` directory.
