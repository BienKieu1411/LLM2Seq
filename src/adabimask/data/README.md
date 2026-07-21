# Included WikiLingua splits

The standalone package includes the Vietnamese WikiLingua splits previously
stored under `llm2seq/datasets/wikilingua`, converted to JSONL with the fields
`id`, `source`, `target`, and `task`.

| Split | Examples |
|---|---:|
| train | 13,999 |
| validation | 1,680 |
| test | 3,901 |

Before packaging, source and target text were passed through
`adabimask.data.clean_wikihow_metadata`. This removes serialized wikiHow image
objects (for example `smallUrl`, `bigUrl`, dimensions, and licensing HTML)
and normalizes whitespace around punctuation. One empty raw test record was
discarded during conversion. The runtime loader applies the same idempotent
cleaner again as a safety check.
