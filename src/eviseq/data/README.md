# Data contract

EviSeq reads canonical JSONL records with `id`, `source`, and `target` fields.
WikiLingua is deliberately shared with V2 so the candidate and legacy result
use byte-identical splits. CNN/DailyMail and PubMed are copied and converted
only from an explicit local directory:

```bash
bash eviseq/run.sh prepare-cnndm /absolute/path/to/cnndm
bash eviseq/run.sh prepare-pubmed /absolute/path/to/pubmed
```

Preparation is deterministic. Training then runs an exact normalized ID,
source, and `(source,target)` cross-split leakage audit before loading models.
