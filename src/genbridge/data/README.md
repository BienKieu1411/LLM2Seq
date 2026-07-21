# Expected summarization datasets

The cleaned WikiLingua-VI JSONL splits are included in `wikilingua/`:

- `wikilingua/train.jsonl`: 13,999 examples
- `wikilingua/validation.jsonl`: 1,680 examples
- `wikilingua/test.jsonl`: 3,901 examples

They were converted from `src/llm2seq/datasets/wikilingua/{train,val,test}.json`.
Serialized WikiHow `smallUrl`/licensing objects are removed during conversion;
each source sentence remains on a separate line. Rebuild them with:

```bash
bash run.sh prepare-wikilingua --input-dir ../llm2seq/datasets/wikilingua
```

The training dataset derives greedy ROUGE-1/2 evidence labels from references.
Test references are never passed to the model.

LR-Sum is prepared separately with `bash run.sh prepare-lrsum` because it is a
different benchmark and must not be silently mixed with WikiLingua.

Current LR-Sum Vietnamese splits:

- `lrsum/train.jsonl`: 11,676 examples
- `lrsum/validation.jsonl`: 1,459 examples
- `lrsum/test.jsonl`: 1,460 examples

LR-Sum uses a one-sentence news prompt and an oracle budget of up to three
source sentences. WikiLingua uses the number of reference steps instead. Both
produce the same model inputs (`input_ids`, `unit_ids`, and evidence labels),
so this is a data/supervision profile rather than a dataset-specific model.
