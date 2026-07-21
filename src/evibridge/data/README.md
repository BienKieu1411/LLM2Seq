# Included summarization datasets

The standalone folder contains the same cleaned JSONL splits used by the
controlled Qwen and T5Gemma experiments:

- `processed/train.jsonl`: 13,999 examples
- `processed/validation.jsonl`: 1,680 examples
- `processed/test.jsonl`: 3,901 examples

Each source sentence is separated by a newline. The WikiLingua dataset profile
preserves those boundaries and derives greedy ROUGE-1/2 evidence labels from
the training reference. Other profiles can use sentences, grouped sentences,
or paragraphs without changing the architecture. Test references are never
passed to the model.

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
