# Paper-grounded factuality evaluation

The headline hallucination result uses **AlignScore-nli_sp**, rather than the
repository's lexical entity/number diagnostic. AlignScore was proposed by Zha,
Yang, Li, and Hu at ACL 2023 as a reference-free factual-consistency metric for
a context/claim pair. The paper trains a unified alignment function on data
from NLI, QA, paraphrasing, fact verification, information retrieval, semantic
similarity, and summarization, then evaluates it on multiple unseen factuality
benchmarks.

Paper: <https://aclanthology.org/2023.acl-long.634/>  
Released implementation and checkpoint format:
<https://github.com/yuh-zha/AlignScore>

## Score used in this project

For a source document `x` and generated summary `y`, the released `nli_sp`
procedure is:

1. Split `x` into source chunks of approximately 350 words using complete
   sentences.
2. Split `y` into claim sentences.
3. Run the AlignScore three-way alignment head on every source-chunk/claim-
   sentence pair. The entailment probability is the class-0 probability in
   the released `nli_sp` checkpoint.
4. For every claim sentence, retain its maximum entailment probability over
   source chunks and average those maxima.

Let the resulting native score be `C`. `C` measures factual consistency and is
high-is-better. To make the paper table use a hallucination-oriented direction,
the evaluator reports:

```text
alignscore_consistency = C
hallucination_score    = 1 - C
```

Thus `hallucination_score` is low-is-better and zero is the best possible value.
It is a transformed factual-consistency score, **not** a calibrated percentage
of hallucinated tokens or a probability that a whole summary is hallucinated.

## Reproducible local evaluation

`src/rouge155/evaluate_alignscore.py` loads both artifacts from local paths:

```bash
PYTHONPATH=src \
  /Users/kieugiangbien/bienkieu_env/bin/python \
  src/rouge155/evaluate_alignscore.py \
  /path/to/last_test_predictions.jsonl \
  --model-path /models/roberta-large \
  --checkpoint-path /models/AlignScore-large.ckpt \
  --batch-size 32 \
  --device cuda:0 \
  --details
```

The script uses `TRANSFORMERS_OFFLINE=1` and `local_files_only=True`; a missing
model or checkpoint fails instead of downloading a different evaluator. The
output records the checkpoint, backbone, maximum pair length, source chunk
size, sentence splitter, and per-example scores. The same checkpoint and
settings must be used for every model, dataset, and ablation.

## Reporting protocol

For each of PubMed, arXiv, BookSum, and GovReport, report the mean
`hallucination_score` and its native `alignscore_consistency`, followed by a
paired bootstrap confidence interval over the same test examples. Also report
ROUGE-1/2/L and BERTScore so that a system cannot obtain a low hallucination
score by producing an empty or excessively short summary.

A small human or independent entailment/QA audit should be included to check
domain-specific errors that automatic AlignScore can miss. AlignScore is the
only automatic hallucination/factuality metric retained in the `rouge155`
evaluation folder; BERTScore remains a semantic similarity metric.

## Limitations to state explicitly

AlignScore is a learned evaluator and can have domain or terminology shift on
biomedical, book, and government text. The chunk/sentence aggregation may miss
cross-sentence relations, and truncating a pair can remove evidence. Therefore
the paper should disclose source truncation rates and treat AlignScore as an
automatic factual-consistency signal, with human inspection as corroboration.

## Citation

```bibtex
@inproceedings{zha-etal-2023-alignscore,
  title = "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function",
  author = "Zha, Yuheng and Yang, Yichi and Li, Ruichen and Hu, Zhiting",
  booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2023",
  pages = "11328--11348",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2023.acl-long.634/"
}
```
