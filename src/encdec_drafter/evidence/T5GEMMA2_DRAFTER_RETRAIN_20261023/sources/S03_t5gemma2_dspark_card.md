# S03 — T5Gemma 2 1B-1B DSpark Base

- Author: d0rj.
- Link: https://huggingface.co/d0rj/t5gemma-2-1b-1b.dspark-base
- Source role: target-specific model artifact and author-reported benchmark.

The card identifies base `google/t5gemma-2-1b-1b` as verifier and describes a 3-layer DSpark draft with Markov and confidence heads. It reports 200,000 mixed-domain training examples, six epochs, and 146,486 final steps. For DSpark Base it reports 40.55% teacher-forced acceptance and 2.418 expected accepted tokens. Its batch-size-one performance table reports gains on some tasks and regressions on others, showing that draft acceptance does not guarantee a latency gain. These are one author's results, not independent reproduction and not a PubMed/summary workload result.
