# S05 — DSpark algorithm documentation

- Publisher: vLLM Speculators project.
- Link: https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md
- Source role: official algorithm documentation and implementation guide.

DSpark augments a block-parallel DFlash draft with a low-rank Markov head that conditions each position's logits on the previous token, and a confidence head that predicts per-position acceptance. The confidence scores can help choose how much of a block to verify. This is a plausible follow-up because independent parallel slots may lose within-block dependencies. It is not the first retrain here: first establish cache correctness and measure the EAL baseline, then justify more draft compute by latency.
