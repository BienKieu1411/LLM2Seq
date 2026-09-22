# AFMR source-delivery ledger

This is an independent experiment built from `eviseq_new`. It keeps the encoder, AFMR bridge, grounded-copy mixture, prompts, CE objective, and generation settings of that baseline. It adds a causal ledger over coarse source regions to the final decoder cross-attention, output readout, and grounded copy.

## Hypothesis

Cross-attention retrieves source features for the current decoder state. Grounded copy converts aligned source positions into decoder-token probability. Neither component explicitly remembers which source evidence has already supported earlier summary tokens. On long inputs, that missing temporal state can cause repeated attention to already-delivered evidence and leave other relevant regions unused.

The ledger divides content tokens into contiguous 32-token regions. At summary step `t`, it estimates which regions the decoder is using from its prefix-conditioned hidden state. A sparse top-4 exposure distribution and a learned, capped write gate update the causal state:

`coverage_t = 1 - (1 - coverage_(t-1)) * exp(-write_t * exposure_t)`

The exposure state informs two output paths. A small, signed gate conditioned on the current decoder state decides whether prior exposure should favor continuing the same source region or favor moving elsewhere:

- a region prior becomes a source-token bias in the final decoder cross-attention;
- a separate region prior enters grounded-copy position scores before copy probabilities are normalized.

An optional bounded read residual is available for controlled experiments, but it is disabled in the main configuration. It acts directly before the LM head and could hide whether cross-attention and grounded copy are useful.

The gates start at the same conservative negative-coverage bias as the original ledger, but can reverse sign when a summary needs to continue using a region. With zero history, both corrections are exactly zero and the first prediction follows the original model. Earlier cross-attention layers remain unchanged, as do grounded copy's lexical/contextual matching and generate/copy mixture. The final cross-attention estimates exposure with its own query/key projections over pooled key memory, including the AFMR source bias. Copy estimates exposure from the final decoder state and pooled value memory. They share the causal write rule, but keep separate cached coverage states because they observe different representations. Training computes causal prefix exposure with an exclusive cumulative sum, avoiding a sequential loop over all target tokens.

This state measures **estimated source exposure**, not verified delivery of facts. It can be wrong when a region is attended but its content is not expressed in the summary.

## Research basis and claim boundary

Coverage in pointer-generator summarization established that repeated source use should be tracked across decoder steps, but that study needed an auxiliary coverage loss and found that directly damping past attention could hurt performance: [See et al., ACL 2017](https://aclanthology.org/P17-1099/). Copy-history modeling showed that recent copying may *help* continue a coherent phrase rather than always signal repetition: [Li et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.336/). Long-document summarizers also exhibit systematic source-position bias: [Wan et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.442/). These results motivate testing a conditional history signal; they do not establish that this CE-only mechanism will improve ROUGE.

The defensible contribution is therefore not “the first coverage mechanism.” If the experiment succeeds, the narrower contribution is a CE-only, prefix-conditioned source-region exposure mechanism with separate, signed cross-attention and copy gates in an independently pretrained encoder/decoder system.

## Defaults

```yaml
decoder:
  delivery_ledger:
    enabled: true
    region_width: 32
    rank: 128
    write_top_k: 4
    write_strength_init: 0.05
    write_strength_max: 0.25
    coverage_strength_init: 1.5
    coverage_strength_max: 3.0
    read_enabled: false
    read_strength_init: 0.05
    read_strength_max: 0.20
    copy_strength_init: 0.50
    copy_strength_max: 1.0
    cross_strength_init: 0.50
    cross_strength_max: 1.0
```

Prompt positions do not update either ledger state during teacher forcing. Generation stores both states beside the decoder cache and reindexes them when finished rows are compacted. Training uses only the original token-level cross-entropy objective; there is no auxiliary coverage loss, evidence label, candidate generation, or second decoder pass.

## PubMed run

```bash
cd /workspace/storage-shared/nlp/dungdx4/bien_projects/LLM2Seq-main
CUDA_VISIBLE_DEVICES=1 \
AFMR_ENCODERS=pplx \
PPLX_ENCODER=/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b \
DECODER_MODEL=/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B \
PROCESSED_DATA_DIR="$PWD/src/eviseq_new/datasets/pubmed" \
AFMR_DELIVERY_LEDGER=true \
AFMR_OUTPUT_DIR="$PWD/runs/delivery_ledger_full" \
ROUGE155_SCRIPT="$PWD/src/rouge155/evaluate_rouge.py" \
PYROUGE_HOME_DIR=/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5 \
bash src/afmr_delivery_ledger/scripts/run_pubmed_pair.sh
```

The script runs only PPLX by default, evaluates `last.pt`, and refuses to overwrite an existing output directory unless `OVERWRITE_OUTPUT_DIR=true` is explicitly set.

For two GPUs, set `CUDA_VISIBLE_DEVICES=0,1`. Training uses DDP; test evaluation runs one shard per GPU and merges the predictions before ROUGE. The PubMed YAML batch size is **per GPU**: with its current batch 74 and accumulation 1, two GPUs give a global batch of 148. Set `TRAIN_BATCH_SIZE=37` to keep the global batch at 74, or override `GRADIENT_ACCUMULATION_STEPS` explicitly if needed.

## Falsification criteria

Reject this direction if it does not beat the matched `eviseq_new` run across seeds, if both learned gates stay near their initialization or collapse to zero, or if output analysis shows premature suppression of regions needed for multi-token facts. The ledger may still overestimate delivery when it only observes a relevant region. A single favorable checkpoint is not enough to claim that the ledger contributes.
