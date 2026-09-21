# AFMR source-delivery ledger

This is an independent experiment built from `eviseq_new`. It keeps the encoder, AFMR bridge, decoder cross-attention, grounded-copy mixture, prompts, CE objective, and generation settings of that baseline. It adds one mechanism after the decoder backbone: a causal ledger over coarse source regions.

## Hypothesis

Cross-attention retrieves source features for the current decoder state. Grounded copy converts aligned source positions into decoder-token probability. Neither component explicitly remembers which source evidence has already supported earlier summary tokens. On long inputs, that missing temporal state can cause repeated attention to already-delivered evidence and leave other relevant regions unused.

The ledger divides content tokens into contiguous 32-token regions. At summary step `t`, it computes prefix-conditioned relevance to each region, discounts regions in proportion to their accumulated delivery state, and obtains a distribution over remaining evidence. A learned, capped write gate then updates the causal state:

`coverage_t = coverage_(t-1) + write_t * p_t * (1 - coverage_(t-1))`

The same remaining-evidence signal is used in two bounded ways:

- a norm-preserving source read is added to the decoder hidden state before the LM head;
- a region prior is added to grounded-copy position scores before copy probabilities are normalized.

Cross-attention remains unchanged, and the lexical/contextual matching and generate/copy mixture remain unchanged. The new component performs temporal accounting rather than another static encoder-to-decoder projection.

## Research basis and claim boundary

Coverage in pointer-generator summarization established that repeated source use should be tracked across decoder steps: [See et al., ACL 2017](https://aclanthology.org/P17-1099/). Copy-history modeling showed that independent copy decisions lose useful temporal information: [Xu et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.336/). Long-document summarizers also exhibit systematic source-position under-coverage: [Ravaut et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.442/).

The defensible contribution is therefore not “the first coverage mechanism.” If the experiment succeeds, the narrower contribution is a CE-only, prefix-conditioned source-region delivery state shared by neural generation and tokenizer-aligned copying in an independently pretrained encoder/decoder system.

## Defaults

```yaml
decoder:
  delivery_ledger:
    enabled: true
    region_width: 32
    rank: 128
    write_strength_init: 0.05
    write_strength_max: 0.25
    coverage_strength_init: 0.10
    coverage_strength_max: 1.0
    read_strength_init: 0.05
    read_strength_max: 0.20
    copy_strength_init: 0.10
    copy_strength_max: 0.50
```

Prompt positions do not update the ledger during teacher forcing. Generation stores the ledger state beside the decoder cache and reindexes it when finished rows are compacted. Training uses only the original token-level cross-entropy objective; there is no auxiliary coverage loss, evidence label, candidate generation, or second decoder pass.

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

## Falsification criteria

Reject this direction if it does not beat the matched `eviseq_new` run across seeds, if learned write/copy/coverage strengths collapse close to zero, or if output analysis shows premature suppression of regions needed for multi-token facts. A single favorable checkpoint is not enough to claim that the ledger contributes.
