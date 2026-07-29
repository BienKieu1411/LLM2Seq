# EviSeq V2

EviSeq V2 extends the EviSeq architecture with two new summarization-specific training mechanisms:

1. **Evidence-Focused Hard Contrastive Learning:** Replaces the saturated cross-document retrieval loss with vectorized, within-document multi-positive InfoNCE. Every oracle evidence sentence is a separate positive; the four most confusable non-evidence sentences are negatives, with current salience false positives prioritised during mining. The projection head is training-only.

2. **Candidate Summary Ranking (Phase 3):** An optional third training phase similar to BRIO (Bringing Order to Abstractive Summarization). After the standard cross-entropy training, the model is trained to rank a set of offline-generated candidate summaries based on their length-normalized log probabilities. The model uses a pairwise margin ranking loss to learn that higher-quality candidates (e.g. by ROUGE) should receive higher probabilities.

The main encoder uses **selective evidence-aware bidirectionality**: lower Qwen blocks keep the exact pretrained causal path and only the top six blocks compute the additional noncausal evidence view. This avoids paying for two quadratic attention paths in every encoder layer.

## Usage

**1. Data Preparation**
Same as V1. See `data/README.md`.

**2. Full PubMed pipeline (1 + 3 + 2 epochs)**
One command performs interface warm-up, full fine-tuning, offline candidate
generation, and two final ranking epochs. Candidate generation uses one greedy
plus four sampled summaries for each of a deterministic random 40k-document
TRAIN subset (at most 200k candidates). The long source is encoded once per
batch and reused by all candidates. Candidate-only decoding uses at most 384
tokens and top-k 64 plus top-p 0.95; final validation/test decoding remains at
the unchanged 512-token limit.
```bash
bash eviseq_v2/run.sh pubmed --overwrite-output-dir
```

The two checkpoints are deliberately separate:

- `runs/eviseq_v2/pubmed_qwen3_evidence/last.pt`: after warm-up + full fine-tune.
- `runs/eviseq_v2/pubmed_qwen3_evidence/ranking/last.pt`: final ranked model.

The runner automatically evaluates the ranked checkpoint. If candidate
generation or ranking fails, the root `COMPLETE` marker is not written.

**3. Optional standalone candidate regeneration**
```bash
python -m eviseq_v2.generate_candidates \
    --config configs/pubmed.yaml \
    --checkpoint runs/eviseq_v2/pubmed_qwen3_evidence/last.pt \
    --output runs/eviseq_v2/pubmed_qwen3_evidence/ranking/candidates.jsonl \
    --num-candidates 5 --max-examples 40000
```

Set `ranking.max_examples: 0` only for an intentional full-corpus run. On
PubMed that means roughly 600k autoregressive candidates and can take days even
on a B200; it does not change the deployed one-encoder/one-decoder graph.
