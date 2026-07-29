# EviSeq V2

EviSeq V2 extends the EviSeq architecture with two new summarization-specific training mechanisms:

1. **Evidence-Focused Hard Contrastive Learning:** Replaces the generic document-level InfoNCE loss with a within-document sentence-level InfoNCE. The model learns to pull the summary representation closer to actual evidence sentences while pushing it away from hard negative sentences (sentences that are semantically similar but do not contain necessary summary information). This directly optimizes factuality and evidence alignment.

2. **Candidate Summary Ranking (Phase 3):** An optional third training phase similar to BRIO (Bringing Order to Abstractive Summarization). After the standard cross-entropy training, the model is trained to rank a set of offline-generated candidate summaries based on their length-normalized log probabilities. The model uses a pairwise margin ranking loss to learn that higher-quality candidates (e.g. by ROUGE) should receive higher probabilities.

## Usage

**1. Data Preparation**
Same as V1. See `data/README.md`.

**2. Standard Training (Phases 1 & 2)**
Train with evidence contrastive learning (enabled by default in `configs/base.yaml`).
```bash
python -m eviseq_v2.training --config configs/pubmed.yaml
```

**3. Offline Candidate Generation**
Generate greedy and sampled candidates for Phase 3 ranking:
```bash
python -m eviseq_v2.generate_candidates \
    --config configs/pubmed.yaml \
    --checkpoint runs/eviseq_v2/pubmed_qwen3_evidence/last.pt \
    --output runs/eviseq_v2/pubmed_qwen3_evidence/candidates.jsonl
```

**4. Candidate Ranking (Phase 3)**
Update your config to enable ranking and point to the generated candidates file:
```yaml
ranking:
  enabled: true
  candidates_file: runs/eviseq_v2/pubmed_qwen3_evidence/candidates.jsonl
training:
  ranking_finetune_epochs: 1
```
Then rerun training. The runtime will automatically resume from `last.pt` and begin the `ranking_finetune` stage.
```bash
python -m eviseq_v2.training --config configs/pubmed.yaml
```
