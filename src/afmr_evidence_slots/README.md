# AFMR Competitive Evidence Slots

This candidate changes only the bridge of `eviseq_new`. The encoder, Qwen cross-attention decoder, grounded-copy head, prompts, CE training objective, data path, and generation path are unchanged.

```text
pretrained encoder H0
  -> existing AFMR depth/feature key adaptation and focus prior
  -> controller-conditioned competitive evidence slots
  -> bounded low-rank key residual
  -> Qwen cross-attention decoder with H0-anchored values
  -> grounded-copy/LM mixture
```

## Bridge design

The document, decoder prompt, and output budget form the existing controller state `c`. The evidence-slot route then performs three steps:

1. **Competitive source read.** Thirty-two learned 256-dimensional slots use orthogonal row initialization scaled to unit RMS. A controller projection conditions the slots. For each attention head, every valid content token distributes responsibility across slots; each slot then renormalizes its assigned mass across source tokens. Prefix and padding tokens receive zero mass. This competition prevents the independent-slot collapse possible when every slot separately softmaxes over the same source.
2. **Slot mixing and source write-back.** One self-attention/MLP block mixes the slots. Every content token attends back to the refined slots, receiving a global evidence context. The write-back is projected through a rank-256 bottleneck.
3. **Key-only bounded update.** The write-back residual is added only to decoder key memory. If `M` is the existing AFMR key anchor and `D` is the slot residual, the bridge uses a controller gate `g` with initialization `0.05` and maximum `0.30`:

```text
scale = RMS(stop_gradient(M)) / sqrt(RMS(M)^2 + RMS(D)^2 + 1e-12)
K-memory = M + g * scale * D
V-memory = projection(H0)
```

Thus the added residual has at most `g` times the key-anchor RMS at each token. The output projection has a tiny nonzero initialization, so the complete slot/read/mix/write path receives CE gradients on the first backward pass. The controller-to-slot projection starts at zero while the slots start at unit RMS; prompt conditioning is learned without swamping slot identity at initialization.

The optional slot-derived source prior is implemented as a learned signed score from normalized `H0` and routing-written slot context. It is zero-centered and bounded. It is **off in the main recipes** (`prior_enabled: false`) because `source_bias` also enters grounded copy; enabling it would mix a key-memory experiment with a copy-bias change. The inherited AFMR focus prior remains unchanged. There is no auxiliary loss, hard top-k selection, contextual-value branch, candidate generation, distillation, or reference-derived inference input.

`architecture.bridge_mode: direct_projection` remains the controlled w/o-bridge arm. It returns before any AFMR or evidence-slot modules are built. Grounded copy remains enabled unless disabled separately.

## Default configuration

The base and PubMed recipes use:

```yaml
architecture:
  name: afmr_value_anchor
  evidence_slots:
    enabled: true
    num_slots: 32
    dim: 256
    num_heads: 4
    rank: 256
    refine_layers: 1
    gate_init: 0.05
    gate_max: 0.30
    prior_enabled: false
    prior_init: 0.05
    prior_max: 0.30
  contextual_value:
    enabled: false

training:
  salience_loss_weight: 0.0
```

PubMed writes to `runs/afmr/pubmed_evidence_slots_value_anchor_copy`; the queue writes to `runs/afmr/pubmed_pair_evidence_slots_copy`. Checkpoints record the complete nested slot settings and reject a mismatched graph.

## PubMed train and evaluation

The sequential wrapper uses local model folders and supports the same one/two-GPU execution path as the base project. To run only the PPLX encoder arm:

```bash
cd src/afmr_evidence_slots
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
CUDA_VISIBLE_DEVICES=0,1 \
AFMR_ENCODERS=pplx \
PPLX_ENCODER=/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/path/to/Qwen3-0.6B \
PUBMED_SOURCE_DIR=/path/to/pubmed \
bash scripts/run_pubmed_pair.sh
```

Use a fresh output directory for every architecture run. The wrapper protects existing artifacts unless `OVERWRITE_OUTPUT_DIR=true` is set deliberately. It trains and evaluates `last.pt` with greedy decoding (`temperature: 0`, `top_k: 0`, `top_p: 1`).

For a direct command:

```bash
cd src/afmr_evidence_slots
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python3 -m eviseq_afmr.cli train configs/afmr_pubmed.yaml

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
python3 -m eviseq_afmr.cli evaluate \
  configs/afmr_pubmed.yaml \
  runs/afmr/pubmed_evidence_slots_value_anchor_copy/last.pt \
  runs/afmr/pubmed_evidence_slots_value_anchor_copy/last_test_predictions.jsonl \
  --split test
```

To run the separately trained direct-projection control, set `AFMR_BRIDGE_MODE=direct_projection` and a distinct `AFMR_OUTPUT_DIR` in the queue. Do not disable a trained bridge only at evaluation and report that as the retrained ablation.

## Other datasets and local models

`configs/afmr_arxiv.yaml`, `afmr_cnndm.yaml`, `afmr_wikilingua.yaml`, and `8192_avg.yaml` inherit the same evidence-slot graph and have distinct output names. Model paths must point to local folders; the run wrappers do not need to download a model. The existing preparation scripts preserve the source/target schema and prompt contract.

ArXiv one- or two-GPU run:

```bash
cd src/afmr_evidence_slots
PYTHON=python3 \
CUDA_VISIBLE_DEVICES=0,1 \
ARXIV_SOURCE_DIR=/path/to/arxiv \
ENCODER_MODEL=/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/path/to/Qwen3-0.6B \
bash scripts/run_arxiv.sh
```

## Verification

The focused offline tests use tiny random backbones and download nothing. They cover competitive normalization, prefix/padding masks, all-invalid rows, slot diversity, first-backward gradients across slot/read/mix/write/key/prior parameters, the final relative-RMS bound, deterministic train/eval bridge behavior, exact H0 value anchoring, and an unchanged direct-projection control.

```bash
cd /absolute/path/to/LLM2Seq
PYTHONPATH=src/afmr_evidence_slots \
/Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q \
  src/afmr_evidence_slots/tests/test_afmr_evidence_slots.py \
  src/afmr_evidence_slots/tests/test_afmr_value_anchor.py \
  src/afmr_evidence_slots/tests/test_afmr_gradients.py \
  src/afmr_evidence_slots/tests/test_afmr_runtime_tiny.py
```

These tests verify graph correctness and gradient flow. They do not establish a ROUGE gain; the candidate still requires a fresh controlled PubMed run against both `eviseq_new` full and the retrained direct-projection + grounded-copy control.
