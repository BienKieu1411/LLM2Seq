# LLM2Seq-v3

LLM2Seq-v3 extends LLM2Seq-v2 with **target-free source-utilization
contrastive learning** to fix the cross-attention sinking problem:

```text
Qwen3-Embedding-0.6B encoder (causal architecture, embedding-trained)
        ↓ token hidden states from several depths
lexical bank ───────────────────────────────────────────────┐
semantic bank ──────────────────────────────────────────────┤
8-layer bidirectional + sentence-context summary bank ──────┤
        ↓                                                   │
Qwen3-0.6B autoregressive decoder                           │
        ↓ every decoder layer independently attends 3 banks│
        ↓ token-adaptive depth router fuses attention outputs ─┘
        ↓ cross_gate_init=0.30
summary
```

## Supported encoder systems

| Profile | Attention before our adapter | Width / layers | Source recipe | Intended role |
|---|---|---:|---|---|
| Qwen3-Embedding-0.6B | causal-derived embedding backbone | 1024 / 28 | Vietnamese summary instruction + EOS; `mean_last` mining | Main conversion claim |
| PPLX Embed v1 0.6B | native bidirectional | 1024 / 28 | raw document, tokenizer-default specials, mean mining | Shape-matched strong encoder candidate |
| Nemotron 3 Embed 1B | native bidirectional | 2048 / 16 | exact `passage: ` prefix, tokenizer-default specials, mean mining | Larger multilingual upper bound |

PPLX is the first encoder to try for score: it matches Qwen's width and depth,
so the adapter and decoder do not have to learn a 2048-to-1024 family/width
alignment. Nemotron uses cross-attention every two decoder layers to keep its
complete training graph below the declared 2B budget. The runtime preflight is
still authoritative.

Both new embedding checkpoints are already bidirectional. Therefore, their
results cannot be presented as evidence that our adapter converted a causal
LLM. They are strong encoder-system comparisons. The original causal-derived
Qwen profile remains necessary for that claim; if PPLX/Nemotron becomes the
main accuracy model, the contribution must instead be framed as a
summary-specialized embedding-LLM-to-generative-LLM bridge.

For each native-bidirectional encoder there are two versions:

- `hiroute`: 8 task-refinement blocks, lexical/semantic/summary banks, and
  token-adaptive routing.
- `native_light`: one memory bank and 2 task-refinement blocks, while retaining
  multi-depth fusion, sentence context, salience supervision, contrastive
  alignment, and source-swap training.

The light profile is a required decision control, not a weaker afterthought:
an already bidirectional encoder may overfit WikiLingua when eight more blocks
and three full-length memories are added. Use held-out ROUGE-2 to choose; do
not assume the larger adapter wins.

PPLX custom code is pinned to Hub revision
`2c4d510dd4a732063c31a0f70193e35067b51fd8`; Nemotron is pinned to
`a5e0f804b9e90a1ca6784ecbf6e41595774fc834`. The same revision is passed to
config, tokenizer, and weights. `check-model` also mutates a future token and
requires an earlier final-layer token state to change for every encoder
declared bidirectional. This checks actual attention behavior rather than
trusting a config flag.

Official references: [PPLX Embed model card](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b),
[PPLX Embed paper](https://arxiv.org/abs/2602.11151),
[Nemotron 3 Embed model card](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16),
and [NVIDIA's embedding recipe](https://docs.nvidia.com/nemotron/nightly/nemotron/embed/README.html).

## What changed from LLM2Seq-v2

### Target-free prompt/source InfoNCE

The core v2 weakness: `cross_gate_init=0.1` means the decoder barely uses
cross-attention. Over 28 layers, the source signal gets diluted and the decoder
relies on its language model prior rather than the actual source document.

The decoder representation is collected at the final fixed prompt token,
before teacher forcing exposes any summary token. For each batch:

- **Positive pair**: `(adapter_memory_i, final_prompt_state_i)`
- **Negative pairs**: `(adapter_memory_j, final_prompt_state_i)`, `j != i`

All examples use the same prompt. If the decoder ignores cross-attention, its
prompt states are source-independent and cannot solve InfoNCE. Target-summary
hidden states are deliberately excluded to prevent a semantic retrieval
shortcut.

The source side blends masked token-mean pooling with the final valid/EOS
state. This preserves distributed evidence while using the global pooling
position on which Qwen3-Embedding was pretrained. The blend is learned per
hidden channel and remains entirely target-free.

### Same-target source-swap ranking

Each batch receives a second decoder pass with the source memory replaced by
the most similar wrong source in the physical batch, while decoder
inputs and gold targets remain unchanged. This target-free hard negative is
more demanding than an arbitrary document swap. The objective
prefers lower sequence NLL under the correct source than under the wrong source:

```text
L_swap = softplus((NLL_correct - NLL_wrong + margin) / temperature)
```

Since both passes see exactly the same target prefix, the only usable signal is
the cross-attended source memory. This training-only computation does not alter
the deployed one-encoder/one-adapter/one-decoder graph.

### Other improvements

1. **Label smoothing** (`label_smoothing=0.1`) regularizes cross-entropy.
2. **Higher cross-gate** (`cross_gate_init=0.3`) — larger initial contribution.
3. **Stronger main adapter** (8 bidirectional layers plus hierarchical
   sentence context; the single-bank control uses 6 layers).
4. **Higher encoder LR** (`1.2e-5` vs `8e-6`) — encoder is causal LLM
   with embedding training, needs stronger adaptation for seq2seq.
5. **Adjusted training schedule**: warmup 2 epochs (down from 3),
   `warmup_ratio=0.08`, `weight_decay=0.03`.

### HiRoute-v3: route attention outputs, not raw memories

The three full-length banks remain lexical, semantic, and bidirectionally
refined summary memory. Unlike v2, they are never averaged before source
matching. Every cross-attention layer computes one independently normalized
attention result per bank with shared pretrained Q/K/V/O weights, then applies
its learned depth routing distribution to the three context vectors:

```text
CrossAttn(query, lexical) ─┐
CrossAttn(query, semantic) ├─ query-adaptive depth router → gated residual
CrossAttn(query, summary) ─┘
```

This preserves bank-specific attention peaks without tripling cross-attention
parameters. Each layer starts exactly from its lexical/middle/summary depth
prior. Zero-initialized query and bank-output scorers then learn a different
mixture for every decoder token from both its current hidden state and the
three post-attention context vectors; their combined logit delta is bounded to
avoid abrupt collapse. Lexical and semantic memories also receive separately
gated, position-aligned residuals from the fully bidirectional summary stream,
so neither bank remains causal-only. It does increase multi-bank attention FLOPs and KV-cache size by
approximately three relative to single-bank v3. A mild global balance loss
prevents all decoder depths from collapsing onto one bank while still allowing
early/middle/late layers to specialize. Prompt InfoNCE uses the decoder's
observed mean routing distribution rather than an unrelated uniform bank
average; source-swap ranking exercises all routed attention paths.

### Deployable parameter counts

| Profile | Encoder | Decoder | Adapter | Training-only Head | Cross-attn | Deployable total |
|---|---:|---:|---:|---:|---:|---:|
| Main adaptive output-routed HiRoute | 595.777M | 596.050M | 159.697M | 0.527M | 176.426M | 1.528B |
| Single-bank control | 595.777M | 596.050M | 108.813M | 0.527M | 176.225M | 1.477B |

The main graph has 1,527,949,729 deployable parameters and 1,528,477,089
training parameters. The prompt projection head is training-only. The model
still deploys exactly one encoder, one adapter, and one decoder. The official
T5Gemma2-1B-1B model card declares a rounded 2B parameters, so the main v3
graph is about 23.6% smaller by the declared model-size convention. Evaluation
also counts the instantiated candidate graph directly; the paper report is
invalid unless that measured count remains below the target.
For the final claim, `final-audit` does not use the rounded 2B value at all: it
compares V3's measured deployable parameters with the unique parameter
elements of the actually loaded T5Gemma checkpoint. It also requires V3's
full training-time total (including the 0.527M alignment head) to be smaller;
the paper claim therefore does not depend on excluding that head.

## Training

- Phase 1: 2 epochs, freeze both backbones, train adapter + cross-attention + alignment head.
- Phase 2: 12 epochs, full fine-tune everything.
- Contrastive warmup: linearly ramp both objectives over Phase 1; Phase 2 starts at full weight.
- Physical batch 64, source length 3072, BF16 autocast.
- Training diagnostics use `rouge==1.0.0`; paper scores use Perl ROUGE-1.5.5.
  The launcher stores separate diagnostic and paper gap reports and refuses to
  declare a paper win when the backend or test-set size differs from T5Gemma.
  It also verifies the canonical test fingerprint (`03fba...3558`), that the
  checkpoint matches the evaluated test manifest, and that the measured
  deployable parameter count is below the declared 2B target.

## Commands

```bash
cd src/llm2seq_v3

bash run.sh setup
bash run.sh test

# Main output-routed HiRoute smoke/full runs
bash run.sh smoke --overwrite-output-dir
bash run.sh pipeline --overwrite-output-dir

# Explicit aliases for the same main profile
bash run.sh smoke-hiroute --overwrite-output-dir
bash run.sh hiroute --overwrite-output-dir

# Single-bank controls
bash run.sh smoke-single-bank --overwrite-output-dir
bash run.sh single-bank --overwrite-output-dir

# Native-bidirectional encoder smoke tests (run on the B200 host)
bash run.sh smoke-pplx --overwrite-output-dir
bash run.sh smoke-pplx-native --overwrite-output-dir
bash run.sh smoke-nemotron --overwrite-output-dir
bash run.sh smoke-nemotron-native --overwrite-output-dir

# Verify the real checkpoint's dimensions, parameter budget, and future-token
# influence before a full run. This command intentionally loads checkpoints.
bash run.sh check-model --config configs/pplx_embed_v1_0_6b_hiroute.yaml
bash run.sh check-model --config configs/nemotron3_embed_1b_hiroute.yaml

# Held-out 2k pilot: same physical batch as main, then a matched single-bank
# control. This is the fast decision gate before spending a full B200 run.
bash run.sh pilot-all --overwrite-output-dir
# Result: runs/llm2seq_v3/pilot_comparison.json

# Rank the three HiRoute encoder systems on the same 512-example held-out slice
bash run.sh encoder-pilot-all --overwrite-output-dir
# Result: runs/llm2seq_v3/encoder_pilot_comparison.json

# Decide whether HiRoute is worth its cost for already-bidirectional encoders
bash run.sh pplx-pilot-all --overwrite-output-dir
# Run the Nemotron pair only if PPLX does not already close the target gap
bash run.sh nemotron-pilot-all --overwrite-output-dir

# Or run both pairs sequentially so the B200 is never idle
bash run.sh native-encoder-pilot-all --overwrite-output-dir
# Results: pplx_hiroute_vs_native_light.json and
#          nemotron_hiroute_vs_native_light.json

# Evaluate existing checkpoint
bash run.sh eval
bash run.sh eval --max-samples 20

# Paper ROUGE after exporting PYROUGE_HOME_DIR
bash run.sh rouge155

# Final artifact-to-artifact proof. The T5Gemma evaluator must be the current
# version so metrics.json contains its exact parameter count and test hash.
bash run.sh final-audit \
  ../T5Gemma/eval_outputs/full_test/predictions.rouge155.json \
  ../T5Gemma/eval_outputs/full_test/metrics.json
```

Every smoke command now writes `smoke_gate.json` and exits non-zero when it
detects empty/fixed-prefix/repetitive generation, a disconnected
cross-attention residual, wrong-source preference, collapsed three-bank
routing, or a conservative total parameter count above the declared target.
Passing this gate only proves that the flow is healthy; `pilot-all` remains the
held-out decision gate. The pilot comparison also requires identical test
fingerprints, checkpoint integrity, and matched validation scope before it can
recommend HiRoute. The complete 3,901-example test is required for the paper comparison.
The final audit exits successfully only when both artifacts use Perl
ROUGE-1.5.5 on the identical locked split, V3 is strictly smaller by actual
runtime parameter count (both total and deployable), and V3 is strictly higher
on ROUGE-1/2/L.

## Ablations

```bash
bash run.sh ablation-no-contrastive --overwrite-output-dir
bash run.sh ablation-no-prompt-alignment --overwrite-output-dir
bash run.sh ablation-no-source-swap --overwrite-output-dir
bash run.sh ablation-cyclic-source-swap --overwrite-output-dir
bash run.sh ablation-cross-gate-0.1 --overwrite-output-dir
bash run.sh ablation-adapter-4layers --overwrite-output-dir
bash run.sh ablation-v2-control --overwrite-output-dir
bash run.sh ablation-pre-attention-routing --overwrite-output-dir
bash run.sh ablation-static-output-routing --overwrite-output-dir
bash run.sh ablation-no-routing-balance --overwrite-output-dir
bash run.sh ablation-no-branch-context --overwrite-output-dir
bash run.sh ablation-no-hiroute --overwrite-output-dir
bash run.sh ablation-mean-only-pooling --overwrite-output-dir
bash run.sh ablation-no-label-smoothing --overwrite-output-dir
bash run.sh ablation-no-fusion --overwrite-output-dir
bash run.sh ablation-no-bidirectional --overwrite-output-dir
bash run.sh ablation-random-cross --overwrite-output-dir
bash run.sh ablation-no-salience --overwrite-output-dir
bash run.sh ablation-sparse-cross --overwrite-output-dir
bash run.sh ablation-all --overwrite-output-dir
```

`no-contrastive` removes both source-utilization objectives.
`no-prompt-alignment` and `no-source-swap` isolate their individual effects;
`no-label-smoothing`, `cross-gate-0.1`, and `adapter-4layers` isolate the other
v3 changes. `v2-control` restores the complete v2 optimization recipe.
`pre-attention-routing` changes only the HiRoute routing location and is the
direct control for the output-routing contribution. `no-routing-balance`
isolates the anti-collapse regularizer. `no-branch-context` keeps HiRoute but
removes the two bidirectional residuals injected into lexical/semantic banks.
`no-hiroute` keeps eight bidirectional
layers and sentence context, removing only the extra banks/router; it is the
clean control for the complete HiRoute memory contribution.
`static-output-routing` retains the three independent bank attentions and
post-attention fusion while removing only token-conditioned routing.
`mean-only-pooling` isolates the use of Qwen3-Embedding's final/EOS state in
the contrastive head.
