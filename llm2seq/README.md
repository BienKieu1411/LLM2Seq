# LLM2Seq: Converting LLM2Vec Encoders into Lightweight Encoder-Decoder Generators

## Overview

LLM2Seq xây dựng mô hình encoder-decoder mới:

- **Encoder**: LLM2Vec hoặc bất kỳ LLM nào đã chuyển sang encoder, trích xuất token-level hidden states.
- **Adaptor**: Layer Fusion + MLP + Optional EncStack, chuyển `d_enc → d_dec`.
- **Decoder**: Lightweight Transformer decoder tự thiết kế với Cross-Attention vào encoder memory.
- **MTP**: Multi-Token Prediction (Parallel / Cascaded) — bật/tắt qua config.
- **Distillation**: Knowledge Distillation từ teacher LLM (Sequence KD / Logits KL / Top-k KL) — bật/tắt qua config.

```text
Input x
  → LLM2Vec Encoder: H_enc
  → Adaptor: H_dec_memory
  → Lightweight Decoder: P(y_t | y_<t, x)
  → Optional MTP Heads / MTP Modules
  → Output y
```

## 4 Cấu hình ablation

| Config | MTP | KD | Mục đích |
|---|---|---|---|
| `baseline.yaml` | Off | Off | Encoder-decoder cơ bản |
| `kd_only.yaml` | Off | On | Đánh giá lợi ích KD |
| `mtp_only.yaml` | On | Off | Đánh giá MTP speedup |
| `kd_mtp_full.yaml` | On | On | Cấu hình đầy đủ |

## Installation

```bash
pip install -r llm2seq/requirements.txt
```

## Quick Start

### 1. Chuẩn bị dữ liệu

```bash
python -m llm2seq.src.data.preprocess \
    --dataset <dataset_name> \
    --output_dir llm2seq/data/processed \
    --source_field article \
    --target_field abstract
```

### 2. Train Baseline

```bash
python -m llm2seq.src.training.trainer --config llm2seq/configs/baseline.yaml
```

### 3. Train với KD

```bash
python -m llm2seq.src.training.trainer --config llm2seq/configs/kd_only.yaml
```

### 4. Train với MTP

```bash
python -m llm2seq.src.training.trainer --config llm2seq/configs/mtp_only.yaml
```

### 5. Train Full KD + MTP

```bash
python -m llm2seq.src.training.trainer --config llm2seq/configs/kd_mtp_full.yaml
```

## Project Structure

```text
llm2seq/
  configs/                    # YAML configs cho 4 cấu hình
  src/
    models/
      llm2seq_model.py        # Main model (Encoder + Adaptor + Decoder)
      encoder_wrapper.py      # LLM2Vec / HF LLM encoder wrapper
      adaptor.py              # LayerFusion + MLP + EncStack
      decoder.py              # Lightweight Transformer decoder (RoPE, RMSNorm)
      mtp_heads.py            # Parallel MTP heads
      mtp_cascaded.py         # Cascaded MTP module
    training/
      trainer.py              # Training loop
      losses.py               # Total loss (CE + KD + MTP)
      kd_loss.py              # Knowledge Distillation losses
      mtp_loss.py             # MTP loss computation
      scheduler.py            # LR scheduler
    inference/
      generate.py             # Autoregressive generation
      generate_mtp.py         # MTP-accelerated generation
      confidence_adaptive.py  # Confidence-based token acceptance
    data/
      dataset.py              # JSONL dataset
      collator.py             # Dynamic padding collator
      preprocess.py           # Data preprocessing
    eval/
      eval_bleu.py            # BLEU / chrF metrics
      eval_rouge.py           # ROUGE metrics
      eval_latency.py         # Latency / throughput benchmarks
      eval_acceptance.py      # MTP acceptance rate metrics
  scripts/                    # Shell scripts cho training/eval
  requirements.txt
```

## Kiến trúc Decoder

Mỗi decoder layer gồm:
1. **RMSNorm → Causal Self-Attention** (RoPE, KV-cache)
2. **RMSNorm → Cross-Attention** (vào encoder memory)
3. **RMSNorm → SwiGLU FFN**

Default config:
- `num_layers = 6`
- `hidden_size = 768`
- `num_heads = 12`
- `ffn_size = 3072`
