# Encoder-Decoder LLM for Vietnamese Text Summarization

This project implements a hybrid model architecture (Encoder-Decoder) by adapting from Large Language Models (LLMs) with a Decoder-Only architecture, specialized for long text summarization tasks in Vietnamese.

## Features

- **Architecture Adaptation:** Convert Decoder-Only LLMs (Gemma, Qwen, Llama) to Encoder-Decoder.
- **Smart Initialization:** Support initializing `cross-attention` from `self-attention` and Cross-Attention Warmup.
- **Optimization (Tied Embeddings):** Share embedding matrix weights, saving 10.5% of parameters.
- **Unbalanced Architecture:** Support Encoder and Decoder with different sizes (e.g., 9B-2B).
- **Vietnamese Dataset:** Integrate the `VietNews` dataset (143k articles).
- **Demo Interface:** Streamlit application displaying the summarization process and real-time performance analysis.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Usage Guide

### 1. Initialize the adapted model

The project uses YAML configuration files in the `configs/` directory.

Run smoke test (GPT-2):
```bash
python scripts/build_adapted_model.py \
  --config configs/smoke_test.yaml \
  --output_dir outputs/smoke-ed
```

Run with an actual model (Qwen 0.5B):
```bash
python scripts/build_adapted_model.py \
  --config configs/vietnews_qwen.yaml \
  --output_dir outputs/qwen-ed
```

### 2. Train the model (Summarization)

```bash
python scripts/train_summarization.py \
  --config configs/smoke_test.yaml \
  --model_dir outputs/smoke-ed \
  --output_dir outputs/smoke-ed-cnn
```

### 3. Evaluation (ROUGE & BERTScore)

```bash
python scripts/eval_summarization.py \
  --config configs/smoke_test.yaml \
  --model_dir outputs/smoke-ed-cnn
```

### 4. Launch Demo App

```bash
streamlit run scripts/demo_streamlit.py
```

## System Architecture

```mermaid
graph LR
A[Tokenizer] --> B(Encoder)
B -->|Cross Attention| C(Decoder)
C -->|Beam Search| D[Summarized Text]
```
