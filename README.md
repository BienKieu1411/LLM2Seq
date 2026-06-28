# LLM2Seq

LLM2Seq is a mini-project on Vietnamese abstractive summarization. The project adapts pretrained LLM-based source encoders into an encoder-decoder summarization system through a gated residual memory adapter, then studies faster inference with self-distilled Multi-Token Prediction (MTP).

This repository contains the model code, training/evaluation scripts, ACL-style report, and a small web demo for running summarization with standard autoregressive decoding or verified MTP decoding.

## What This Project Contains

- `src/llm2seq/`: main LLM2Seq implementation, configs, training scripts, evaluation scripts, and WikiLingua data files.
- `src/T5Gemma/`: T5Gemma baseline code and configs.
- `App/backend/`: FastAPI backend for the demo app.
- `App/frontend/`: React/Vite frontend for the demo app.
- `Report/`: ACL-style LaTeX report, bibliography, figures, and compiled PDF.
- `Paper/`: reference papers used while writing the report.

The experimental results and analysis are documented in the report, so this README focuses on setup and repository usage.

## Core Idea

LLM2Seq uses three main parts:

1. A pretrained LLM-based source encoder reads the input document.
2. A gated residual memory adapter maps encoder states into decoder cross-attention memory.
3. A lightweight Transformer decoder generates the summary.

After the main summarizer is trained, Phase 3 freezes the main path and trains MTP heads. At inference time, the MTP heads draft future tokens and the main decoder verifies them.

## Web Demo

The demo lets you paste a Vietnamese source document, choose a decoding mode, set the maximum output length, and view latency/token statistics.

### Install Backend Dependencies

```bash
cd App/backend
python3 -m pip install -r requirements.txt
```

The backend loads checkpoints from Hugging Face as configured in:

```text
App/backend/config.yaml
```

If the model repository requires authentication, set:

```bash
export HF_TOKEN=your_huggingface_token
```

### Install Frontend Dependencies

```bash
cd App/frontend
npm install
```

### Run Backend and Frontend Together

From the repository root:

```bash
bash run.sh
```

Default URLs:

- Backend API: `http://localhost:8000`
- Frontend web app: `http://localhost:5173`

Stop both servers with `Ctrl+C`.

### Run Separately

Backend:

```bash
cd App/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd App/frontend
npm run dev
```

## Demo UI Theme

The demo frontend uses a light neo-brutalist academic-tool style: cream background, white cards, thick dark borders, hard offset shadows, and flat accent colors.

The full UI recreation guide is here:

```text
App/frontend/UI_THEME_SPEC.md
```

Use that file if another LLM or developer needs to rebuild the same visual style.

## Training and Evaluation

The main training pipeline is under:

```text
src/llm2seq/
```

Start by copying the environment template:

```bash
cd src/llm2seq
cp env.example.txt env.txt
```

Edit `env.txt` for local paths, checkpoint locations, Hugging Face settings, and Python executable.

Useful scripts:

```bash
bash smoke_check.sh
bash install_deps.sh
bash run_pipeline.sh
```

The main configs are:

```text
src/llm2seq/configs/wikilingua_qwen_phase1.yaml
src/llm2seq/configs/wikilingua_qwen_phase2.yaml
src/llm2seq/configs/wikilingua_qwen_phase3.yaml
```

The Llama-oriented configs and VLSP configs are also kept under `src/llm2seq/configs/`.

## Training Phases

- Phase 1 trains the adapter, decoder, global memory tokens, and LM head while keeping the encoder frozen.
- Phase 2 adds LoRA adaptation for the source encoder and continues summarization training.
- Phase 3 freezes the main summarizer and trains only the MTP module.

For implementation details, see:

```text
src/llm2seq/README.md
src/llm2seq/src/models/
src/llm2seq/src/inference/
```

## Report

The ACL-style report is in:

```text
Report/VDT_LLM2Seq.tex
Report/VDT_LLM2Seq.pdf
```

To compile the report from `Report/` with a local LaTeX setup:

```bash
cd Report
tectonic VDT_LLM2Seq.tex --keep-logs --keep-intermediates
```

The bibliography file is:

```text
Report/custom.bib
```

## Data

WikiLingua files used by the project are stored under:

```text
src/llm2seq/datasets/wikilingua/
src/T5Gemma/datasets/wikilingua/
```

The expected split files are:

```text
train.json
val.json
test.json
```

Each example contains source sentences and target summary sentences. Preprocessing scripts convert them into the format used by training and evaluation.

## Notes

- The demo backend currently loads the Qwen-based LLM2Seq checkpoint from Hugging Face.
- The app supports two decode modes: `autoregressive` and `mtp_verified`.
- Quantitative results should be read from the report rather than this README.
