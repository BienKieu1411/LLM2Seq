# Encoder-Decoder LLM Research

Nghiên cứu các kiến trúc Encoder-Decoder dựa trên LLM cho các bài toán sequence-to-sequence.

## Cấu trúc Repository

### [`v1_encdec_adaptation/`](v1_encdec_adaptation/)

**Phiên bản 1**: Adaptation từ Decoder-Only LLM sang Encoder-Decoder.

- Chuyển đổi kiến trúc Decoder-Only (Qwen, GPT-2, ...) sang Encoder-Decoder bằng cách thêm Cross-Attention.
- Hỗ trợ Tied Embeddings, Cross-Attention Warmup, Unbalanced Architecture.
- Vietnamese Text Summarization trên VietNews dataset.

### [`llm2seq/`](llm2seq/)

**Phiên bản 2 (LLM2Seq)**: LLM2Vec Encoder + Lightweight Decoder.

- Dùng LLM đã chuyển sang encoder (LLM2Vec) để hiểu input ở mức token-level.
- Lightweight Transformer Decoder tự thiết kế với Cross-Attention vào encoder memory.
- Hỗ trợ Multi-Token Prediction (MTP) — Parallel và Cascaded.
- Hỗ trợ Knowledge Distillation — Sequence KD, Logits KL, Top-k KL.
- 4 cấu hình ablation: Baseline, KD-only, MTP-only, KD+MTP.

## Tài liệu nghiên cứu

- [`RESEARCH.md`](RESEARCH.md) — Tổng hợp nghiên cứu.
- [`Gemini_Research.md`](Gemini_Research.md) — Phân tích chi tiết.
- [`LLM2Seq_Implementation_Plan.md`](LLM2Seq_Implementation_Plan.md) — Kế hoạch triển khai LLM2Seq.
