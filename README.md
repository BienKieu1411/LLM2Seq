# Encoder-Decoder LLM Research

Nghiên cứu các kiến trúc Encoder-Decoder dựa trên LLM cho các bài toán sequence-to-sequence.

## LLM2Seq Architecture Overview

![LLM2Seq Architecture](./llm2seq/figures/image.png)

LLM2Seq được thiết kế nhằm tối ưu hóa việc truyền thông tin từ Encoder sang Decoder mà không làm suy giảm chất lượng:

- **LLM2Vec Encoder**: Mô hình encoder hai chiều xử lý context cực dài (được đóng băng ở Phase 1, dùng LoRA ở Phase 2).
- **Gated Residual Adaptor (Cầu nối Encoder-Decoder)**: Chuyển đổi và chắt lọc không gian chiều (dimension) từ Encoder sang Decoder một cách mượt mà. Khối cầu nối này bao gồm các thành phần con:
  - *Layer Fusion*: Tích hợp các đặc trưng theo token từ nhiều tầng khác nhau của encoder.
  - *Gated Residual Projection*: Ánh xạ chiều dữ liệu kết hợp với skip-connection để giữ thông tin gốc.
  - *Salience Gate*: Bộ lọc nội dung (token-level), loại bỏ các token dư thừa giúp decoder bớt nhiễu.
  - *EncStack*: Tinh chỉnh lại bộ nhớ (memory refinement) trước khi đưa vào Cross-attention.
  - *Global Memory Tokens*: Các token toàn cục học được (learnable) gắn vào đầu dãy memory để tóm lược đại ý văn bản.
- **Lightweight Decoder**: Decoder Transformer tự thiết kế (ví dụ cấu hình LLM2Seq: 8 layers, 1024 hidden size) chuyên trách việc sinh văn bản.
- **MTP (Multi-Token Prediction) Heads**: Các module phụ trợ (được train riêng ở Phase 3) giúp tăng tốc sinh từ bằng kỹ thuật Speculative Decoding (Dự đoán nhiều token cùng lúc).

## Evaluation Results: LLM2Seq vs T5Gemma2-1B-1B

Dưới đây là kết quả so sánh hiệu năng của **LLM2Seq (Phase 2 - LoRA Encoder)** với **T5Gemma-1b-1b (LoRA)** trên tập test WikiLingua (3,901 mẫu).

**Cấu hình mô hình:**
- **LLM2Seq (Llama Encoder)**: Sử dụng LLM2Vec-Sheared-LLaMA-mntp làm Encoder + Decoder Transformer tự thiết kế.
- **LLM2Seq (Qwen Encoder)**: Sử dụng Qwen3-Embedding-0.6B làm Encoder + Decoder Transformer tự thiết kế.

| **Chỉ số (Metric)** | **LLM2Seq (Llama Encoder)** | **LLM2Seq (Qwen Encoder)** | **T5Gemma (1B-1B)** | **Đánh giá chi tiết**                 |
| :--------------------| :---------------------------:| :--------------------------:| :-------------------:| :--------------------------------------|
| **Tổng tham số**    | ~1.5B                       | ~1B                        | ~2B                 | *T5Gemma lớn nhất, Qwen gọn nhẹ nhất* |
| **ROUGE-1**         | 48.36                       | 53.91                      | **54.24**           | *Qwen kéo LLM2Seq ngang ngửa T5Gemma* |
| **ROUGE-2**         | 15.54                       | 20.74                      | **27.42**           | *T5Gemma vẫn vượt trội về độ mượt*    |
| **ROUGE-L**         | 29.05                       | 31.77                      | **33.89**           | *T5Gemma nhỉnh hơn một chút*          |

*Ghi chú: Điểm được lấy ở Phase 2 của các mô hình LLM2Seq.*

### Phân tích & Nhận xét
1. **Sức mạnh của Qwen Encoder**: 
   - Việc thay đổi Encoder từ Llama sang **Qwen** đã mang lại sức mạnh vượt bậc cho LLM2Seq. Điểm ROUGE-1 tăng phi mã từ 48.36 lên 53.91, trực tiếp cạnh tranh sòng phẳng với T5Gemma (54.24).
   - Tuy nhiên, T5Gemma-1B-1B vẫn là mô hình sinh ra câu từ mượt mà, tự nhiên nhất.
2. **Kết luận**: 
   - **T5Gemma-1B-1B** phù hợp khi cần một đoạn tóm tắt diễn giải chi tiết, văn phong mượt mà tự nhiên.
   - **LLM2Seq (Qwen)** là kiến trúc toàn diện: chất lượng ROUGE tương đương T5Gemma và năng lực tóm tắt súc tích, đi thẳng vào vấn đề.
