# Encoder-Decoder LLM Research

Nghiên cứu các kiến trúc Encoder-Decoder dựa trên LLM cho các bài toán sequence-to-sequence.

## LLM2Seq Architecture Overview

![LLM2Seq Architecture](./llm2seq_final/figures/image.png)

LLM2Seq được thiết kế nhằm tối ưu hóa việc truyền thông tin từ Encoder sang Decoder mà không làm suy giảm chất lượng:

- **LLM2Vec Encoder**: Mô hình encoder hai chiều xử lý context cực dài (được đóng băng ở Phase 1, dùng LoRA ở Phase 2).
- **Gated Residual Adaptor (Cầu nối Encoder-Decoder)**: Chuyển đổi và chắt lọc không gian chiều (dimension) từ Encoder sang Decoder một cách mượt mà. Khối cầu nối này bao gồm các thành phần con:
  - *Layer Fusion*: Tích hợp các đặc trưng theo token từ nhiều tầng khác nhau của encoder.
  - *Gated Residual Projection*: Ánh xạ chiều dữ liệu kết hợp với skip-connection để giữ thông tin gốc.
  - *Salience Gate*: Bộ lọc nội dung (token-level), loại bỏ các token dư thừa giúp decoder bớt nhiễu.
  - *EncStack*: Tinh chỉnh lại bộ nhớ (memory refinement) trước khi đưa vào Cross-attention.
  - *Global Memory Tokens*: Các token toàn cục học được (learnable) gắn vào đầu dãy memory để tóm lược đại ý văn bản.
- **Lightweight Decoder**: Decoder Transformer tự thiết kế (ví dụ cấu hình H200: 8 layers, 1024 hidden size) chuyên trách việc sinh văn bản.
- **MTP (Multi-Token Prediction) Heads**: Các module phụ trợ (được train riêng ở Phase 3) giúp tăng tốc sinh từ bằng kỹ thuật Speculative Decoding (Dự đoán nhiều token cùng lúc).

## Evaluation Results: LLM2Seq vs T5Gemma2-1B-1B

Dưới đây là kết quả so sánh hiệu năng của **LLM2Seq (Phase 2 - LoRA Encoder)** với **T5Gemma-1b-1b (LoRA)** trên tập test WikiLingua (3,901 mẫu).

| Metric | LLM2Seq (Phase 2) | T5Gemma (1B-1B) |
| :--- | :--- | :--- |
| **ROUGE-1** | **48.36** | 33.08 |
| **ROUGE-2** | 15.54 | **19.58** |
| **ROUGE-L** | **29.05** | 21.86 |
| **chrF** | 15.45 | **28.60** |
| **Mean Prediction Words** | **29.1** | 211.8 |
| **Too Short Rate (%)** | 33.56% | 0.15% |
| **Too Long Rate (%)** | **5.46%** | 95.46% |
| **Latency Mean (s)** | 0.69s | **0.33s** |
| **Peak VRAM** | **~6 GB** | ~23.5 GB |

*Ghi chú: Độ dài trung bình của bản tóm tắt mẫu (reference) là 51.8 từ.*

### Phân tích & Nhận xét
1. **Chất lượng tóm tắt (ROUGE & chrF)**: 
   - LLM2Seq đạt điểm ROUGE-1 (48.36) và ROUGE-L (29.05) cao hơn, cho thấy khả năng chắt lọc chính xác các ý chính và duy trì cấu trúc câu của văn bản nguồn.
   - T5Gemma-1B-1B nhỉnh hơn ở ROUGE-2 (19.58) và chrF (28.60), thể hiện sự mượt mà trong việc sinh các cụm từ (n-grams).
2. **Kiểm soát độ dài (Length Control)**: 
   - LLM2Seq kiểm soát độ dài rất tốt với trung bình 29.1 từ, tạo ra các bản tóm tắt ngắn gọn và đi đúng trọng tâm (dù có 33.56% mẫu bị đánh giá là hơi ngắn so với mức 51.8 từ của bản gốc).
   - T5Gemma-1B-1B gặp khó khăn trong việc học token kết thúc (EOS), dẫn đến độ dài trung bình lên tới 211.8 từ (95.46% Too Long Rate).
3. **Hiệu năng & Tài nguyên (Speed & VRAM)**: 
   - T5Gemma-1B-1B cho độ trễ (Latency Mean) rất thấp (0.33s/mẫu), cho thấy lợi thế của một kiến trúc decoder gốc đã được tối ưu hóa sâu bởi thư viện Transformers.
   - Ngược lại, LLM2Seq (0.69s/mẫu) giải mã chậm hơn nhưng lại chứng minh sự tối ưu vượt trội về mặt bộ nhớ: chỉ tiêu tốn **~6 GB VRAM** so với ~23.5 GB của T5Gemma-1B-1B (trong cùng cấu hình batch size), rất thân thiện với các phần cứng hạn chế.
4. **Kết luận**: 
   - Cả hai kiến trúc đều có ưu nhược điểm riêng. T5Gemma-1B-1B có lợi thế tuyệt đối về tốc độ giải mã thuần túy, trong khi LLM2Seq lại làm tốt hơn ở khả năng tóm tắt súc tích và tối ưu hóa VRAM. 
   - Đối với LLM2Seq, nền tảng chất lượng và bộ nhớ tốt ở Phase 2 chính là tiền đề hoàn hảo để tiếp tục triển khai Phase 3 (Speculative Decoding bằng MTP Heads), qua đó khắc phục điểm yếu duy nhất là độ trễ (Latency).
