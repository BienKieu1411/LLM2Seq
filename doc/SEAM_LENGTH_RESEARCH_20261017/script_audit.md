# Audit các script tham khảo

Ngày: 2026-10-17. Các file gốc nằm trong `/Users/kieugiangbien/Downloads/`; không sửa hoặc chạy entrypoint training của chúng. SHA256 nằm trong [kết quả kiểm chứng](checks/result.json). Số dòng dưới đây ứng với snapshot đó.

## `lafrt.py`

- `length_reward`, khoảng dòng 117: reward `max(0,1-|actual-target|/target)`; ROUGE được đo trong evaluation, không phải content reward của hàm này. Không coi đây là recipe đã đồng thời tối ưu ROUGE.
- Header và loader: `input` đã là prompt được format; `content` là reference; `output` được ghi là synthetic và bị bỏ qua. Không dùng adapter này nguyên trạng cho JSONL `text/summary` hoặc `input/abstract_sum` của dự án.
- Loader truyền `target_length`/`ground_truth` vào reward, nhưng không tự chèn giá trị đó thành chỉ dẫn trong input. Có nhãn số cho reward chưa có nghĩa model đã nhận số đó trong prompt.
- Mặc định `WORD_COUNT_MODE="larft"`, regex ASCII/CJK, không phù hợp trực tiếp với bộ đếm Unicode tiếng Việt. Chế độ `unicode` của file này còn thiếu bước NFKC trong `count_words`; cần đồng nhất cả normalization.
- Auxiliary khoảng dòng 523-568 tạo task đếm văn bản do policy vừa sinh. Đây là một task/forward bổ sung, không phải CE trên summary thật. Chưa đưa vào hướng SEAM vì tăng chi phí và khác mục tiêu CE nhỏ người dùng yêu cầu.

## `lead_eigen.py`

- ROUGE weight 0.75 và Eigen weight 0.25. Eigen alignment so reference/candidate không tự chứng minh candidate được source hỗ trợ.
- Length lấy từ **decoder subword tokens** của reference và candidate, khoảng dòng 639-690. Khác metric word-count của benchmark tiếng Việt.
- Sau khi tính Q, `semantic_reward=2Q-1`; khoảng dòng 804 chỉ nhân Gaussian length khi semantic reward >0. Do đó length không tham gia phân biệt reward khi Q<=0.5.
- Việc bảo vệ reward âm khỏi phép nhân có lý do về dấu, nhưng tạo vùng không học length. Không khắc phục bằng cách nhân cả reward âm với Gaussian: candidate sai length sẽ bị đẩy về 0 và có thể được ưu tiên.

## `lead_eigen_threshold.py`

- Gần như cùng file với `lead_eigen.py`, nhưng có `CONTENT_THRESHOLD=0.25` và `semantic_reward=Q-0.25`, khoảng dòng 797.
- Ngưỡng length-inactive chuyển từ 0.5 xuống 0.25; không loại bỏ cơ chế ngắt này. Không phải một thuật toán length độc lập chỉ vì đổi threshold.
- Vẫn dùng token length. Cần thay cả đơn vị, conditioning và grouping nếu áp dụng cho SEAM.

## `lead_len_control.py`

- **Snapshot thực tế có ROUGE_WEIGHT=1.0, EIGEN_WEIGHT=0.0** ở dòng 46-47; Eigen call bị comment. Không kết luận file này đang dùng 0.75/0.25.
- `word_count`, dòng 286: whitespace split, khác NFKC Unicode evaluator ở dấu câu/normalization.
- `compute_reward`, dòng 342-362: Q chính là ROUGE trung bình; semantic reward `2Q-1`, length chỉ có hiệu lực khi **Q>0.5**.
- Kiểm chứng với hàm reward trích bằng AST, controlled ROUGE Q=0.4: candidate 50, 100, 150 từ khi target=100 đều nhận reward -0.2 dù Gaussian khác nhau. Với Q=0.6, candidate đúng 100 từ được 0.2, candidate 50/150 từ khoảng 0.027067. Đây là phép kiểm tra đại số, không phải model evaluation.
- Hàm ROUGE dùng `rouge_score.RougeScorer(...,use_stemmer=True)` và tokenizer mặc định. Cần kiểm tra Unicode trước khi dùng cho tiếng Việt; không tương đương protocol Perl của dự án.
- Có target cho reward chưa bảo đảm instruction chứa target. Loader/template cần được kiểm riêng.

## `semantic_lafrt_len_balanced.py`

Điểm nên học: khoảng dòng 639, chi phí over/under luôn được trừ khỏi reward, kể cả khi content reward âm. Có báo cáo length theo bin và thống kê signed error; hữu ích để tìm nhóm dài/ngắn học kém.

Những chỗ cần sửa nếu tái sử dụng:

- Mặc định `larft` vẫn sai hệ đếm so với evaluator. `unicode` trong file này có NFKC và khớp evaluator trên các ví dụ đã kiểm chứng, không có nghĩa mọi preprocessing khác đều tương đương.
- BARTscore được tính từ reference sang candidate, không phải source sang candidate; không dùng nó làm bằng chứng factuality. `exp(min(0,BARTscore))` và ROUGE có thang/phân phối khác nhau; weight 0.5/0.5 không bảo đảm đóng góp ngang nhau.
- Có tám bin độ dài, hai dual mỗi bin và thêm adaptive hardness. Nhiều state và hyperparameter cần kiểm chứng hơn baseline.
- `_update_duals` dòng 484 và `_update_bin_hardness` dòng 427 lấy trung bình rows cục bộ. Callback khoảng dòng 705 gọi update sau mỗi lần reward; không thấy global reduction ở các update này. Chưa đủ để bảo đảm dual giống nhau giữa hai GPU.
- `compute_length_bin_update_prior`, dòng 862: trọng số inverse frequency hoặc sqrt inverse frequency được nhân vào policy reward. **Đây là reweighting, không phải sampler tạo dataset cân bằng**. File đặt `scale_rewards=False`, nên không kết luận weight chắc chắn bị group-standardization triệt tiêu trong chính file này.
- Có hindsight counting trên các prefix của văn bản sinh; không cần thêm nếu mục tiêu chỉ là content+length GRPO và gold CE.
- `joint_quality` là trung bình semantic quality và `exp(-relative_error)`. Tăng scalar này không chứng minh vượt từng chỉ tiêu R2 và MAE.

## `schedule_running.py`

Chỉ poll VRAM GPU rồi gọi một shell script có đường dẫn hard-code. Không phải schedule giữa SFT/GRPO, không điều khiển reward hay learning rate. Không có cơ chế giữ chỗ GPU giữa lần kiểm tra và lúc chạy. Không dùng làm training scheduler cho dự án.

## Kết luận áp dụng

Giữ ba ý: length request xuất hiện trong input, chi phí sai length luôn tham gia reward, và chẩn đoán theo nhóm độ dài. Không bê nguyên stack decoder-only, regex đếm, task hindsight hoặc cơ chế cập nhật dual cục bộ vào SEAM. [Thiết kế](2026-10-17_design.md) ghi rõ phần còn thiếu ở prompt, bridge, full-actor sync và vLLM cache.

Lệnh chạy lại audit model-free trên Mac:

```bash
/Users/kieugiangbien/bienkieu_env/bin/python \
  doc/SEAM_LENGTH_RESEARCH_20261017/checks/audit_supplied_scripts.py
```

Đây là audit có phạm vi hẹp: không xác minh chất lượng model, thời gian training hay tác dụng thực nghiệm của Eigen/BART/dual adaptation.
