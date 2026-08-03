# Checklist kiểm thử KD

Phạm vi tham chiếu: `src/llm2seq/src/training/kd_loss.py`,
`src/llm2seq/src/data/dataset.py` và `src/llm2seq/src/data/collator.py`.

- [ ] **Shape:** `sequence_kd` nhận logits `[B,T,V]` và target IDs `[B,T]`; full-KL nhận teacher logits `[B,T,V]`; top-k nhận teacher values/indices `[B,T,K]`. Kiểm tra loss là scalar hữu hạn, `K <= V`, chỉ số top-k nằm trong `[0,V)`, và mismatch bị báo lỗi rõ ràng.
- [ ] **Mask:** padding của `labels` phải là `-100`; thay đổi logits/teacher ở vị trí `labels == -100` không được đổi loss. Kiểm tra riêng batch có toàn vị trí bị mask (đặc biệt `sequence_kd`, hiện gọi thẳng `F.cross_entropy`). `decoder_attention_mask` không thay thế cho loss mask.
- [ ] **Gradient:** gọi `.backward()` cho cả ba nhánh: student có gradient khác 0 ở vị trí hợp lệ; vị trí bị mask không tạo gradient; teacher không có gradient (`compute_kd_loss` dùng `.detach()` cho KL/top-k, còn sequence KD chỉ dùng IDs). Khi tắt KD hoặc thiếu teacher input, tổng loss chỉ có CE.
- [ ] **Teacher cache:** mỗi record có `sample_id` duy nhất, target không rỗng và được collate đúng độ dài. Cache text-only được retokenize bằng tokenizer student; đường chạy mặc định không có bước kiểm tra tương thích metadata.
- [ ] **Sequence KD:** kiểm tra riêng gold CE và pseudo-target CE; pseudo-target không được mang evidence labels sang nhánh gold. Top-k/full-logit KD không thuộc đường chạy đơn giản này.

Fixture tối thiểu: một hàng có `-100` và một cache record hợp lệ; chạy loss, backward, invalid shape/index, all-masked và round-trip cache.
