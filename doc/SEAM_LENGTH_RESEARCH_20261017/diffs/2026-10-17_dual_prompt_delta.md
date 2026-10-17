# Delta so với báo cáo đầu

- User yêu cầu khai thác cả encoder và decoder prompts.
- Chuyển phương án chính từ decoder-length-prompt-first sang dual-role prompts, cùng W.
- Giữ bridge hiện tại vì đã nhận prompt decoder và pooled source; numeric word branch vẫn là đối chứng tiếp theo.
- Bổ sung CTRLsum, GSum, prior art hai phía, entity planning, LAAM và repository tác giả vào nguồn.
- Làm rõ bridge dùng pooled token embeddings, không contextual decoder states.
- Bổ sung causal-prefix limitation, prefix budget/copy offset, no-grad rollout reuse và actor-version cache.
- Bổ sung group-cost cancellation trong GRPO và giới hạn tổng quát hóa sang các W khác gold.
- Ba agent review đã hoàn thành; chỉ cập nhật research artifacts, chưa chỉnh runtime.
