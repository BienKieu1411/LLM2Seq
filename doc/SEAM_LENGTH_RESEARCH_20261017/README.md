# SEAM: điều khiển độ dài và tăng ROUGE

**Kết luận sau cập nhật:** training và inference đưa cùng số từ mong muốn vào **encoder và decoder, với hai vai trò khác nhau**. Encoder chuẩn bị biểu diễn nguồn theo yêu cầu; bridge điều chỉnh memory/source bias; decoder viết summary. Huấn luyện bằng GRPO nội dung/độ dài cộng CE nhỏ trên summary thật. Chỉ thêm nhánh số học riêng nếu đối chứng cho thấy cần thiết.

Đây là kết quả nghiên cứu và kế hoạch tích hợp. **Chưa có training mode length mới trong trainer hiện tại**, chưa chạy GPU và chưa có bằng chứng tăng score của SEAM. Các file hiện có không bị thay đổi bởi nghiên cứu này.

- **[Thiết kế mới nhất: dual-role prompts](2026-10-17_dual_prompt_update.md)**
- **[Reward mới: MAE theo chỉ tiêu và source-discrimination có kiểm chứng](2026-10-17_reward_update.md)** - nhánh teacher reward chưa được triển khai/bật mặc định.
- [Báo cáo đầu và công thức reward](2026-10-17_design.md) - khuyến nghị encoder prefix chung đã được thay thế bởi cập nhật trên.
- [Phản biện từ ba agent](reviews/dual_prompt_review.md)
- [Phản biện riêng cho custom reward](reviews/reward_review.md)
- [Phản biện sáu script được cung cấp](script_audit.md)
- [Kế hoạch, giả thuyết và giới hạn](plan.md)
- [Nguồn đã kiểm tra](sources.csv), [ghi chú từng nguồn](sources/)
- [Kiểm chứng số học và fingerprint code](checks/audit_supplied_scripts.py), kết quả ở `checks/result.json`.
- [Các điểm cần cập nhật khi triển khai](refresh_targets.md)

Không gọi đây là một phương pháp mới đã được chứng minh. Prompt độ dài, RL độ dài và tối ưu có ràng buộc đều có tiền lệ; giá trị cần kiểm chứng ở đây là tích hợp đúng vào luồng encoder - bridge - decoder của SEAM.
