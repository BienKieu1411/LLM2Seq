# Review custom length reward

Ngày: 2026-10-17. Ba agent được giao follow-up trên cùng câu hỏi, chỉ đọc code và nghiên cứu nguồn.

## Kết quả từ literature agent

CMDP hỗ trợ tách content/cost; ROUGESal hỗ trợ salient reference matches và RL/CE; coverage reward cung cấp tiền lệ source-derived reward nhưng không bảo đảm ROUGE. CPMI là đối chiếu source-conditional/target-only **ở decoding**, không phải SEAM GRPO reward. Source-swap margin là adaptation chưa được chứng minh.

## Kết quả từ code agent

Scorer SEAM đã trả active-token likelihood của full mixture, phù hợp làm nền cho teacher diagnostic. Reward callback hiện chỉ nhận candidate/reference, chưa nối nguồn/teacher. Beta=0 không load frozen reference; không có teacher miễn phí. Teacher correct/wrong source đều cần decoder scoring, nhưng encoder/bridge có thể chia sẻ giữa candidate trong một group.

Không có attention weights trong SDPA output. Copy distribution khác semantic support. Source mask rỗng kích hoạt generator-only fallback, nên không dùng all-masked source như một phép can thiệp vô hại.

## Kết quả từ adversarial agent

Reward ROUGE+length đã gửi policy gradient về encoder qua graph hiện tại; không cần giả định phải thêm reward riêng mới học được encoder. Source semantic coverage chưa có evidence-unit definition ổn định. Source-swap score có thể dùng làm proxy source discrimination, không factuality, và cần diagnostic trước. Cùng length cost trong group vẫn bị triệt tiêu bởi centering.

## Quyết định coordinator

- Sửa length cost theo MAE/B để gần chỉ tiêu tuyệt đối người dùng đã khai báo.
- Giữ source-swap teacher margin là nhánh nghiên cứu sau diagnostic; không tự bật trong training chính.
- Phân biệt reward specification, math-only verification và model validation. Không dùng các kiểm tra số học để tuyên bố tăng ROUGE hay giảm hallucination.
- Không tự thêm teacher/model download, PRM, semantic head, token-level objective hoặc runtime code trong nghiên cứu này.
