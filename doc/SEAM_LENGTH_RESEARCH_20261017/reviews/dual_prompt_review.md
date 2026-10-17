# Phản biện cho cập nhật hai prompt

Ngày: 2026-10-17. Ba `luna_worker` làm việc độc lập, chỉ đọc code/tài liệu. Coordinator đối chiếu lại nguồn chính và tổng hợp dưới đây. Không phải kết quả train hoặc ba lần replication.

## Agent literature: `dual_prompt_papers`

Đề xuất dùng GSum và CTRLsum làm tiền lệ guidance; LAAM sát với budget-aware source access hơn. Đã bổ sung kiểm tra primary paper của LAAM, ghi rõ nó sửa cross-attention và có pretraining riêng. Entity-chain planning khác với latent conditioning của SEAM, không được coi instruction là một planner đã học.

Điều tiếp thu: encoder-side conditioning là giả thuyết có cơ sở; numeric bridge không tự trở thành cần thiết. Không chuyển nguyên stack sinh plan hay synthetic pretraining vào recipe này.

## Agent code: `dual_prompt_code`

Decoder prompt hiện đi vào cả decoder và controller qua pooled token embeddings. Encoder prefix có thể thay đổi token states của nguồn, nhưng tiêu tốn `max_source_length`; copy offsets hiện dùng prefix cố định. Native scoring/rollout yêu cầu decoder prompt cùng width, trong khi caller vLLM hiện chưa truyền từng cặp prompt theo sample.

Full-actor gradients có đường native; vLLM chỉ rollout và cần nhận mọi trọng số actor đã đổi. Group G cùng source/W có thể chia sẻ no-grad encoder/bridge state; hiện caller repeat source G lần. Cache across W không hợp lệ nếu W nằm trong encoder input.

Điều tiếp thu: cần source-span control, prefix offset theo record, cùng renderer mọi đường, group-local cache và không detach encoder trong actor loss. Chỉ `architectures` trong model config không đủ để chứng minh attention mask thực tế; kiểm tra behavior của từng adapter.

## Agent phản biện: `dual_prompt_adversarial`

Không có bằng chứng SEAM hai prompt tốt hơn một prompt. Có prior art cho length conditioning cả hai phía; không claim novelty chỉ từ “hai prompt”. Dùng encoder prompt và decoder prompt khác vai trò, giữ pretrained task marker nhất quán giữa các đối chứng.

Group centering triệt tiêu cost giống nhau giữa candidate; tăng penalty không tự tạo tín hiệu. CE phải khớp W; kiểm tra request ngoài phân phối gold length. Cần tách token cap khỏi word request.

Điều tiếp thu: dual-role prompts là phương án chính; decoder-only prompting trong **SEAM** là đối chứng quan trọng; nhánh numeric bridge là follow-up. Không thêm entropy constraint, reward model hoặc generated counting task trong lần thử đầu.

## Quyết định coordinator

Giữ ba biến thể D/E/ED để tách hai phía; SFT cũ là tham chiếu. ED tận dụng source representation và bridge có sẵn. Đo ROUGE, requested-length error, cap rate, cost dispersion và compute. Chưa viết training mode hay tuyên bố tăng score trong cập nhật nghiên cứu này.
