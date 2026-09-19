# PubMed region bridge: kết quả âm và quyết định tiếp theo

Ngày 2026-09-19. Các ROUGE dưới đây do người dùng báo; máy hiện tại không có checkpoint, log, resolved config hoặc predictions của hai run mới. Vì vậy đây là đánh giá thiết kế và code, **không phải chẩn đoán nguyên nhân đã chứng minh**.

| Hệ thống | R-1 | R-2 | R-L | So với full AFMR |
|---|---:|---:|---:|---:|
| `eviseq_new` full | 49.686 | 22.153 | 45.939 | — |
| `direct_projection` + grounded copy | 49.671 | 22.135 | 45.956 | −0.015/−0.018/+0.017 |
| `afmr_query_regions` | 49.433 | 21.991 | 45.757 | −0.253/−0.162/−0.182 |
| `afmr_query_qspace` | 49.568 | 22.087 | 45.858 | −0.118/−0.066/−0.081 |

Q-space hơn hidden-space 0.135/0.096/0.101, nhưng vẫn thấp hơn **cả full và direct** ở cả ba metric. Đừng diễn giải nó là R-2 gain bằng cách so với mốc full cũ 49.626/21.901/45.895. Chênh lệch này cũng chưa đủ để xác định ý nghĩa thống kê nếu thiếu paired predictions và seed lặp.

## Điều code audit xác nhận

- Cả hai nhánh lấy key từ các vùng trung bình của bridge memory `M`, value từ các vùng trung bình của `H0`, cửa sổ 128 token/stride 64. Mỗi decoder layer đọc cùng một bank và full-token cross-attention vẫn tồn tại.
- Hidden-space variant cộng vector vùng trước `q_proj`; Q-space variant cộng residual theo từng Q head trước `q_norm`. Cả hai chỉ thay đổi truy vấn tới token cross-attention; chúng không đưa một cơ chế chọn vùng tường minh vào token logits.
- Nhánh region khởi tạo `out_proj=0`, nhận gradient CE ở output trước rồi mới truyền lên q/k/v sau cập nhật. Optimizer warmup có chứa tham số region. Tiny tests đã kiểm tra parity lúc khởi tạo và cache generation. Không thấy lỗi gradient hoặc train/eval mismatch chắc chắn từ code.
- Cả hai dùng mặc định **một** region-attention head; Q-space sau đó chiếu cùng vector vùng ra mọi token-attention head. Đây là giới hạn biểu diễn, chưa chứng minh là nguyên nhân giảm ROUGE.

## Giả thuyết ưu tiên và lý do

`mean(H0)` trong 128 token có thể làm mờ entity, số, phủ định và quan hệ; nhánh mới sau đó sửa query ở *mọi* decoder layer trong khi cross-attention vốn đã query-conditioned và đọc đủ token. Đây là giả thuyết có thể kiểm định, không phải kết luận từ điểm aggregate. [TopDownFormer](sources/s01_topdown.md) cho thấy lựa chọn pooling ảnh hưởng mạnh trong một hệ khác, còn [Ling–Rush](sources/s02_coarse_fine.md) cảnh báo hard chunk selection có thể làm chất lượng kém hơn standard attention. [SEASON](sources/s03_season.md) là tiền lệ tách tín hiệu chọn nguồn ở key khỏi source values. [HEPOS và LQSUM](sources/s04_hepos_lqsum.md) hỗ trợ giữ tính đa dạng theo head và giữ original source path; cả hai không dự báo kết quả AFMR.

Hai thử nghiệm kế tiếp đều quay về đúng graph `eviseq_new` và chỉ thay bridge; decoder, grounded copy và CE giữ nguyên:

- `src/afmr_adaptive_topdown`: learned token-weighted region pooling, global region mixing, rồi top-down correction trở lại token memory keys. Nó kiểm tra trực tiếp liệu mean pooling và thiếu global-to-token refinement có phải nút thắt hay không.
- `src/afmr_evidence_slots`: một bank latent evidence slots đọc toàn bộ content tokens, tự trộn thông tin rồi được token query ngược để nhận global evidence context. Nó kiểm tra một bottleneck không phụ thuộc biên cửa sổ cố định.

Cả hai giữ base-projected `H0` làm decoder values và grounded-copy anchor; không hard top-k, không auxiliary loss và không sửa decoder. Đây là hai *candidate* CE-only, chưa được coi là đóng góp paper trước khi thắng đối chứng.

## Phép thử quyết định, trước khi tốn thêm một run lớn

1. Trên **cùng checkpoint Q-space đã train**, chạy validation với region bật và tắt (`out_proj` trả 0), cùng ID và decoding. Nếu tắt tốt hơn, nhánh đang gây hại khi inference; nếu gần như không đổi, nhánh có thể dormant hoặc thay đổi optimization trong train mới là nguyên nhân. Nên ghi thêm CE, RMS của Q residual theo layer, gate và attention entropy. Tắt nhánh sau train không tương đương ablation được train lại; nó chỉ phân biệt cơ chế.
2. Kiểm tra `resolved_config.yaml`, data fingerprint, PPLX path, prompt, seed, batch × accumulation, số optimizer steps, `last.pt`, generation flags và ROUGE-1.5.5 của cả bốn run. Script mặc định đánh `last.pt`; các paper có thể dùng checkpoint selection khác, không được so trực tiếp như cùng protocol.
3. Chọn hyperparameter và checkpoint bằng validation. Sau khi khóa cấu hình, so `direct+copy`, static AFMR và candidate mới với matched seeds; paired bootstrap trên cùng IDs. Với mốc mong muốn ~+0.3 R-1/2/L, chỉ viết claim bridge rõ nếu cả ba vượt ở protocol khóa trước. Không điều chỉnh từ test score lặp đi lặp lại rồi gọi test là phép đo độc lập.

## Giới hạn và claim

Năm nguồn sơ cấp đưa ra cơ sở thiết kế nhưng **không cung cấp ba nguồn độc lập chứng minh candidate sẽ tăng ROUGE trong AFMR**; hiện trạng là insufficient evidence cho claim tăng điểm. Soft coarse-to-token attention cũng có tiền lệ: novelty cần nằm ở interface cụ thể với AFMR value anchor/grounded copy và kết quả ablation rõ ràng, không thể tuyên bố chỉ vì ghép module. Nếu candidate không thắng direct+copy, bridge không nên được diễn giải là đã có đóng góp thực nghiệm.
