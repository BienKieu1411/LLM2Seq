# F05 — Kiểm định lập luận cho bridge theo decoder query

Ngày 2026-09-18. Nguồn đã đọc: code `src/eviseq_new` và ứng viên
`src/afmr_query_regions`; điểm ROUGE người dùng cung cấp; các trích xuất
SEASON S02, TopDownFormer S01, DYLE S10, Fast Evidence S14 và Latent Queries
S15. Các file nguồn giấy tờ lưu riêng trong `../sources/`. Không có checkpoint
PubMed full/direct hoặc paired predictions để đo hiệu quả thật. Đây là
**deep-read của cơ chế và lập luận**, không phải kết quả thực nghiệm PubMed.

## Mệnh đề trung tâm

Một source prior cố định theo toàn bộ quá trình sinh có thể chưa đủ cho
decoder quyết định lúc nào đọc mục tiêu, phương pháp, kết quả hay kết luận.
Đọc các vùng nguồn bằng query ở từng bước, rồi dùng phần đọc ấy sửa nhẹ
query của token cross-attention, là một phép thử hợp lý để bridge thực sự
tham gia vào lựa chọn nội dung. Nó chỉ là giả thuyết cho đến khi `C−B` và
`C−A` được đo trên cùng protocol.

## Cây lập luận

1. Code audit cho thấy AFMR tạo `source_bias[B,S]` một lần tại encode, rồi
   phát cùng bias đến mọi decoder layer và mọi bước sinh. QK token attention
   vẫn động; chỉ **prior riêng của bridge** là cố định. Quan sát `B−A` gần
   bằng không ở ba điểm cũ phù hợp với nhiều cách giải thích khác nhau, không
   chỉ sự cố định này.
2. Các paper cho thấy ba thành phần có tiền lệ: hướng coarse-to-token
   (TopDownFormer), tách tín hiệu chọn nguồn khỏi value gốc (SEASON), và
   weighting phụ thuộc decoder history (DYLE). Chúng hỗ trợ khả năng thiết kế,
   không suy ra mức tăng ROUGE của AFMR.
3. Ứng viên C giữ `M`, `H0`, grounded copy và full-token QK; thêm ngân hàng
   các vùng chồng lấn. Mỗi layer đọc vùng theo hidden của decoder và sửa query
   với residual giới hạn RMS. Do vậy tín hiệu region có đường nhân quả tới
   logits và CE, nhưng hiệu ứng văn bản sau train chưa biết.
4. Đối chứng phản biện mạnh nhất: pretrained cross-attention vốn đã tự học
   query-dependent selection; thêm region read có thể dư thừa hoặc làm lệch
   lựa chọn từ. DYLE cũng không cho thấy kiểu weighting động thắng mọi hệ
   thống. Cần kiểm tra `C−B`, không dựa vào cơ sở paper để hứa +0.3.

## Evidence ledger

| Claim | Evidence, location | Relationship | Confidence | Caveat |
| --- | --- | --- | --- | --- |
| Prior AFMR cố định theo bước giải mã | `src/eviseq_new/eviseq_afmr/modeling/afmr.py` tạo `source_bias`; `decoder.py` broadcast `[B,S]` | Giải thích giới hạn biểu đạt của riêng prior | **Source fact or data** | Token QK và copy vẫn phụ thuộc query |
| Full cũ không thể chứng minh hơn w/o bridge | Điểm người dùng: full `49.686/22.153/45.939`, direct `49.671/22.135/45.956`; `../sources/s09_user_scores.md` | Chênh lệch gần 0, R-L đổi dấu | **Source fact or data** | Protocol cũ có thể bị confound bởi salience loss; không có paired outputs |
| Coarse-to-token có thể hữu ích cho PubMed | TopDownFormer, `../sources/s01_topdownformer.md`, phần ablation | Cho tiền lệ cơ chế | **Author's stated position** | Model và vị trí can thiệp khác; không được chuyển mức tăng sang AFMR |
| Tách selection khỏi source value là hợp lý | SEASON, `../sources/s02_season.md`, phương pháp | Hỗ trợ giữ H0 value | **Author's stated position** | Paper dùng hướng dẫn salience và news data |
| Decoder-conditioned source weights đã có tiền lệ | DYLE, `../sources/s10_dyle.md`, phần method/results | Hỗ trợ tính khả thi, giới hạn novelty | **Author's stated position** | Không có bằng chứng DYLE làm AFMR tăng ROUGE |
| Query-region residual mới có thể tăng cả ba ROUGE | Chưa có full server run | Là dự đoán cần kiểm tra `C−B` và `C−A` | **Unverified** | Cross-attention gốc có thể đã đủ; thêm params/VRAM |
| C có đường gradient CE và sinh với cache | Tiny tests trong `src/afmr_query_regions/tests/` | Kiểm tra implementation nhỏ | **Source fact or data** | Không thay cho DDP, B200 hay long-document test |

## Khái niệm và giới hạn

`H0` là biểu diễn encoder được project thẳng sang decoder width; `M` là
semantic memory sau depth/feature residual; region bank là các cửa sổ content
token chồng lấn, không phải snippet được oracle chọn. `A=direct`, `B=static`,
`C=static+query-region`. `C−B` là hiệu ứng của nhánh mới, còn `C−A` gồm cả
bridge cũ. Nếu không có matched seed/config và cặp prediction theo ID, không
được viết `C−A` như hiệu ứng riêng của region read. Không thể gọi đây là
phương pháp đầu tiên dùng query-aware source selection; paper trước đã làm.

## Câu hỏi kiểm tra sau train

1. Khi khóa checkpoint C và tắt riêng region read ở validation, CE và các
   prediction đổi bao nhiêu? Gate, RMS và attention entropy có biến thiên
   theo bước sinh hay gần hằng số?
2. `C−B` có dương trên từng ROUGE sau khi cùng seed, data, batch, update count,
   checkpoint và preprocessing? Khác biệt có ổn định qua seed/paired bootstrap?
3. Nếu ROUGE tăng nhưng groundedness giảm, liệu region read đang ưu tiên từ
   phổ biến trong gold hơn là chứng cứ nguồn?
