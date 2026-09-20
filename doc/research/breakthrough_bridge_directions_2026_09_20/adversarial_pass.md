# Adversarial pass

## 1. Điều gì có thể làm kết luận hiện tại sai?

Các run mới chỉ có aggregate ROUGE và không có paired outputs, repeated seeds
hoặc full resolved configs. Shared compression/write-back pattern là lời giải
thích hợp nhất cho việc chọn hướng nghiên cứu, nhưng chưa phải nguyên nhân nhân
quả đã chứng minh. Protocol drift hoặc optimization noise vẫn có thể giải thích
một phần chênh lệch.

## 2. Hướng nào nhìn mới nhưng thực chất lặp lại lỗi cũ?

- Layerwise K/V sẽ lặp lỗi nếu K và V chọn hai encoder depths độc lập hoặc nếu
  memory mới thay thế final-tap anchor ngay từ đầu.
- Query-hierarchical sẽ lặp `query_regions` nếu block score bị biến thành một
  static `[B,S]` source bias trước decoder thay vì được tính từ từng decoder
  query.
- Side-memory sẽ lặp evidence slots nếu compact entries được concatenate vào
  cùng token softmax hoặc được write-back vào token keys.
- Hyper-bridge sẽ lặp key residual nếu nó chỉ tạo một document-specific K delta
  không có identity path và RMS bound.

## 3. Phản biện mạnh nhất cho từng hướng

1. **Layerwise K/V:** PPLX/Qwen không phải encoder-decoder được pretrain đồng bộ;
   lower taps có thể chỉ làm lệch representation distribution. Dùng cùng
   token-aligned `M_l` cho K/V, final-tap anchor và zero-init output.
2. **Side-memory:** branch có thể dormant; zero gate ban đầu cũng chặn gradient
   tới resampler cho đến khi gate rời zero. Cần log gate, output norm và branch-off
   logit delta; cache side K/V riêng ở inference.
3. **Exact routing:** soft block routing tăng complexity; hard routing có thể bỏ
   entity, number hoặc evidence qua boundary. Lần đầu dùng non-overlap blocks,
   soft routing và dense fallback.
4. **Hyper-bridge:** không thêm information mới, khó tách khỏi decoder adaptation
   và có thể tăng gradient variance theo document. Chỉ thử sau một positive
   bridge control.

## 4. Evidence nào sẽ khiến ta đổi quyết định?

- Nếu intermediate PPLX taps không có lexical/entity probe tốt hơn final tap,
  hạ layerwise K/V xuống sau side-memory.
- Nếu side-memory gate tăng nhưng branch-off không đổi logits, branch bị decoder
  hấp thụ hoặc redundant; dừng hướng này.
- Nếu exact routing đạt high reference-evidence recall trên validation và API
  query/cache được kiểm thử, có thể nâng nó lên trước side-memory.
- Nếu direct projection variance qua ba seeds lớn hơn chênh lệch candidate, không
  được claim bridge improvement từ một seed.

## Quyết định sau phản biện

Chạy layerwise coupled-depth K/V trước, gated side-memory thứ hai,
query-hierarchical exact-token thứ ba và document-conditioned operator cuối.
Hai arm đầu khác nhau đủ rõ, có exact base fallback và ít nguy cơ tái tạo các
region/slot failures nhất.
