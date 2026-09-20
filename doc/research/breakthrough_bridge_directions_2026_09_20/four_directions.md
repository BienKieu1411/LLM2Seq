# Bốn hướng bridge mới sau các thí nghiệm thất bại

Ngày: 2026-09-20

## Kết luận thiết kế

Bốn run `query_regions`, `query_qspace`, `adaptive_topdown` và
`evidence_slots` đều thay đổi hoặc bổ sung source memory bằng một biểu diễn đã
nén trước khi decoder thực hiện phép đọc thật sự. Chúng khác nhau ở cách tạo
region/slot, nhưng cùng một họ cơ chế: **source-side compression rồi
write-back/residual vào token path**. Kết quả âm chưa chứng minh duy nhất cơ chế
này là nguyên nhân, nhưng đủ mạnh để không tiếp tục tạo thêm một biến thể region
hoặc slot.

Bốn hướng dưới đây can thiệp vào bốn trục khác nhau:

1. decoder layer nào đọc encoder depth nào;
2. decoder query ở từng bước đọc chính xác span nào;
3. decoder có một global-memory path độc lập hay không;
4. toán tử cross-attention tự thích nghi với từng tài liệu như thế nào.

Tất cả đều giữ `direct_projection + grounded_copy` làm anchor, dùng token-level
CE, không sinh candidate, không contrastive/salience/R-Drop/NEFTune, và có đường
lùi về đúng baseline.

## Hướng 1 — Layerwise coupled-depth K/V

### Cơ chế

Thay vì trộn bốn PPLX taps một lần trong bridge rồi đưa cùng một memory vào mọi
Qwen layer, mỗi decoder layer `l` có một phép trộn depth riêng:

```text
pi_l      = softmax(router(layer_id_l, document_controller))
D_l       = sum_i pi_l[i] * LN_i(H_i)
Delta_l   = Out_l(Down_l(D_l - H_final))
M_l       = P(H_final) + g_l * Delta_l
K_l, V_l  = CrossK_l(M_l), CrossV_l(M_l)
```

`K_l` và `V_l` luôn sinh từ **cùng token-aligned memory `M_l`**. `Out_l` được
zero-init, còn `g_l` nhỏ và bounded, nên logits lúc khởi tạo đúng với
direct-projection dù nhánh depth vẫn nhận gradient ngay từ bước đầu. Grounded
copy tiếp tục đọc `P(H_final)`; nó không đọc `D_l`.

### Vì sao khác các bản đã thử

- AFMR cũ sửa một source memory dùng chung cho mọi decoder layer.
- Hướng này không dùng region, slot, pooling theo cửa sổ hay top-down write-back.
- Decoder layer thấp và cao có thể lấy mức biểu diễn khác nhau nhưng vẫn giữ
  đúng vị trí từng source token và một cặp K/V nhất quán.

### Kỳ vọng và rủi ro

Đây là ứng viên có cơ sở mạnh nhất để tăng R-2/R-L: lower/intermediate taps có
thể giữ tín hiệu lexical/local, còn final tap giữ semantics. Rủi ro lớn nhất là
PPLX và Qwen không phải hai stack được pretrain cùng nhau; routing theo depth có
thể chỉ thêm nhiễu. Vì vậy không map cứng layer-to-layer; dùng learned convex
router, final-tap anchor và bounded residual.

Cơ sở: COMPOSITION, Layer-Wise Coordination, Dense Information Flow và
DecoderLens. Các paper chứng minh cơ chế layer-specific encoder access là hợp
lệ; chúng không chứng minh ROUGE PubMed sẽ tăng.

## Hướng 2 — Query-hierarchical exact-token read

### Cơ chế

Chia source thành block/sentence có chỉ mục cố định. Mỗi decoder query trước hết
chọn block, sau đó chọn **token gốc** trong block:

```text
p_l,t(b)     = softmax_b(q_l,t . k_block_b)
p_l,t(i|b)   = softmax_{i in b}(q_l,t . k_token_i)
p_hier(i)    = p_l,t(block(i)) * p_l,t(i | block(i))
C_hier       = sum_i p_hier(i) * V0_i
C_out        = C_dense + gate_l,t * (C_hier - C_dense)
```

Giai đoạn đầu dùng soft routing trên mọi block, không hard top-k. `C_dense` là
cross-attention direct-projection hiện tại và luôn còn làm fallback. Grounded
copy vẫn dùng exact `H0` token positions.

### Vì sao khác các bản đã thử

- `query_regions` dùng vector region để sửa decoder query rồi vẫn thực hiện một
  full-token read; region mean trực tiếp làm lệch query.
- Hướng mới chỉ dùng block vector làm **routing index**. Readout cuối cùng luôn
  lấy original token values, và block choice thay đổi theo target step.
- Không có pooled region được cộng trở lại source keys.

### Kỳ vọng và rủi ro

Đây là ứng viên trực tiếp nhất cho R-2: một target continuation nhận evidence từ
một span cục bộ nhất quán thay vì các token rời rạc. Dense fallback bảo vệ R-1
và entity/number. Rủi ro là block router bỏ sót evidence hoặc boundary cắt qua
một phrase; dùng sentence blocks khi có boundary, nếu không dùng 64-token blocks
và offset khác nhau theo head-group.

Cơ sở: Selective Attention, Landmark Attention, DYLE và Hierarchical
Transformers for Multi-Document Summarization. DYLE có oracle/consistency losses;
ta chỉ chuyển cơ chế dynamic routing, không tuyên bố tái tạo DYLE.

## Hướng 3 — Gated side-memory cross-attention

### Cơ chế

Tạo 16–32 global evidence tokens `E` bằng một resampler đọc `H_final`, nhưng cho
chúng một cross-attention branch độc lập:

```text
C_token = Attn_token(Q, K0, V0)
C_side  = Attn_side(Q, K(E), V(E))
H       = H + C_token + cap_RMS(tanh(alpha_l) * C_side, 0.10)
```

Hai branch có normalization và softmax riêng. `alpha_l=0` lúc khởi tạo; chỉ gắn
side branch vào mỗi bốn decoder layers ở thử nghiệm đầu. Token K/V và copy state
không thay đổi.

### Vì sao khác `evidence_slots`

`evidence_slots` cũ đọc source rồi write-back vào token keys; sau đó decoder vẫn
chỉ có một attention path. Hướng này không write-back. Compact memory truyền
thẳng một output riêng tới decoder, nên nó không thể làm token bank mất attention
mass và không thể phá lexical matching.

### Kỳ vọng và rủi ro

Hướng này là cách kiểm tra sạch nhất xem global compressed evidence có hữu ích
hay chỉ cách injection cũ gây hại. Nó bảo vệ R-1/R-L tốt, nhưng branch có thể
dormant hoặc resampler vẫn làm mất chi tiết. Phải log gate, side/native output
norm và logit delta khi tắt side branch sau train.

Cơ sở: Flamingo, LongT5, multi-source decoder attention và GTCA. Đây chủ yếu là
mechanism transfer từ multimodal/long-context; bằng chứng task-transfer sang
PubMed còn hạn chế.

## Hướng 4 — Document-conditioned low-rank cross-attention operator

### Cơ chế

Không tạo thêm source memory. Dùng document controller hiện có để điều khiển một
mixture nhỏ của các low-rank basis trên cross-attention của từng decoder layer:

```text
rho_l(c)       = softmax(Hyper(c, layer_id_l))
DeltaW_l^x(c)  = sum_m rho_l(c)[m] * B_l,m^x A_l,m^x
x'_l           = W_l^x x + gamma_l * DeltaW_l^x(c) x
                where x in {query, key, value}
```

Không sinh full weight matrix theo từng sample. Dùng 4 experts, rank 8–16,
zero-init `B`, `gamma` bounded và RMS cap. Chỉ module cross-attention thay đổi;
Qwen self-attention, LM head, original token memory và grounded copy giữ nguyên.

### Vì sao khác AFMR controller cũ

Controller cũ tạo một source-side residual/prior dùng chung. Hyper-bridge mới
điều khiển **cách từng Qwen layer đọc PPLX trong từng document**. Nó có thể điều
chỉnh representation mismatch mà không nén cả tài liệu thành một vector dùng để
thay thế evidence.

### Kỳ vọng và rủi ro

Đây là hướng novelty cao nhất và cũng rủi ro nhất. Nó có thể tạo một
document-specific alignment operator hữu ích, nhưng cũng có thể làm weights dao
động theo batch, tăng gradient variance hoặc chỉ học độ dài/chủ đề. Khởi đầu chỉ
modulate K/V; chỉ thêm Q khi validation cho thấy K/V-only branch hoạt động nhưng
thiếu query alignment.

Cơ sở: HyperFormer, HINT, HyperLoader và input-dependent prompt/adaptor work.
Không paper nào kiểm chứng chính cấu hình PPLX-to-Qwen này; evidence cho tăng
ROUGE hiện là **insufficient**.

## Xếp hạng thử nghiệm

| Thứ tự | Hướng | Upside dự kiến | Rủi ro | Khả năng chẩn đoán |
|---:|---|---|---|---|
| 1 | Layerwise coupled-depth K/V | cao cho R-2/R-L | vừa | rất rõ |
| 2 | Gated side-memory | vừa, an toàn cho base path | vừa | rất rõ |
| 3 | Query-hierarchical exact-token read | cao cho R-2, giữ lexical | vừa-cao | rõ |
| 4 | Document-conditioned operator | cao nhưng chưa chắc | cao | vừa |

Nếu có hai GPU cho hai full runs song song, chạy **layerwise K/V** và **gated
side-memory** trước. Query-hierarchical đứng thứ ba vì cần query-specific mask và
generation cache mới; implementation sai rất dễ biến nó trở lại thành static
region prior. Không gộp hai hướng trong cùng run đầu. Chỉ khi một hướng thắng
direct-projection trên validation mới thử kết hợp.

## Protocol quyết định

Mọi arm giữ nguyên encoder/decoder checkpoint, prompt, data fingerprint, seed,
batch × accumulation, optimizer, số update, `last.pt`, grounded copy, decoding
và ROUGE-1.5.5. Chọn kiến trúc trên validation; khóa test cho phép đo cuối. Với
mỗi candidate, cần ba đối chứng:

1. `direct_projection + grounded_copy` được train lại cùng protocol;
2. candidate full;
3. cùng checkpoint candidate nhưng tắt branch khi inference để đo branch có
   thực sự được dùng hay chỉ thay đổi optimization.

Log tối thiểu: gate theo layer, base/new output RMS, attention entropy, logit
delta khi tắt branch, grounded-copy mass và gradient norm riêng của branch. Một
candidate bị loại nếu validation không thắng direct control, branch gần như
dormant, hoặc chỉ tăng R-2 bằng cách làm R-1/R-L giảm.

## Ranh giới claim

Nguồn hiện có tạo cơ sở thiết kế, không tạo bảo đảm tăng điểm. Chênh lệch test
đã quan sát nhiều lần không còn là một test set sạch để tiếp tục chọn kiến trúc.
Một claim paper vững cần cấu hình được chọn trên validation, sau đó đánh giá trên
một test protocol khóa trước và lặp seed hoặc paired bootstrap. Bốn hướng trên là
bốn giả thuyết kiến trúc có thể bác bỏ, chưa phải bốn đóng góp đã được chứng minh.
