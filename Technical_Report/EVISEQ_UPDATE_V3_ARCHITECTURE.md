# EviSeq Update v3: cải tiến kiến trúc đọc nguồn với CE-only

> **Thiết kế ban đầu, đã được thay thế.** Sau yêu cầu xử lý bốn hạn chế và dùng
> 4×128, cấu hình hiện hành là hierarchical coverage + norm-preserving fusion,
> tắt cross gate mới trong backbone. Xem
> [report bản sửa hiện tại](EVISEQ_UPDATE_V3_COVERAGE_REVISION.md).
> Các mô tả/defaults và 253 tests phía dưới thuộc lần triển khai trước.

Ngày: 2026-09-08. Tài liệu ghi lại cơ sở nghiên cứu, thiết kế đã chọn và protocol
đánh giá. Phạm vi triển khai riêng: `src/eviseq_update_v3`, xuất phát từ bản CE-only
v2 tại commit `381ebe7`; API generation temperature/top-p được giữ lại từ nhánh
hiện tại. Phần 9 ghi kết quả kiểm tra local; chưa có run PubMed của v3.

## 1. Kết quả đã biết và câu hỏi cần giải quyết

| Run do người dùng cung cấp | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| eviseq_new | 49.626 | 21.901 | 45.895 |
| eviseq_update v1 | 49.352 | 21.734 | 45.598 |
| eviseq_update v2 | 49.488 | 21.953 | 45.776 |
| v2 − new | −0.138 | +0.052 | −0.119 |
| v2 − v1 | +0.136 | +0.219 | +0.178 |

V2 tăng ROUGE-2 so với new, nhưng chưa vượt cả ba metric. Đây là một tín hiệu
đáng thử tiếp; chênh lệch nhỏ và chưa có predictions, nhiều seed hay khoảng tin
cậy để kết luận cải thiện chắc chắn. Không suy ra trực tiếp rằng ROUGE-1 giảm do
thiếu ý, ROUGE-L giảm do kém mạch lạc, hoặc ROUGE-2 tăng chứng minh học cụm từ tốt
hơn. Length, precision/recall, lựa chọn từ và checkpoint đều có thể ảnh hưởng.

Baseline new được xác nhận dùng 1 GPU, batch48, accum2, epoch4, clip1.0. V1 từng
dùng 2 GPU, batch84/GPU, accum1, epoch5. Protocol thực của kết quả v2 mới báo chưa
được xác minh; YAML trong repo không thay thế resolved config của run đó. Không
dùng các chênh lệch trên làm phép cô lập tác dụng kiến trúc.

Mục tiêu v3 là kiểm tra một cải tiến kiến trúc dưới cùng supervised CE, trước khi
thử objective bổ sung. Training không có contrastive learning, R-Drop, NEFTune,
pseudo-reference, candidate generation hay self-improvement. Temperature/top-p
là tùy chọn của API sinh; không đi vào CE hay tạo dữ liệu training.

## 2. Hai giới hạn thiết kế quan sát được ở v2

### 2.1. Cross-attention có nhiều head nhưng gate chưa phụ thuộc bước sinh

V2 đã thêm cross-attention nhiều head/GQA vào mọi decoder layer, giữa self-attention
và FFN. QK phụ thuộc decoder hidden nên việc đọc nguồn vốn đã thay đổi theo prefix.
Tuy nhiên, cross residual được nhân với một scalar học được cho mỗi layer:

```text
u_t = cross_norm(hidden_t)
cross_t = W_O concat_h Attention(Q_h(u_t), K_h, V_h)
hidden'_t = hidden_t + gamma_layer * cross_t
```

`gamma_layer` giống nhau với mọi token và mọi head trong layer. Thiết kế này chưa
cung cấp một gate riêng để decoder điều chỉnh đóng góp của từng head tại từng
bước. Đây là một ràng buộc biểu diễn; chưa phải bằng chứng rằng gate scalar gây
regression trong run v2.

### 2.2. Semantic read cuối chỉ có một attention map

Ngoài cross-attention backbone, v2 có semantic read độc lập với lexical copy tại
output head. Nó đọc native encoder positions bằng một attention rank128, tạo
context rồi cộng residual trước LM head. Mọi chiều của semantic context dùng
cùng một phân phối attention trên nguồn.

Giới hạn RMS residual hiện là 0.10 lần RMS decoder hidden. Chưa có log cho biết
nhánh này thường xuyên gần cap, vì vậy không tăng cap chỉ dựa trên điểm ROUGE.
V3 giữ cap này để kiểm tra thay đổi cách đọc nguồn trước.

## 3. Cơ sở nghiên cứu và phần áp dụng của EviSeq

[Attention Is All You Need, Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
đưa ra multihead attention: các head tính attention riêng rồi kết hợp qua output
projection. Đây là nền tảng cho việc tách semantic read thành nhiều head; không
được claim multihead attention là đóng góp mới của EviSeq.

[Gated Attention for Large Language Models: Non-linearity, Sparsity, and
Attention-Sink-Free, Qiu et al., NeurIPS 2025](https://arxiv.org/html/2505.06708v1)
khảo sát vị trí và granularity của gate. Paper tìm thấy lợi ích của gate phụ thuộc
query sau SDPA trong các LLM được huấn luyện ở nghiên cứu đó. Kết quả không xác
nhận ROUGE PubMed của EviSeq. V3 áp dụng headwise gating ở cross-attention mới thêm
vào decoder, dùng `2 * sigmoid` với zero-init để bảo toàn output ban đầu. Hệ số2
và việc giữ static residual gate là lựa chọn tích hợp của dự án, khác sigmoid
nguyên bản của paper. Không nhận các cải thiện attention sink/tốc độ trong paper
là kết quả đã đo của v3.

Thiết kế cụ thể của v3 kết hợp quyền điều chỉnh từng cross head theo prefix với
semantic read nhiều head, trong khi lexical copy vẫn độc lập. Đây là một giả
thuyết kiến trúc cần ablation; việc ghép các thành phần có tiền lệ không tự tạo
ra bằng chứng novelty hay giá trị khoa học đủ cho một paper.

## 4. Thay đổi A: query-dependent head gate trong decoder

Ký hiệu `B` là batch, `T` số vị trí decoder, `D` decoder hidden width, `H_c` số
query heads của cross-attention, `d_c` head dimension. Giữ toàn bộ Q/K/V, GQA,
source masks, source bias và static cross gate của v2. Với output SDPA:

```text
U                       [B,T,D]       = cross_norm(hidden)
O                       [B,H_c,T,d_c] = SDPA(Q(U), K(memory), V(value_memory))
G                       [B,T,H_c]     = 2 * sigmoid(U W_g + b_g)
O'_b,h,t,:                            = G_b,t,h * O_b,h,t,:
cross                                = W_O concat_heads(O')
hidden'                              = hidden + gamma_layer * cross
```

`W_g` và `b_g` khởi tạo bằng0 nên `G=1` tại initialization. Đây là gate theo query
head; dù GQA chia sẻ KV heads, mỗi query head vẫn có hệ số riêng. Gate có khoảng
`(0,2)` và có thể giảm đóng góp gần0 hoặc tăng tới gần2 lần trước static gate.
Không gọi nó là một chặn RMS cho cross residual.

Config bật/tắt: `decoder.query_cross_gate`, mặc định v3 là `true`.

Đặt gate trước `o_proj` để điều chỉnh từng head trước khi các head trộn vào hidden
space. Static `gamma_layer` tiếp tục điều chỉnh mức residual chung của layer.
Gate mới không dùng source/reference labels, không cần một loss riêng và không
dùng target tương lai. Các vị trí teacher forcing vẫn tính song song.

Khi tắt gate mới, graph cross-attention trở lại v2. Khi bật tại initialization,
output phải khớp trường hợp tắt với cùng shared weights và đầu vào. Việc tạo
tham số mới phải giữ RNG initialization của các module chung, để phép đối chứng
không vô tình đổi pretrained/copy/bridge weights.

## 5. Thay đổi B: native semantic read nhiều head, cùng tổng rank

Mặc định v3 dùng `H_s=4`, tổng rank `R=128`, mỗi head `d_s=32`. V2 là `H_s=1`,
`R=128`. Tổng output width của Q/K/V và output projection vẫn128; không tăng
rank cùng lượt đổi số head.

Các trường config tương ứng: `decoder.grounded_copy.semantic_read.num_heads=4`,
`rank=128`, `attention=independent_source`, `max_relative_rms=0.10`.

Gọi `H0` là encoder final state sau base projection, `b_j` là source prior, `m_j`
là native content mask, `h_t` là decoder hidden sau final norm. Với từng head:

```text
Q = reshape_heads(W_Q RMSNorm(h))             [B,H_s,T,d_s]
K = RMSNorm_per_head(reshape_heads(
        W_K RMSNorm(H0)))                    [B,H_s,S,d_s]
V = reshape_heads(W_V RMSNorm(H0))            [B,H_s,S,d_s]
A_h = softmax_j(Q_h K_h^T / sqrt(d_s) + b_j) [B,T,S]
C_h = A_h V_h                                [B,T,d_s]
C = concat_heads(C_h)                        [B,T,R]
```

Attention phải bằng0 ở các vị trí không hợp lệ. Trường hợp source content rỗng
phải tạo semantic contribution bằng0 và LM fallback hữu hạn. Đường một head dùng
log-softmax với finite floor rồi mask lại; đường nhiều head dùng hành vi safe
fully-masked attention của SDPA, đã kiểm tra trên CPU FP32/BF16 ở môi trường local.

Với nhiều head, implementation dùng SDPA, dropout0, source bias và mask `-inf`,
với hàng source rỗng trả context0; fusion cũng kiểm tra `semantic_mask.any()` để
fallback về hidden gốc. Công thức `A_h` trên mô tả attention về mặt
toán học; không yêu cầu giữ tensor `A_h` ở Python. Cache có thể lưu K/V dạng phẳng
`[B,S,R]` sau khi normalize key theo head, rồi reshape tại read. Với một head,
giữ đường explicit log-softmax v2 để bảo toàn thứ tự phép tính của control.

Sau khi concat, giữ cách fusion của v2: một RMSNorm trên toàn bộ context rank128,
một scalar semantic gate theo query/context, output projection zero-init, rồi
phép giới hạn RMS trơn cũ. Không thêm headwise semantic gate ở đây:

```text
c_norm = RMSNorm(C)
beta = sigmoid(W_beta [concat_heads(Q); c_norm] + b_beta)
raw = beta * W_out(c_norm)
cap = rho * RMS(h), rho = 0.10
delta = raw * cap / sqrt(cap^2 + RMS(raw)^2 + 1e-12)
h_lm = h + delta
```

Với `H_s=1`, RMSNorm key, attention scaling và context fusion phải khớp semantic
read v2. Với `H_s=4`, attention maps thay đổi dù Q/K/V cùng tổng width; không coi
đây là phép biến đổi bảo toàn function sau khi đã học. Zero-init `W_out` chỉ bảo
toàn output LM ban đầu khi semantic residual bằng0.

Lexical copy tiếp tục có attention, gate và sparse token alignment riêng:

```text
P(y_t) = g_copy,t * P_copy(y_t) + (1 - g_copy,t) * P_LM(y_t | h_lm,t)
P_copy(v) = sum_{j: source_token_j = v} A_copy,t,j
L = -sum_{gold target positions t} log P(y_t)
```

Không buộc semantic heads bằng copy attention và không thay marginalization theo
token ID. Những khác biệt attention giữa các head tự được học qua CE; không ép
diversity bằng auxiliary loss.

## 6. Gradient, cache và chi phí dự kiến

CE truyền qua output mixture tới decoder, cross-attention, head gate mới,
semantic read, lexical copy và upstream encoder/AFMR đang được mở train theo
stage. Không detach main path. Static cross gate ban đầu dương nên gate mới có
thể nhận gradient ở backward đầu. Với semantic `W_out=0`, output projection
nhận gradient trước; q/k/v/gate của semantic read có thể cần các update sau khi
output projection đã khác0 để nhận tín hiệu có ích.

Source K/V phụ thuộc source, prompt/output budget cố định của mẫu và weights,
nên vẫn cache một lần ở generation.
Cross head gate tính lại theo hidden hiện tại; không đưa gate theo prefix vào
source cache. Cache semantic lưu đủ tổng width; model/checkpoint spec giữ cấu hình số head, hỗ trợ
`index_select` theo batch khi các mẫu kết thúc generation. Cached/full-prefix, greedy và sampling
phải dùng cùng công thức đọc nguồn.

Gate headwise thêm khoảng `(D + 1) * H_c` tham số trên mỗi decoder layer, tùy
linear bias của implementation. Nó không tạo thêm attention pass. Semantic Q/K/V
projections giữ tổng width128, và KV storage vẫn tỷ lệ `B*S*128`. Tuy nhiên, số
phần tử attention score về mặt toán học tăng từ `B*T*S` lên `4*B*T*S`; dù phần
dot-product giữ tổng rank, softmax không miễn phí. SDPA có thể tránh giữ toàn bộ
score/log-attention khi dùng kernel hỗ trợ, nên không suy ra peak VRAM tăng4 lần.
Chunked loss giới hạn working tensors trong phạm vi chunk. Kernel được chọn và
mask/source lengths thực quyết định chi phí cần đo trên GPU.

Không có encoder/decoder backbone forward thứ hai, candidate generation hay cache
evidence. Chưa có GPU benchmark nên không đưa phần trăm thời gian/VRAM tăng và
không chuyển con số overhead của paper sang implementation này.

## 7. Đối chứng tối thiểu và protocol PubMed

| Variant | Cross head gate | Semantic heads | Tổng semantic rank | Cap |
|---|---|---:|---:|---:|
| CE v2 control | Tắt | 1 | 128 | 0.10 |
| Gate only | Bật | 1 | 128 | 0.10 |
| Heads only | Tắt | 4 | 128 | 0.10 |
| V3 | Bật | 4 | 128 | 0.10 |

Control đầu tái tạo graph CE v2 trong cùng package/trainer, với shared parameters
và preprocessing khớp. Nó không tự tái tạo score49.488/21.953/45.776 nếu protocol
server thực khác. Các variants cần train độc lập từ cùng pretrained backbones;
tắt một module của checkpoint đã train chỉ là diagnostic, không thay thế trained
ablation.

Giữ PPLX/decoder, dataset split và normalized content, seed, prompt, source/target
length, optimizer groups, LR, precision, checkpoint selection và ROUGE155 flags
giống nhau. Protocol căn theo thông tin baseline đã biết:

- Global batch96: 2 GPU × batch48 × accum1, hoặc 1 GPU × batch48 × accum2.
- 4 epochs với stage split1 warm-up +3 full của CE v2 cục bộ, clip1.0.
- Dùng lịch linear decay riêng từng stage của CE v2; không đổi sang cosine đồng
  thời với thay đổi kiến trúc.
- Cùng số update và cách xử lý remainder/accumulation/DDP target-token weighting.
- Đánh giá cùng decoding đã định trước. Temperature/top-p có trong API nhưng
  không bật sampling chỉ để lấy score tốt hơn đối chứng greedy/deterministic.

Nếu cần dùng batch84/GPU vì yêu cầu vận hành, chạy cả control và v3 cùng protocol
đó và ghi global batch168; không gọi đây là tái lập baseline global96.

## 8. Diagnostic và tiêu chí đánh giá tiếp

Trước full run, cần kiểm tra parity control v2, masked/empty-source behavior,
causal alignment, cache compaction, dense/chunked loss và gradients FP32/BF16,
cập nhật gate mới, checkpoint architecture guards và DDP. Kết quả kiểm tra local
được ghi riêng ở phần 9; chúng không xác nhận hiệu quả summarization trên PubMed.

Trên validation, so ROUGE-1/2/L cùng precision/recall/F-score nếu evaluator xuất
đủ, độ dài summary, EOS/truncation và repetition. Có thể xem thêm gate theo layer/
token và RMS semantic residual/cap để kiểm tra cơ chế có được sử dụng. Những
diagnostic này là protocol phân tích đề xuất; không mặc định rằng training logger
hiện đã tự động ghi tất cả.

Chọn checkpoint/cấu hình bằng validation, định trước cách xử lý trade-off giữa
ba metric và báo đủ cả ba. Với yêu cầu vượt đồng thời R1/R2/RL, không chọn chỉ theo
R2 rồi che giảm ở R1/L. Một checkpoint có train/validation CE thấp hơn vẫn cần
generation validation để xác định tác động ROUGE.

Giữ test để báo cấu hình đã chốt, không dùng các vòng xem test liên tiếp để tune
gate/head/cap. Các kết quả test đã biết vẫn phải được ghi nhận minh bạch. Khi có
predictions, dùng paired bootstrap và nhiều seed khi ngân sách cho phép; khoảng
tin cậy theo mẫu và độ biến thiên theo seed trả lời hai câu hỏi khác nhau.

Thiết kế v3 có mục đích và đối chứng rõ ràng, nhưng chưa bảo đảm tăng ROUGE,
vượt T5Gemma/decoder-only hay đủ novelty cho paper. Các claim đó cần kết quả thực
nghiệm của kiến trúc CE-only và một so sánh công bằng trước khi thêm contrastive.

## 9. Rà soát phản biện sau triển khai

### 9.1. Điểm đã thay đổi và điểm chưa giải quyết

| Câu hỏi | Kết luận sau rà code |
|---|---|
| Gate chung mọi token/head ở mỗi layer? | Đã thêm gate theo token/query head, giữ scalar residual gate cũ. |
| Semantic context buộc chung một attention map? | Đã tách thành 4 attention maps, mỗi head 32 chiều. |
| Có bảo đảm đọc đủ nội dung hơn? | Không. Các head có thể cùng đọc một vùng; chưa có cơ chế theo dõi coverage. |
| Có bảo đảm giữ nguyên lợi thế ROUGE-2? | Không. Copy formula giữ nguyên nhưng cross gate đã học làm thay đổi hidden đầu vào copy. |
| Có xử lý trực tiếp thứ tự các ý trong summary? | Chưa thêm sentence planning hay trạng thái coverage; vẫn dựa vào pretrained decoder và CE. |
| Có loại bỏ perturbation sau final norm? | Không. V3 giữ cách cộng residual của v2 và cap0.10. |
| Có tăng rank mỗi head? | Không. Đây là đổi 1×128 thành 4×32, có trade-off giữa nhiều attention maps và chiều matching mỗi head. |

Không nên gọi 4×32 là một lớp biểu diễn luôn bao trùm hoặc tốt hơn 1×128. Một
head cũ có thể kết hợp đầy đủ128 chiều trong một score; mỗi head mới chỉ kết hợp
32 chiều, đổi lại các nhóm values được đọc với các distributions riêng. Số
heads thích hợp là câu hỏi thực nghiệm, không suy ra từ tổng rank không đổi.

Gate mới nhân **sau** SDPA. Với Q/K/V cố định, nó thay mức đóng góp của một head,
không chuyển attention của head đó sang một vùng source khác. Query ở các layer
sau và weights qua training có thể thích nghi, nhưng không được mô tả gate này
như một thuật toán sửa alignment tức thì hoặc một điểm confidence của evidence.

Các heads vẫn chia sẻ source prior và đọc values từ final-state anchor H0.
AFMR refined memory vẫn tác động qua cross-attention keys và source prior;
semantic K/V không trực tiếp lấy từ refined memory. Đây là lựa chọn bảo toàn
value anchor, không phải đứt gradient, và chưa được v3 thay đổi.

Semantic context vẫn được RMS-normalize rồi dùng một scalar gate; phần magnitude
của weighted context không được giữ nguyên. Residual vẫn cộng sau final norm.
Giới hạn `RMS(delta) <= 0.1 * RMS(h)` không giữ nguyên norm `h+delta` hoặc thứ tự
vocab logits. Vì vậy không thể coi cap là bằng chứng sẽ không đổi từ theo hướng
bất lợi cho ROUGE-1/L. Chưa đổi normalization/fusion cùng lúc để giữ ablation rõ.

### 9.2. Bằng chứng triển khai đã có

- Đối chiếu trực tiếp với source package CE v2 tại commit `381ebe7`, nạp tạm từ
  Git rồi xóa thư mục tạm tự động. Control `query_cross_gate=false,num_heads=1`
  khớp **chính xác** shared initialization, RNG, logits, chunked CE, toàn bộ
  gradients và architecture spec ở CPU FP32/BF16. Đã kích hoạt semantic output,
  depth/feature residual và focus output khác0; không chỉ so lúc zero-init.
- Toàn bộ suite local: **253 passed trong41.20s**, CPU offline, PyTorch2.12.1
  và Transformers5.16.1. Gồm kiểm tra causality và counterexample các heads
  trùng focus; không dùng kiểm thử phần mềm để suy ra tăng coverage/ROUGE.
- Gradient oracle so SDPA với phép tính tường minh từng head; dense/chunk CE và
  tất cả gradients khớp trong tolerance FP32/BF16. Q/K/V của từng semantic head,
  output và gates cập nhật qua CE trong cả hai stage.
- DDP2 tiến trình Gloo so với serial cùng effective batch đã qua: bao gồm v3,
  token counts không đều, accumulation remainder, batch đệm zero-label,
  clipping, chuyển stage và checkpoint/resume. Đây không phải thử NCCL/CUDA.
- Cached/full-prefix và batch compaction với **gates/residual đã khác zero**,
  future-target mutation, masked/empty source, checkpoint khác head count bị
  từ chối, standalone temperature/top-p và greedy đã được kiểm tra.
- Smoke tiny v3 đã qua2 stages/4 optimizer updates, checkpoint/resume/eval;
  CE toy từ4.4907 xuống4.4769. Con số này chỉ kiểm tra pipeline, không dự báo ROUGE.

Chưa có CUDA/PubMed run, head-diversity/gate-distribution measurements sau full
training, predictions để phân tích ROUGE, hoặc phép chứng minh đã khắc phục
regression. Kết luận đúng hiện tại là **v3 có hai thay đổi kiến trúc cụ thể,
triển khai đã kiểm tra local, còn giả thuyết tăng ROUGE chưa được xác nhận**.
