# EviSeq v3: 4×128, đọc phân cấp có trạng thái prefix và fusion giữ norm

> **Snapshot trước Luna review, đã được cập nhật.** Mặc định hiện tại bỏ hard
> partition và đặt head gate sau context normalization. Xem
> [bản sửa sau review](EVISEQ_UPDATE_V3_REVIEW_REVISION.md).
> Mô tả partition/gate và kết quả tests bên dưới thuộc snapshot trước.

Ngày: 2026-09-08. Đây là thiết kế tại thời điểm đó trong `src/eviseq_update_v3`, thay thế
bản v3 ban đầu có cross gate trong backbone và semantic4×32. Training chỉ dùng
gold-token CE. Temperature/top-p vẫn chỉ là API sinh candidates ngoài training.
Chưa có kết quả PubMed của bản sửa này.

## 1. Quyết định và giới hạn kết luận

| Vấn đề cần xử lý | Thay đổi cụ thể | Phần chưa thể bảo đảm |
|---|---|---|
| Head giảm xuống32 chiều và có thể cùng đọc một vùng | 4 heads×128, mỗi head nhận tập vùng nguồn rời nhau; gate riêng trên output từng semantic head | Các vùng khác nhau vẫn có thể chứa thông tin trùng nhau |
| Chưa theo dõi nội dung prefix đã dùng | Usage tracker theo vùng, cumulative coverage và prior tiếp nối vùng đang diễn đạt | Usage là ước lượng học được, không phải xác nhận fact đã được tóm tắt đúng |
| Residual sau final norm làm trôi magnitude LM input | Bỏ thành phần radial, chặn phần tangent rồi chuẩn hóa về norm hidden gốc | Thứ tự logits vẫn có thể đổi; đó là cách nhánh mới thay nội dung dự đoán |
| Gate backbone có thể làm đổi query của copy | Mặc định tắt gate backbone mới; mọi thay đổi đặt trong nhánh semantic/LM | Shared encoder/decoder/prior vẫn học qua CE, nên không thể cam kết R2 không giảm |

Không có kiến trúc hữu ích nào vừa được phép sửa phân phối dự đoán vừa bảo đảm
mọi lựa chọn từ hoặc ROUGE trên dữ liệu chưa chạy giữ nguyên. Bản sửa này giải
quyết các đường tác động và hạn chế biểu diễn cụ thể; nó chưa chứng minh đã vượt
new/T5Gemma hoặc có thể giữ nguyên mức tăng R2 của run v2.

## 2. Cơ sở phương pháp và phần tích hợp riêng

- [See et al., Get To The Point, ACL2017](https://aclanthology.org/P17-1099/):
  pointer-generator và coverage là tiền lệ để theo dõi việc sử dụng nguồn và
  hạn chế lặp. Bản này không chép coverage loss của paper: chỉ giữ CE, dùng một
  tracker độc lập với coverage để tính prefix bằng cumulative sum song song.
- [Cohan et al., A Discourse-Aware Attention Model, NAACL2018](https://aclanthology.org/N18-2097/):
  tiền lệ attention phân cấp cho tài liệu dài và summarization khoa học. Không
  gọi các block64 tokens ở đây là discourse sections của bài báo.
- [Pang et al., Long Document Summarization with Top-down and Bottom-up Inference,
  EACL2023](https://aclanthology.org/2023.findings-eacl.94/): cơ sở xem xét đồng
  thời mức vùng và token. Model của paper điều chỉnh representation qua hai
  chiều inference; implementation ở đây là output-side hierarchical retrieval,
  không tái triển khai hay nhận kết quả của paper làm kết quả EviSeq.
- [Qiu et al., Gated Attention, NeurIPS2025](https://arxiv.org/abs/2505.06708):
  tiền lệ gate theo query/head sau attention. Ở bản sửa, gate đặt trên semantic
  context, không thay attention backbone mặc định. `2*sigmoid` zero-init là
  adaptation giữ hệ số ban đầu bằng1.

Phân công vùng rời nhau bằng modulo, usage tracker song song với CE, và tangent
fusion dưới đây là lựa chọn tích hợp của dự án. Không có paper nguồn nào được
viện dẫn như bằng chứng rằng tổ hợp này chắc chắn tăng ROUGE PubMed. Không claim
multihead/hierarchy/coverage/gating là những phát minh mới của EviSeq.

## 3. Graph tổng thể

```text
encoder ── final-state value anchor H0 ── semantic K/V ── source regions
   └── AFMR refined keys/prior ── v2 cross-attention backbone ── h_t

h_t ── lexical copy attention/gate ───────────────────────────── P_copy
 │
 ├── observed-prefix tracker ── cumulative coverage / latest region
 │
 └── 4×128 semantic queries ── region selection ── within-region token read
          ── per-head gate ── concat ── scalar gate/output projection
          ── tangent projection + bounded norm-preserving fusion ── LM ── P_LM

P = g_copy P_copy + (1-g_copy) P_LM
loss = gold-token cross entropy(P)
```

Backbone giữ cross-attention nhiều head/GQA của v2 ở mọi layer và static cross
gate cũ. `decoder.query_cross_gate=false` là mặc định. Source semantic K/V vẫn
lấy từ H0; AFMR refined memory ảnh hưởng qua backbone keys và source prior.
Không chạy thêm encoder, decoder hoặc sinh summary phụ trong training.

## 4. Matching đầy đủ 4×128 và phạm vi đọc rời nhau

`semantic_read.rank` là **tổng rank**:512, `num_heads=4`, mỗi head128. Q/K/V đều
project ra512 rồi reshape theo head. Output projection nhận concat512. Keys
chuẩn hóa RMS theo128 chiều từng head, scale attention là `sqrt(128)`.

Source content được đánh số compact sau khi bỏ prompt/padding. Vùng `r` gồm64
content positions liên tiếp. Vùng cuối có thể ngắn hơn. Mỗi vùng thuộc đúng một
head `owner(r)=r mod4`; mỗi head vẫn có thể đọc nhiều vùng ở nhiều đoạn tài liệu.
Không bỏ top-K vùng: hợp của các tập head vẫn phủ toàn bộ source nhìn thấy.

Đây là phân vùng cứng của source positions, không phải phân loại ngữ nghĩa hoặc
phát hiện các phần Objective/Methods/Results/Conclusion. Block boundary có thể
cắt một câu; encoder vẫn encode toàn bộ source như cũ nên mỗi token giữ ngữ cảnh
encoder. Khi tài liệu có ít hơn4 vùng, một số head không có vùng hợp lệ và trả0.

Với mỗi head, pooled region key là mean của token keys hợp lệ rồi RMSNorm;
region prior là mean source prior. Không có labels/reference trong source cache.

```text
s_region[h,t,r] = q[h,t]·K_region[h,r]/sqrt(128) + b_region[r]
                 - lambda_cov log(1 + C[t,r]/coverage_scale)
                 + lambda_cont R_recent[t,r]
pi[h,t,:]       = masked_softmax(s_region, valid_region AND owner==h)
a[h,t,j|r]      = softmax_within_region(q[h,t]·K_token[h,j]/sqrt(128) + b_token[j])
A[h,t,j]        = pi[h,t,region(j)] * a[h,t,j|region(j)]
c[h,t]          = sum_j A[h,t,j] V[h,j]
gain[h,t]       = 2*sigmoid(W_head RMSNorm(h_t) + b_head)[h]
c_t            = concat_h(gain[h,t] * c[h,t])
```

Region-first normalization tách xác suất chọn vùng khỏi số token trong vùng;
vùng dài không tự có nhiều mass chỉ vì nhiều token. Implementation dùng
scatter-reduce max và scatter-add denominator, tránh softmax NaN ở hàng rỗng.
Đường hierarchical hiện materialize attention theo từng CE chunk; không tuyên
bố nó đang dùng FlashAttention hoặc chắc chắn nhanh hơn đường SDPA cũ.

`partition_heads=false` là ablation cho các head đọc mọi vùng. Trong mặc định
true, attention supports của các head rời nhau theo source position. Điều đó
không bảo đảm nội dung các vùng khác nhau không trùng fact.

## 5. Prefix coverage và tiếp nối nội dung

Tại vị trí decoder`t`, hidden`h_t` chỉ thấy các input tokens đến`t`; nó dự đoán
token`t+1`. Tracker nhận `h_t`, không nhận label`t+1` hoặc phần summary tương lai:

```text
u_t = masked_softmax(W_track RMSNorm(h_t) · K_region_flat^T / sqrt(512) + b_region)
m_t = sigmoid(W_usage RMSNorm(h_t) + b_usage)
C_t = C_initial + sum_{i<=t} summary_input_mask_i * m_i * u_i
R_recent_t = u của input summary hợp lệ gần nhất; bằng0 nếu chưa có summary
```

`summary_input_mask` được suy ra từ vị trí thật sau attention mask và số token
prompt cố định của mẫu. Nó không dùng giá trị hoặc mask reference labels. Prompt
và padding không tăng coverage. Vì input`t` đã có trong prefix khi dự đoán`t+1`,
inclusive cumulative sum ở đây là causal.

Tracker độc lập với coverage nên train tính tất cả`u_t` và cumulative sum song
song. Generation chỉ cộng usage của token vừa nhận, lưu `[B,num_regions]` cho
coverage và latest-region state. Prefill xử lý prompt; không đánh dấu prompt như
một phần nội dung đã được tóm tắt.

`lambda_cov` và `lambda_cont` là các scalar học được, giới hạn sigmoid; mặc định
init0.2/max2.0, `coverage_scale=8`. Coverage giảm ưu tiên vùng đã tích lũy usage;
continuity tăng ưu tiên vùng mà prefix đang diễn đạt để không đổi vùng sau mỗi
token. Có thể tắt từng hiệu ứng bằng `use_coverage`/`use_continuity` để ablate.

Đây là kế hoạch đọc nguồn dựa trên prefix, chưa phải một sentence planner xuất
danh sách các fact hoặc một thuật toán biết chắc fact nào đã được hoàn thành.
Tracker có thể sai hoặc học usage không phù hợp. CE và phép đo validation phải
xác minh nó có giúp tránh lặp, giữ đủ ý và diễn đạt dài hơn hay không.

## 6. Fusion giữ norm, không cộng residual tự do sau final norm

Giữ query/gate/output projection semantic nhưng đổi phép fusion với hidden đã
qua final norm. Gọi `d_raw` là correction do head dự đoán:

```text
d_perp = d_raw - h * dot(h,d_raw)/dot(h,h)
cap = rho * RMS(h), rho=0.10
d = d_perp * cap / sqrt(cap^2 + RMS(d_perp)^2 + 1e-12)
h_LM = (h+d) * ||h|| / ||h+d||
```

Zero hidden có fallback bằng0. Source không hợp lệ fallback đúng hidden gốc.
Với h khác0, correction vuông góc h trước renormalization; trong số học chính
xác, `||h_LM||=||h||` và độ dịch chuyển sau fusion không vượt giới hạn relative
rho của construction này. Tests cho phép sai số làm tròn FP32/BF16.

Output projection zero-init nên ban đầu correction0 và `h_LM=h` chính xác.
Gradient đầu mở output projection; các projections/tracker/gates học sau đó.
Không detach hidden, source, prefix scan hoặc main loss trong training.

Phép này giữ magnitude của đầu vào vocabulary head; **không giữ nguyên logits**.
Thay hướng hidden là điều cần thiết để dự đoán từ khác. Giữ norm không phải bảo
đảm câu tự nhiên hơn hoặc bảo đảm ROUGE-L; đó vẫn là mục tiêu thực nghiệm.

## 7. Bảo toàn đường copy và giới hạn bảo vệ ROUGE-2

Copy dùng hidden`h` trước semantic fusion và cùng contextual/lexical keys, gate,
token-ID marginalization của v2. Gate mới trong backbone mặc định tắt. Khi shared
weights và source/prefix cố định, thay các tham số riêng của semantic/planner
không làm đổi decoder hidden hoặc phân phối copy. Đã có test trực tiếp cho điều này.

Điều không được cam kết: sau training, shared encoder/decoder/source prior vẫn
nhận gradient từ CE trộn LM/copy. Vì vậy v2 và revised-v3 có thể đi theo hai quỹ
đạo tối ưu khác nhau; ngay cả P_copy sau train cũng có thể khác. Muốn đóng băng
toàn bộ baseline hoặc chạy teacher riêng sẽ là một bài toán/training protocol
khác; bản này không âm thầm thêm những cơ chế đó.

## 8. Chi phí, cache và DDP

Với decoder width1024, copy rank128:

| Semantic variant | Semantic + planner params | Tổng grounded head params |
|---|---:|---:|
| V2 1×128 | 524,545 | 918,018 |
| V3 ban đầu4×32 | 524,545 | 918,018 |
| Revised v3 4×128 + planner/head gates | 2,627,592 | 3,021,065 |

Đếm bằng code module, chưa phải số tham số toàn model. Semantic K/V rộng512 thay
128 nên riêng phần cache này tăng4 lần; không suy ra tổng VRAM hoặc thời gian
train tăng4 lần. Có thêm tracker attention trên vùng và các phép scatter/cumsum.
Chi phí token matching gần tỷ lệ tổng rank, nên cần đo throughput/peak VRAM thật.

CE chunking giữ được: plan tính trên full prefix một lần, từng chunk nhận đúng
slice của coverage/recent. Các slice là tensor nằm trong graph, không có mutable
state cập nhật trong checkpointed loss. Không reset coverage ở ranh giới chunk.

History generation tách khỏi source cache, reset khi bắt đầu/kết thúc request,
và chọn lại các rows cùng self/cross/source caches khi compact batch. Cached
decoding trong training bị từ chối; training dùng prefix scan song song.
Checkpoint spec ghi rank/heads/attention/fusion/region size/partition và planner
flags/scales/maxima; checkpoint khác graph bị từ chối trước khi nạp weights.

## 9. Chạy và đối chứng

`scripts/run_pubmed_pair.sh` mặc định: revised-v3, 4×128, CE-only, PPLX rồi
Qwen-Embedding tuần tự, mỗi run2GPU; batch48/GPU×2×accum1=96, clip1.0, 1 epoch
interface warmup +3 full, linear decay riêng từng stage. Không thay scheduler.

Chỉ PPLX và eval validation khi chọn kiến trúc:

```bash
RUN_ENCODERS=pplx EVAL_SPLIT=validation bash scripts/run_pubmed_pair.sh
```

Đánh giá test cho cấu hình đã chốt: `EVAL_SPLIT=test` (mặc định). Đặt cùng
`ROUGE155_SCRIPT` và cùng flags/preprocessing với baseline; Python rouge1.0.0
trong evaluator không được coi là ROUGE155 của các score đã báo.

Các ablation phải dùng cùng data/model/seed/global batch/schedule/clip/decoding:

| Variant | SEMANTIC_RANK | SEMANTIC_HEADS | AFMR_SEMANTIC_VARIANT | Khác |
|---|---:|---:|---|---|
| V2 control | 128 | 1 | independent_bounded | CROSS_QUERY_GATE=false |
| Free heads4×32 | 128 | 4 | independent_bounded | CROSS_QUERY_GATE=false |
| Free heads4×128 | 512 | 4 | independent_bounded | CROSS_QUERY_GATE=false |
| Revised v3 | 512 | 4 | hierarchical_coverage | Defaults |
| Không partition | 512 | 4 | hierarchical_coverage | PARTITION_HEADS=false |
| Không coverage | 512 | 4 | hierarchical_coverage | USE_COVERAGE=false |
| Không continuity | 512 | 4 | hierarchical_coverage | USE_CONTINUITY=false |
| Fusion cũ | 512 | 4 | hierarchical_coverage | SEMANTIC_FUSION=residual |

Script chọn fusion=residual cho các independent/shared variants và norm_preserving
cho hierarchical_coverage, trừ khi override. V3 ban đầu có thể tái tạo bằng
free4×32 cộng `CROSS_QUERY_GATE=true`. Tên output ghi variant/rank/head/fusion/flags
để tránh trộn các ablation. Khi đổi seed/data/batch, dùng RUN_ROOT riêng.

Kiểm tra local: **286 tests passed trong44.29s**, CPU offline, gồm left padding,
copy isolation, ablation, gradient oracle, dense/chunk CE FP32/BF16 và DDP2
tiến trình Gloo. Smoke revised graph qua2 stages/4 updates, finite CE gradients
mọi tham số head, checkpoint/resume/greedy eval. Chưa benchmark CUDA và chưa có
ROUGE của bản revised; cần báo đủ R1/R2/L để kiểm chứng bốn mục tiêu.
