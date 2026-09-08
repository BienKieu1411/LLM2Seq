# EviSeq Update: giảm ROUGE, khác biệt training và ứng viên sửa v2

Ngày 2026-09-08. Phạm vi: code trong `src/eviseq_update` và thông tin người dùng
cung cấp. Không có checkpoint, prediction hay log của hai run thực để phân tích
theo mẫu; không thể suy ra nguyên nhân duy nhất hoặc ý nghĩa thống kê từ ba điểm tổng.

## 1. Những gì đã biết

| Run | ROUGE-1 | ROUGE-2 | ROUGE-L | Epoch | GPU | Batch/GPU | Accum | Batch hiệu dụng |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eviseq_new | 49.626 | 21.901 | 45.895 | 4 | 1 | 48 | 2 | 96 |
| eviseq_update v1 | 49.352 | 21.734 | 45.598 | 5 | 2 | 84 | 1 | 168 |
| Update − new | −0.274 | −0.167 | −0.297 | | | | | |

Người dùng xác nhận update dùng PPLX và Perl ROUGE155, new có gradient clipping 1.0,
đồng thời quan sát training loss update epoch 5 cao hơn new epoch 4. Recipe/code
update trước lần sửa này để clipping `null` theo yêu cầu bỏ clip trước đó. Không có
resolved config thực để xác minh các thiết lập khác hoặc con số loss cụ thể.

Với cùng N mẫu training, bỏ qua làm tròn batch cuối:

```text
new:     4 * N/96  optimizer updates
update:  5 * N/168 optimizer updates
ratio = (5/168)/(4/96) = 5/7 ≈ 0.7143
```

Update có **khoảng 28.6% ít optimizer updates hơn**, dù đã chạy thêm một epoch.
Mỗi epoch update có khoảng 42.9% ít updates hơn. Số mẫu đã xem và số lần cập nhật
là hai ngân sách khác nhau. Scheduler trong code giảm LR theo số bước của từng
stage, nên đổi batch còn đổi lịch LR theo bước và độ nhiễu gradient.

Training loss cao hơn phù hợp với khả năng chưa tối ưu/fitting tốt bằng baseline;
không chứng minh đứt gradient, không chứng minh overfit và cũng không loại trừ
kiến trúc khó tối ưu hơn. Train CE được ghi khi weights còn thay đổi trong epoch;
nên so thêm validation CE tại checkpoint cố định, cùng dữ liệu và cách tính loss.
Clipping 1.0 hay không clip là một biến thực nghiệm khác; chưa biết đóng góp riêng
của nó vào chênh lệch ROUGE. Batch và LR cần được xem xét cùng nhau, không có quy
tắc rằng batch lớn luôn tốt hoặc luôn xấu. [Shallue et al., 2019](https://arxiv.org/abs/1811.03600).

Recipe PubMed hiện có trong repo là 1 warm-up + 3 full, trong khi run update được
báo tới epoch 5. Do đó không được coi YAML cục bộ là resolved config của run thực.
Encoder của baseline, seed, LR, stage split, preprocessing và decode flags cần
được giữ giống nhau trong lần đối chứng tới; chúng chưa được xác minh đầy đủ ở đây.

## 2. Hai điểm yếu của graph v1 và thay đổi v2

### 2.1. Dùng chung attention cho copy và semantic read

V1 dùng keys gồm context encoder được pool theo char-overlap và lexical embedding
của token decoder. Cùng một attention vừa phân phối xác suất copy, vừa lấy semantic
values để thay đổi phân phối LM. Vì vậy CE qua đường LM cũng tác động trực tiếp vào
query/keys vốn dùng chọn token copy. Biên token copy còn quyết định cách pool values.

Đây là một ràng buộc thiết kế quan sát được từ code, **chưa phải bằng chứng gradient
hai mục tiêu xung đột trong run thực**. Context dùng chung giữa generation và copy
đã có trong pointer-generator và có thể hoạt động tốt; không nên gọi mọi shared
attention là lỗi. [See et al., 2017](https://aclanthology.org/P17-1099/).

V2 thêm query/key riêng cho semantic read và đọc trực tiếp các vị trí H0 encoder:

```text
q = W_q RMSNorm(h)
K = RMSNorm(W_k RMSNorm(H0)), V = W_v RMSNorm(H0)
a_sem = softmax(mask(q K^T / sqrt(rank) + source_prior))
c = a_sem V
```

Native content mask loại prompt/padding. Lexical copy attention, sparse alignment
và copy mixture vẫn giữ nguyên. CE vẫn truyền gradient tới hai nhánh, decoder,
encoder và prior chung; v2 chỉ bỏ việc dùng chung query/key và pooling theo token
copy ở đường semantic. Nó không đảm bảo mọi gradient sẽ cùng hướng hay ROUGE tăng.

### 2.2. Residual sau final norm chưa có giới hạn độ lớn

V1 cộng `beta * W_out RMSNorm(c)` sau final decoder norm, trước LM head. Dù
`beta` là sigmoid và khởi tạo nhỏ, norm của W_out có thể tăng, nên gate không cho
một chặn trên cố định đối với residual. W_out khởi tạo bằng 0 chỉ bảo toàn model
ban đầu, không bảo đảm residual tiếp tục nhỏ khi đã training.

V2 dùng phép giới hạn trơn:

```text
raw = sigmoid(W_beta [q; RMSNorm(c)] + b_beta) * W_out RMSNorm(c)
cap = rho * RMS(h)
delta = raw * cap / sqrt(cap^2 + RMS(raw)^2 + 1e-12)
h_lm = h + delta
```

Với `rho=0.10`, RMS(delta) không vượt 10% RMS(h) trước làm tròn dtype. Đây là chặn
đối với vector bổ sung, **không phải chặn 10% cho thay đổi logits, xác suất hay
hallucination**. LM head vẫn có thể nhạy với hướng của residual. 0.10 là lựa chọn
khởi điểm chưa được validation xác nhận; giới hạn quá chặt cũng có thể giảm khả năng học.

Phép tính không detach hay hard clamp; dùng vector norm có gradient hữu hạn tại
zero hidden. W_out vẫn zero-init, giữ weights chung/logits/loss ban đầu khớp
copy-only với cùng seed. W_out học ở backward đầu, các projections và gate mới
nhận gradient có ích sau đó. Tất cả nằm trong optimizer group cross_attention ở
cả warm-up và full fine-tuning.

### 2.3. Chi phí và compatibility

Với hidden=1024, copy rank=128, semantic rank=128: v1 thêm 262,401 tham số semantic;
v2 thêm 524,545; tổng grounded head v2 là 918,018. V2 thêm một lượt attention trên
source và cache keys/values native `[B,S,rank]`. Chưa đo throughput/VRAM trên GPU;
không coi đây là tối ưu tốc độ hay giải pháp mở rộng context vượt source đang thấy.

Resolved config cũ thiếu hai trường mới vẫn dùng shared, uncapped v1. Checkpoint
guard phân biệt attention/rank/cap, tránh dùng weights v1 dưới graph v2. V2 cần
train lại từ pretrained backbones. Checkpoint mới còn lưu training_spec với batch,
accumulation, world size, effective batch và clipping để việc đối chiếu sau rõ ràng hơn.

## 3. Protocol đã sửa và đối chứng nên chạy

PubMed mặc định trở về **global batch 96, 4 epochs, clip 1.0**. Recipe trực tiếp một
GPU là batch48/accum2; queue hai GPU là batch48/GPU/accum1. Stage split hiện là
1 warm-up + 3 full. Đây là căn chỉnh các thiết lập đã biết của baseline, không phải
cam kết tái tạo đúng điểm cũ khi seed, dữ liệu và cấu hình server chưa xác minh.

Để phân biệt hiệu ứng training với kiến trúc, dùng cùng PPLX, dữ liệu, preprocessing,
max lengths, LR/scheduler, stage epochs, seed, DDP setup và decode/evaluation flags:

| Đối chứng | Chọn trong queue | Mục đích |
|---|---|---|
| Copy-only | `AFMR_SEMANTIC_READ=false` | Kiểm tra baseline graph trên cùng training engine |
| V1 | `AFMR_SEMANTIC_VARIANT=shared_v1` | Đo lại update cũ dưới protocol batch96/clip1 |
| V2 | `AFMR_SEMANTIC_VARIANT=independent_bounded` | Đo tác dụng tổng của sửa nhánh semantic |
| Shared + cap | `AFMR_SEMANTIC_VARIANT=shared_bounded` | Tách tác dụng giới hạn residual |
| Independent không cap | `AFMR_SEMANTIC_VARIANT=independent_unbounded` | Tách tác dụng attention riêng |

Ưu tiên ba dòng đầu; hai dòng cuối phục vụ ablation khi kết quả validation đủ hứa
hẹn, không cần lập tức train cả năm run. Đặc biệt, V1 ở batch96/clip1 là đối chứng
cần thiết trước khi quy sự thay đổi điểm cho v2. Giữ output riêng theo variant và
protocol. Có thể chỉ chạy PPLX bằng `RUN_ENCODERS=pplx`.

Chọn hyperparameter/checkpoint bằng validation. Test chỉ dùng để đánh giá cấu hình
đã chốt, cùng Perl155 preprocessing/flags với baseline. Cần nhiều seed hoặc paired
bootstrap trên predictions để bàn về độ chắc chắn của mức chênh vài phần mười điểm;
ba score tổng hiện tại không cung cấp được khoảng tin cậy.

## 4. Phạm vi xác minh phần mềm

Tests của graph v2 bao gồm: bảo toàn weights chung/logits/loss lúc khởi tạo;
dense/chunked CE và gradient từng tham số trong FP32/BF16 autocast; gradient/cập nhật
weights mới qua hai stages; semantic read không backprop qua lexical copy query/key
khi xét riêng đường context LM; độc lập với tokenization copy; mask/padding/nguồn rỗng;
bound khi projection lớn và zero hidden; cache compaction; checkpoint round-trip/
compatibility; train, resume và greedy eval trên model nhỏ.

Test DDP chạy **hai tiến trình CPU/Gloo thật**, so gradient trước optimizer step và
weights sau step với một tiến trình ở cả hai stages. Nó kiểm tra token weighting,
accumulation dư, rank không có labels, LM/copy/v1/v2 và clipping 1.0 cho v1/v2.
Tests queue thực thi config generator thật nhưng giả lập CUDA/train/eval.

Các kiểm tra trên xác nhận cơ chế tính toán trong các ca đã thử. Máy hiện tại không
có CUDA để train PPLX/Qwen đầy đủ, đo VRAM/tốc độ NCCL hoặc đo ROUGE v2. Kết luận
"v2 vượt new/T5Gemma" chỉ được đưa ra sau thực nghiệm chất lượng tương ứng.

Kết quả kiểm tra cục bộ ngày 2026-09-08: **198 tests passed**, CPU/Gloo,
PyTorch 2.12.1, Transformers 5.16.1. Smoke qua `scripts/run_afmr.sh smoke` báo
`status=ok`, `semantic_attention=independent_source`, cap=0.1, hoàn thành 2 stages/
4 optimizer updates. Loss trên fixture giảm 4.490708 → 4.476895; con số này chỉ
xác nhận pipeline học trên fixture, không dự đoán ROUGE PubMed.
