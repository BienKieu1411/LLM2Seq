# Audit mục tiêu kiến trúc AFMR

Bảng xác minh ngắn gọn sau vòng review độc lập nằm tại
[VERIFICATION_REPORT.md](VERIFICATION_REPORT.md).

Ngày audit: 2026-09-10
Phạm vi: thiết kế AFMR, code nền `eviseq_new`/`eviseq_update`, giao thức train/eval PubMed và các giả thuyết dùng cho claim ROUGE.
Trạng thái: **đây là snapshot thiết kế; package AFMR đã có trên `main`, nhưng chưa có checkpoint hay điểm ROUGE AFMR**.

## 1. Trạng thái trích xuất và nguồn bằng chứng

Đã đối chiếu:

- `DESIGN.md`, `IMPLEMENTATION_PLAN.md`, `EVALUATION_PLAN.md`, `DECISION_REPORT.md`, `RESEARCH.md`, `ARCHITECTURE_AUDIT.md` và review lịch sử trong `PLAN_ARCHITECTURE_REVIEW.md`.
- Code nền trong `src/eviseq_update/eviseq_update/modeling/{grounded_copy,decoder,model}.py`, `training/{engine,optimizer,checkpoint}.py`, `data/sampling.py`, `runtime.py` và các script/config train/eval.
- Các mốc người dùng đã báo: `eviseq_new`, `eviseq_update_v2`, T5Gemma.
- Báo cáo SDACL tại [`/Users/kieugiangbien/Downloads/Paper/VAI/Summarization/SDACL.pdf`](/Users/kieugiangbien/Downloads/Paper/VAI/Summarization/SDACL.pdf), chỉ dùng cho vòng contrastive sau; không đưa objective đó vào AFMR.
- Nguồn gốc về probability mixture, pointer-generator, residual initialization, summarization dynamics, T5Gemma 2 và các kiến trúc foundation model hiện đại.
- Các nguyên tắc về likelihood, numerical stability, optimizer, regularization, sequence modeling và phương pháp thực nghiệm trong [Deep Learning](https://www.deeplearningbook.org/).

Snapshot code nền của audit là commit `88486a2`; `Paper/README.md` và
`Paper/main.pdf` đã sửa từ trước. Package AFMR hiện đã được track trên `main`;
những thay đổi triển khai nằm trong `src/afmr_core` và được kiểm tra
riêng bằng acceptance suite hiện hành.

Các mục sau là phân loại bằng chứng:

- **Đã xác nhận trong code/tài liệu:** có thể dùng để thiết kế contract.
- **Số liệu người dùng báo:** dùng làm mốc tham chiếu, chưa coi là phép tái lập.
- **Suy luận thiết kế:** giả thuyết cần thí nghiệm, không phải kết quả.
- **Chưa xác minh:** không được dùng trong claim chính trước khi tạo artifact tương ứng.

### Bảng bằng chứng

| Mệnh đề | Loại | Độ tin cậy | Cách xử lý |
|---|---|---:|---|
| `new` đạt R1/R2/RL = 49.626/21.901/45.895 | Số liệu người dùng báo | Trung bình | Rerun common-control, lưu predictions và resolved config |
| `update_v2` đạt 49.488/21.953/45.776 | Số liệu người dùng báo | Trung bình | Rerun cùng manifest trước khi gọi là baseline causal |
| T5Gemma đạt 49.580/21.990/45.463 | Số liệu người dùng báo | Trung bình | Rerun hoặc gắn nhãn historical; không chọn checkpoint từ test |
| V2 semantic reader dùng `H0n=RMS(H0)`, `K=RMS(Wk(H0n))`, `V=Wv(H0n)` | Code v2 | Cao | Giữ đúng thứ tự; viết test shape/value |
| Copy hiện hữu đọc `H0`; AFMR memory `M` phục vụ retrieval/cross-attention | Code v2 | Cao | Không cho semantic sửa copy state; API phải tách key/value source |
| `Wo=0` cho `delta=0` và `Ps=P0` ở forward | Suy ra từ công thức | Cao | Dùng parity endpoint, không gọi đây là candidate chất lượng |
| `Wo=0` làm Q/K/V không nhận gradient qua `u` ở backward đầu | Chain rule | Cao | C5 chỉ kiểm tra đúng chain rule; candidate chính dùng tiny |
| Probability mixture có thể tăng họ phân phối nhưng tăng cost/overfit risk | Paper MoS | Cao | Chỉ hai readout, chung vocabulary head, có hidden-fusion control |
| Cross-attention ở mọi layer có ích trong T5Gemma 2 | Ablation của paper | Cao cho T5Gemma, thấp cho EviSeq | Giữ cross-attention trong trunk; không suy ra ROUGE EviSeq |
| Main AFMR chắc chắn vượt target | Chưa có kết quả | Không có | Không claim; chỉ có acceptance gate và điều kiện dừng |
| Kiến trúc AFMR giảm hallucination | Chưa đo factuality | Không có | Để claim vòng sau; ROUGE không đủ làm bằng chứng |

## 2. Kết luận tổng hợp

Giả thuyết đáng kiểm tra là semantic read của v2 có thể tăng n-gram dài nhưng hidden residual làm thay đổi toàn bộ vocabulary prediction, từ đó làm giảm unigram/sequence overlap. V5.1 tách ba xác suất ở **output space**: vocabulary base `P0`, vocabulary semantic `Ps` và copy `Pcopy`. Semantic có mass riêng bị chặn, còn copy gate `g` của `new` giữ nguyên ở giá trị forward. Cách này trực tiếp kiểm tra trade-off R1/RL–R2 mà không thêm contrastive loss, self-improve, candidate mining, reranking, R-Drop hoặc NEFTune.

Tên chính xác của main là **copy-mass-preserving capped simplex**, không phải independent simplex. Semantic cap phụ thuộc giá trị `g` đã detach. Đây là lựa chọn có chủ ý để bảo vệ R2. Một simplex hoàn toàn độc lập có base floor được giữ làm control F để đo cái giá của việc cho router thay đổi copy mass.

Có một bất khả thi toán học cần ghi rõ. Ba yêu cầu sau không thể đồng thời đúng với mọi `g`:

1. `pi_copy = g` chính xác;
2. `pi_base >= f` với một floor dương cố định;
3. `pi_sem > 0` ngay cả khi `g > 1-f`.

Main ưu tiên (1), và thực hiện (2) khi còn đủ mass; khi `g` quá cao, semantic bị cap về zero và `pi_base=1-g` nhỏ hơn `f`. Control F ưu tiên floor thật và cho phép copy mass thay đổi. Không được đổi tên hai hành vi này để làm chúng có vẻ tương đương.

## 3. Cây lập luận thiết kế

```text
Mục tiêu: R1/RL > new và R2 > update_v2 (stretch: R2 > T5Gemma)
|
+-- Vấn đề cần kiểm tra
|   +-- nested semantic gate làm semantic mass = (1-g)*beta
|   +-- residual semantic thay đổi hidden đưa vào vocabulary và copy
|   +-- v4 thêm nhiều cơ chế nhưng không có causal isolation
|   +-- mốc lịch sử dùng protocol khác nhau
|
+-- Quyết định kiến trúc main
|   +-- giữ trunk new/v2, cross-attention mọi layer, copy alignment
|   +-- semantic reader flat rank 128, K=H0, V=H0
|   +-- hai vocabulary readout cùng W_lm
|   +-- mix ở probability space, pi_copy giữ g
|   +-- residual cap duy nhất; không gate .05 kép
|
+-- Quyết định tối ưu
|   +-- Wo=tiny là candidate; Wo=0 chỉ parity
|   +-- CE trên P hoàn chỉnh ở log-domain FP32
|   +-- token-normalized DDP và một scheduler toàn run
|   +-- fixed-small-set smoke trước PubMed
|
+-- Quyết định kiểm chứng
    +-- legacy endpoints, dense oracle, gradient/caching/compaction tests
    +-- 1GPU/2GPU cùng logical batches và token budget
    +-- validation chọn checkpoint; test một lần bằng ROUGE-1.5.5
    +-- ba seed + paired bootstrap; architecture claim tách system claim
```

## 4. Hợp đồng kiến trúc cuối cùng

### 4.1. Trunk và hai memory

Giữ encoder, AFMR và decoder cross-attention của `new/v2`. Ký hiệu:

- `H0 ∈ R^{B×S×D}`: value anchor dùng cho cross-attention/copy.
- `M ∈ R^{B×S×D}`: AFMR retrieval memory dùng làm key ở cross-attention và chỉ dùng làm semantic key trong variant `AFMR-KM`.
- `b ∈ R^{B×S}`: source prior sau khi đã mask.
- `W`: chiều rộng candidate copy sau character-overlap alignment; `W` không nhất thiết bằng native source length `S`.

`bridge.value_memory` là state bắt buộc. Nếu backend không tạo được state này,
v5 phải dừng với lỗi rõ ràng; fallback sang `bridge.memory` sẽ làm sai contract
`K=H0,V=H0` và phá parity với `new/v2`.

Các state phải được truyền riêng:

```text
prepare(
    copy_memory=H0,
    semantic_key_memory=H0,       # main; M trong AFMR-KM
    semantic_value_memory=H0,
    source_prior=b,
    copy_state=...
)
```

Không được dùng một tham số `memory` duy nhất rồi đổi config `key_source=M`, vì khi đó implementation có thể đổi nhầm key của copy hoặc vẫn thực tế dùng `H0`. Variant `K=M,V=H0` chỉ là named ablation; source prior phải giảm/tắt trong variant để đo double-count.

### 4.2. Semantic reader

Main giữ đúng graph đã có trong v2:

```text
H0n = RMS_FP32(H0)
q   = Wq(RMS_FP32(h))
Ks  = RMS_FP32(Wk(H0n))
Vs  = Wv(H0n)
A   = masked_softmax(q Ks^T / sqrt(r) + semantic_prior_scale * b)
c   = A Vs
u   = RMS_FP32(c)
d0  = Wo u
d   = smooth_relative_RMS_cap(d0, h, rho=0.10)
hs  = h + d
```

Để tái lập được cap, định nghĩa `RMS(x)=sqrt(mean_hidden(x^2))` ở FP32,
`t=max(rho*RMS(h),1e-6)`, `r=RMS(d0)`, `x=r/t`,
`d=tanh(x)/max(x,1e-6)*d0`, với giới hạn `tanh(x)/x→1` khi `x→0` và `d=0`
khi `r=0`. C5/C8 phải kiểm tra `RMS(d)≤t` trong tolerance và gradient hữu hạn.

Shapes:

```text
h, hs: B×T×D        q,u: B×T×128
Ks,Vs: B×S×128      A: B×T×S
```

`inner_gate=false`: không tạo semantic sigmoid gate `.05` thứ hai. RMS cap `rho=0.10` là bound duy nhất của residual main. Source mask rỗng trả context zero an toàn; không softmax một hàng toàn `-inf`.

Control `legacy_semantic` phải bật lại `semantic_gate_init=0.05` đúng như code
v2; nếu dùng `inner_gate=false` cho control thì C2 không còn là phép tái lập v2.

Control này phải dùng đúng graph v2:

```text
delta_v2 = sigmoid(gate_v2([q_v2, RMS(c_v2)])) * W_out_v2(RMS(c_v2))
P_semantic_v2 = softmax(W_lm(h + delta_v2))
P_v2 = (1-g) * P_semantic_v2 + g * Pcopy
```

`g`, copy state, source path, mask và cap (nếu bật trong v2 config) phải giữ
nguyên. C2 chỉ pass khi control khớp logits/loss của implementation v2 trong
tolerance đăng ký.

### 4.3. Output probability và cap

```text
P0    = softmax(z0), z0 = W_lm h
Ps    = softmax(zs), zs = W_lm hs
Pcopy = scatter_add(exp(log_attention), token_ids)

alpha_raw = alpha_max * sigmoid(r_sem) * evidence
 g_route  = stop_gradient(g)
 alpha    = min(alpha_raw, max(0, 1 - g_route - generate_reserve))
 pi_copy  = g
 pi_sem   = alpha
 pi_base  = 1 - g - alpha
 P        = pi_base*P0 + pi_sem*Ps + pi_copy*Pcopy
```

Giá trị pilot được đăng ký trước: `alpha_max=0.20`, `generate_reserve=0.05`, router bias để `alpha≈0.05` khi evidence đầy đủ. `g=0` khi copy mask rỗng; `evidence=0` khi semantic source mask rỗng. Cần log tỷ lệ cap-hit và `g` quantiles, vì cap-hit cao cho thấy semantic route thực tế không có đủ simplex mass.

`g_route` chỉ dùng cho feature và cap. `pi_copy` dùng `g` có gradient; do đó output giữ copy mass ở forward nhưng shared trunk vẫn có thể làm copy thay đổi sau các update. Claim đúng là **copy-mass preserving at the mixture interface**, không phải parameter isolation.

### 4.4. Independent control F

Control không được viết bằng `l_copy=logit(g)` kết hợp `l_base=log(1-g)` vì hai logit đó không tạo ra tỷ lệ copy mong muốn khi đưa vào softmax ba nhánh. Dùng:

```text
l_base = log(clamp(1-g_route, eps, 1)) + r_base
l_copy = log(clamp(g_route,   eps, 1)) + r_copy
l_sem  = log(eps_sem) + r_sem       # eps_sem=exp(-20) ở bước 0
q      = softmax([l_base,l_sem,l_copy])

pi_base = base_floor + (1-base_floor)*q_base
pi_sem  = (1-base_floor)*q_sem
pi_copy = (1-base_floor)*q_copy
```

`r_base`, `r_sem`, `r_copy` là output router khởi tạo zero; expert rỗng mask bằng `-inf`. `base_floor=0.05` chỉ thuộc control này. Nếu cần khôi phục gần `q=(1-g,0,g)`, kiểm tra numerically trên `g∈{0,0.1,0.5,0.9,0.999}` trước khi train.

### 4.5. Mixture NLL và gauge

Loss duy nhất của vòng architecture là gold CE trên **phân phối cuối**:

```text
l0 = log_softmax(z0)
ls = log_softmax(zs)
lc = log(Pcopy)                    # -inf nếu y không copyable
lp = logsumexp(
       log(pi_base)+l0,
       log(pi_sem)+ls,
       log(pi_copy)+lc,
     )
loss_sum = -sum(lp[label != -100])
```

Không dùng `pi_base*CE0 + pi_sem*CEs + pi_copy*CEcopy`; đó là objective khác. Tất cả mask, normalizer, `logsumexp`, `P`, gate và `h+d` chạy FP32. Matmul có thể BF16 theo common recipe.

Output score trước processor:

```text
Z0 = logsumexp(z0)
output_logits = log(P) + Z0
```

Nếu cả source route inactive, trả trực tiếp `z0` để giữ fallback legacy. `log(P)+Z0` là scalar gauge theo timestep; vì repetition penalty phụ thuộc dấu/độ lớn, phải chạy probe penalty 1.0 và setting headline 1.05. Temperature/top-k/top-p chỉ được áp lên final mixed scores trong sampled candidate API, không sample từng expert rồi mới mix.

### 4.6. Gradient contract

`Wo=0` cho:

```text
delta=0, Ps=P0, dL/dalpha=0 ở backward đầu
 dL/dWo hữu hạn nếu W_lm và u có signal
 dL/dQ,K,V = 0 qua đường u ở backward đầu
```

Vì vậy main candidate dùng `Wo=tiny_rms_1e-3`, còn zero chỉ là D parity/smoke. Tiny được định nghĩa tái lập: tạo `Wo_raw` bằng seed cố định, chạy một calibration batch không dùng label, tính `r=RMS(Wo_raw u)/RMS(h)`, rồi rescale `Wo=1e-3*Wo_raw/max(r,eps)`; yêu cầu `r_final∈[0.5e-3,2e-3]`. Phép cộng `h+d` ở FP32 trước khi cast để BF16 không xóa correction.

Đừng kết luận v5 chậm hơn v2 từ zero-init: v2 cũng có `W_out=0` và inner gate khoảng `.05`; so sánh phải ghi gradient/optimizer step thực tế. C5 yêu cầu sau một số optimizer steps, trên batch có semantic signal, Q/K/V/router nhận gradient hữu hạn khác zero.

### 4.7. Cache và compaction

Có thể cache theo source:

```text
H0,M, semantic K/V, copy token IDs/masks/bias, cross K/V
```

Phải tính lại theo prefix:

```text
h,q,u,alpha,g,z0,zs,output_logits
```

Không cache query, alpha, `u`, logits hoặc graph training. Khi EOS compaction/reorder, index-select đồng thời self KV, cross KV, semantic K/V, copy IDs/masks và source prior. Reset khi source, alignment version, checkpoint, dtype/device đổi.

## 5. Hợp đồng training chính xác

### 5.1. Logical batch và DDP

Tạo `canonical_global_batch_manifest` ghi thứ tự example IDs và số valid target tokens cho từng optimizer update. Hai chế độ tương đương phải là:

```text
1 GPU:  batch_per_gpu=48, accum=2
2 GPU:  batch_per_gpu=48, accum=1
```

Cả hai phải dùng cùng global examples, cùng target-token budget và cùng tail policy. Trong một logical accumulation window:

```text
N_global = all_reduce(sum_r local_valid_target_tokens_r)
L_r      = sum local token NLL trên rank r
loss_r   = world_size * L_r / N_global
```

DDP average một lần sẽ cho `sum_r L_r/N_global`. Không chia thêm cho world size, accumulation hoặc local token count. Clip chỉ sau backward cuối cùng, sau gradient synchronization và trước optimizer step.

### 5.2. Optimizer, scheduler, precision

Để so sánh công bằng, khóa một common recipe cho A/B/E/F:

- AdamW `beta1=.9, beta2=.95, eps=1e-8`; weight decay và nhóm no-decay (norm/bias) ghi trong resolved config.
- Một cosine schedule trên toàn bộ số optimizer updates; không reset scheduler ở stage transition. Nếu code cũ reset stage-local scheduler, phải sửa trước baseline rerun.
- FP32 master parameters/optimizer states + BF16 autocast là main. Từ chối FP16 nếu chưa có `GradScaler` và unscale trước clip.
- Log `grad_norm_pre_clip`, `clip_coefficient`, `grad_norm_post_clip`, `fraction_steps_clipped`; giá trị `clip_grad_norm_` trả về thường là norm trước clip, nên pre-clip khoảng 2.0 khi max norm 1.0 là bình thường.
- Không dùng epoch thay cho budget; ghi số updates và valid tokens.

Đây là chuyển hóa các nguyên tắc numerical stability/optimization của [chương Numerical Computation](https://www.deeplearningbook.org/contents/numerical.html), [chương Optimization](https://www.deeplearningbook.org/contents/optimization.html) và [chương Methodology](https://www.deeplearningbook.org/contents/ml.html) thành invariant kiểm thử.

### 5.3. Checkpoint và selection

Checkpoint phải chứa architecture fingerprint gồm `readout.mode`, `rank`, `num_heads`, key/value source, `inner_gate`, init mode, cap/reserve, gauge, router feature schema, tokenizer/alignment version và decoder settings. Resume/load từ chối mismatch.

Mọi model dùng `selection_metric=validation_ce`, `save_best=true`, cùng validation manifest và cùng tie-break. Evaluator phải kiểm tra checkpoint/config/split/decoder fingerprint **trước** khi trả metric từ prediction cache. Config T5Gemma lịch sử đang có `eval_strategy: no`, nên strong system comparison cần rerun hoặc ghi rõ final-checkpoint protocol cho tất cả model.

## 6. Hợp đồng đánh giá và target

### 6.1. Mốc lịch sử

| Hệ | R1 | R2 | RL | Trạng thái |
|---|---:|---:|---:|---|
| `eviseq_new` | 49.626 | 21.901 | 45.895 | historical, cần common rerun |
| `eviseq_update_v2` | 49.488 | 21.953 | 45.776 | historical, cần common rerun |
| T5Gemma | 49.580 | 21.990 | 45.463 | historical, cần common rerun |

Target được tách thành hai gate:

- **Architecture gate:** dưới common EviSeq recipe, một checkpoint AFMR phải `R1>49.626`, `R2>21.953`, `RL>45.895`; tối thiểu so với từng baseline rerun cùng seed/manifest.
- **System stretch:** sau khi rerun T5Gemma theo protocol minh bạch, một checkpoint duy nhất phải vượt `new` ở R1/RL và vượt T5Gemma ở R2. Không ghép metric từ các checkpoint khác nhau.

Nếu baseline rerun lệch mốc lịch sử, dùng common rerun làm mốc causal chính và giữ số lịch sử chỉ trong bảng provenance. Không được tuyên bố vượt T5Gemma bằng một baseline chưa cùng checkpoint-selection rule.

### 6.2. ROUGE và fairness

- Cùng processed split manifests; hard fail nếu có cross-split content overlap. `ALLOW_CROSS_SPLIT_CONTENT` không được bật ngầm.
- Cùng source/reference preprocessing, character-overlap alignment policy và visible source budget; ghi tokenizer-visible characters/tokens và truncation rate.
- Cùng greedy settings (`max_new_tokens`, `min_new_tokens`, repetition penalty, no-repeat n-gram) cho headline; sampled temperature/top-k/top-p là probe/candidate API riêng.
- Cùng Perl ROUGE-1.5.5 command, XML data/dependencies và normalization; nếu thiếu `XML::Parser`, lưu lỗi môi trường và không trộn Python ROUGE vào headline.
- Predictions phải có document ID, checkpoint/config/split/decoder fingerprint; stale cache bị từ chối.
- Tối thiểu seed `42/43/44`; báo từng seed, mean/spread, output length/copy rate/truncation.
- Đăng ký trước paired percentile bootstrap theo document trên cùng IDs
  (10.000 resamples, two-sided 95% CI); điều chỉnh family-wise bằng Holm ở mức
  `alpha=0.05` cho ba primary comparisons. Bootstrap không thay thế variance
  giữa seed và không được dùng để chọn checkpoint.

### 6.3. Causal ablation

A `new/common-control`, B `update_v2/common-control`, D zero parity, E tiny candidate, F independent simplex, G `K=M,V=H0`, H hidden interpolation, I constant alpha và J gauge probe. Không sweep test để chọn cap/gauge/epoch/length. E chỉ promotion sau fixed-small-set smoke và validation pilot.

Diễn giải:

- E > A/B và F/H/I: bằng chứng cho decomposition cụ thể, không tự động là định luật mixture.
- F > E nhưng R2 giảm: giữ E cho mục tiêu R2; không chọn F chỉ vì R1.
- H ngang E: không claim probability fusion là nguyên nhân.
- Chỉ R2 tăng, R1/RL giảm: kiểm tra length/copy/source visibility trước khi thêm module.
- ROUGE tăng nhưng chưa có factuality metric: không claim giảm hallucination.

## 7. Acceptance tests C1–C19

| ID | Điều kiện pass |
|---|---|
| C1 | `base_only` trùng output/loss/gradient `new` với cùng weights/input; mọi parameter vẫn ở optimizer group đúng |
| C2 | `legacy_semantic` trùng probability/loss/gradient v2 flat; raw logits chỉ so khi control dùng gauge `Zs` |
| C3 | Dense FP64/FP32 oracle bảo đảm mass=1, duplicate ID marginalization, copyable/non-copyable và mask rỗng |
| C4 | `Wo=0` đối chiếu trực tiếp kernel `_mix_logits` của `new`; không tự tạo oracle khác rồi gọi là parity |
| C5 | Chain rule đúng ở backward đầu; sau optimizer update Q/K/V/router có gradient khác zero ở batch có signal |
| C6 | Semantic rỗng → `alpha=0`; copy rỗng → `g=0`; cả hai rỗng → trả `z0`/`P0` |
| C7 | Future target/reference không đổi logits quá khứ; router không đọc gold/candidate |
| C8 | Dense/chunked logits, loss và gradient tương đương trong tolerance đã khóa |
| C9 | 1/2 GPU cùng canonical global batches, token budget và schedule cho update gần nhau |
| C10 | FP32/BF16 finite; reductions/gates/logsumexp/mixture/residual add ở FP32; FP16 bị từ chối nếu thiếu scaler |
| C11 | Gauge `log(P)+Z0` giữ endpoint với repetition penalty 1.0 và 1.05 |
| C12 | Teacher-forced/incremental cache, reorder, EOS compaction và source reset tương đương |
| C13 | Save/load/resume giữ output, optimizer, RNG, scheduler và fingerprint; mismatch bị từ chối |
| C14 | Temperature/top-k/top-p hợp lệ, áp sau final mixture, seed tái lập; greedy headline không bị đổi |
| C15 | Runner in resolved config, manifest hash, global batch, updates, dtype, clip và evaluator command |
| C16 | Prediction cache stale bị từ chối trước metric; không return sớm chỉ vì ID/reference prefix khớp |
| C17 | Với `g_route=stop_gradient(g)`, kiểm tra Jacobian `dP/dg=Pcopy-P0` và alpha không tạo đường trực tiếp vào g; log `dL/dg` thực tế riêng, shared-trunk drift báo riêng |
| C18 | Fixed set 16–32 examples: CE giảm rõ ràng, semantic/copy masks hoạt động; fail thì dừng trước PubMed |
| C19 | Canonical manifest replay: 1GPU/2GPU tạo cùng update IDs, token denominators và tail handling; log all-reduce denominator |

## 8. Quyết định giữ, sửa, bỏ, hoãn

| Thành phần | Quyết định | Lý do |
|---|---|---|
| Native encoder/AFMR/cross-attention mọi layer | Giữ | Giữ pretrained path và phù hợp ablation T5Gemma 2 |
| Copy alignment và duplicate marginalization | Giữ | Bảo vệ R2 và lexical fidelity |
| Flat semantic rank 128, một head | Giữ main | V2 đã có graph rõ; chưa có bằng chứng thiếu capacity |
| `K=H0,V=H0` | Main | Ít double-focus, dễ parity với v2 |
| `K=M,V=H0` | Variant G | Kiểm tra retrieval addressing tách khỏi content value |
| Probability mixture P0/Ps/Pcopy | Main | Test semantic ngoài hidden nhưng còn base/copy anchor |
| Copy-mass-preserving cap | Main | Bảo vệ `pi_copy=g`; semantic cap phụ thuộc g là trade-off có chủ ý |
| Independent capped simplex | Control F | Đo lợi ích/rủi ro khi copy mass được thay đổi; không gọi là main |
| `Wo=tiny_rms_1e-3` | Main candidate E | Đánh thức Q/K/V sớm; calibration tái lập |
| `Wo=0` | D parity/smoke | Endpoint đại số, không phải recipe train chính |
| Inner gate `.05` | Bỏ main | Tránh gate kép và giảm đường gradient ẩn |
| Partition heads 4×32/4×128 | Hoãn | Nhiều head chưa bảo đảm coverage; cost/redundancy |
| Planner/coverage/region hierarchy | Bỏ main | Chưa có chẩn đoán causal; dễ làm giảm R1/L và trộn nhiều biến |
| Contrastive SDACL, R-Drop, NEFTune, self-improve | Hoãn vòng sau | Architecture CE-only phải vượt trước; không dùng objective để che regression |
| LongT5/CoLT5/MLA/MoE/MoD/AttnRes | Hoãn | Nguồn tham khảo cho bottleneck context/scale, không phù hợp thay nhiều biến ở PubMed 1B |

## 9. Liên hệ nghiên cứu và giới hạn chuyển giao

- [Mixture of Softmaxes](https://arxiv.org/html/1711.03953v3) là tiền lệ cho việc trộn nhiều vocabulary distributions; nó cũng nhắc rằng output compute tăng và mixture dư có thể overfit. V5 chỉ dùng hai readout chung `W_lm`.
- [Pointer-Generator](https://aclanthology.org/P17-1099/) và [Generalized Pointer Generator](https://aclanthology.org/D19-1390/) hỗ trợ tách đường copy khỏi abstraction; v5 không claim copy head mới và không đưa coverage/target aligner vào main.
- [ReZero](https://arxiv.org/html/2003.04887) nghiên cứu residual scalar khởi tạo zero, còn [DeepNet](https://arxiv.org/html/2203.00555) nghiên cứu scaling/initialization cho Transformer rất sâu. Đây là động lực khái niệm, không chứng minh `Wo=tiny` thắng trên PubMed.
- [Training Dynamics for Summarization](https://arxiv.org/abs/2110.08370) trên CNN/DM, XSum và MediaSum cho thấy copy propensity và hallucination có thể thay đổi ở các giai đoạn khác nhau; vì vậy log branch diagnostics theo step thay vì suy ra từ một epoch loss, và không tổng quát hóa kết quả đó sang PubMed nếu chưa đo.
- [T5Gemma 2](https://arxiv.org/html/2512.14856v1) báo cross-attention mọi layer, tied embeddings, merged attention và adaptation UL2; v5 chỉ giữ source access trong trunk, không nhận rằng output mixture sẽ tự vượt model đã pretrain/adapt khác.
- [UL2](https://arxiv.org/abs/2205.05131) là mixture-of-denoisers cho pretraining; không thêm vào CE fine-tuning của v5 vì sẽ đổi objective và budget.
- [LongT5](https://arxiv.org/abs/2112.07916) và [CoLT5](https://arxiv.org/abs/2303.09752) giải quyết input dài/conditional compute; chỉ dùng nếu audit visibility chứng minh truncation là bottleneck.
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258), [DeepSeek-V3](https://arxiv.org/abs/2412.19437) và [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) là hướng foundation-model quy mô lớn; không chuyển thẳng vào model 1B vì khác scale, pretraining và mục tiêu.
- [SDACL PDF](/Users/kieugiangbien/Downloads/Paper/VAI/Summarization/SDACL.pdf) là cơ sở cho contrastive salience-distance ở vòng sau; không được dùng để mô tả v5 architecture claim.

Novelty hợp lệ chỉ có thể là decomposition cụ thể trong một summarizer ghép checkpoint: bounded source-conditioned semantic correction, copy responsibility giữ riêng ở output interface, và một bộ controls chứng minh trade-off. Probability mixture, pointer copy, sigmoid routing và semantic attention riêng lẻ đều có prior art.

## 10. Trình tự thực hiện không được đảo

1. Sửa docs/config schema và tạo fingerprint trước khi tạo model module.
2. Implement A/B endpoints; chạy C1–C3 và rerun common baselines.
3. Implement dense FP32 oracle + D zero parity; chạy C4–C8.
4. Implement E tiny calibration, independent F và API K/V tách biệt; chạy C5/C6/C17/C18.
5. Implement DDP manifest, global schedule, checkpoint selection, stale cache guard; chạy C9/C13/C15/C16/C19.
6. Chạy cache/generation/gauge và profile dense/chunked; chạy C10–C14.
7. Pilot validation một seed; chỉ promotion candidate khi smoke + gates pass.
8. Full A/B/E/F tối thiểu ba seed; test một lần bằng ROUGE-1.5.5, paired bootstrap và provenance đầy đủ.
9. Chỉ sau khi architecture gate pass mới mở SDACL/contrastive hoặc factuality/hallucination study.

## 11. Câu hỏi tự kiểm tra theo deepread

1. Vì sao `pi_copy=g`, base floor dương và semantic mass dương không thể cùng bảo đảm khi `g` gần 1? Main đã chọn ưu tiên điều nào?
2. Tại sao `l_copy=log(g)` phù hợp với `l_base=log(1-g)` hơn `logit(g)` trong softmax ba nhánh?
3. Với `Wo=0`, gradient nào bằng zero ở backward đầu và gradient nào vẫn hữu hạn? Điều này ảnh hưởng cách thiết kế tiny init thế nào?
4. Vì sao `Pcopy` phải marginalize duplicate token IDs trước mixture và `lc=-inf` cho target không copyable?
5. Vì sao chỉ đổi config `key_source=M` là chưa đủ để triển khai `K=M,V=H0`?
6. Công thức `loss_r=world_size*L_r/N_global` triệt tiêu DDP average như thế nào? Tại sao không được chia thêm accumulation?
7. Vì sao `grad_norm=2.0` vẫn có thể xuất hiện khi `max_grad_norm=1.0`?
8. Tại sao temperature/top-k/top-p phải áp sau final mixture và không được dùng trong greedy headline?
9. Một gain ROUGE duy nhất có đủ để claim giảm hallucination không? Diagnostics nào cần thêm?
10. Nếu F tăng R1 nhưng giảm R2, vì sao giữ E là quyết định đúng với target hiện tại?

## 12. Giới hạn và điều kiện bác bỏ

Audit này không tạo ra bằng chứng rằng AFMR sẽ thắng. Snapshot ban đầu không có implementation/checkpoint/predictions; package hiện đã được dựng và có acceptance evidence trong `IMPLEMENTATION_EVIDENCE.md`, còn baseline lịch sử vẫn thiếu resolved config/predictions đầy đủ. Nếu C1/C2/C3 hoặc fixed-small-set fail, dừng train PubMed và sửa implementation. Nếu E không vượt common A/B sau ba seed, không thêm complexity để cứu claim; báo negative result và chuyển sang chẩn đoán source visibility, length, copy calibration hoặc objective ở vòng riêng.
