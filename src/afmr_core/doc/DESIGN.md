# Đặc tả kiến trúc AFMR

## 1. Phương án main

V5.1 giữ một đường dự đoán vocabulary trực tiếp của `new`, semantic read độc lập của `update_v2` và copy head hiện hữu. Semantic branch không còn nằm trong nested gate `(1-g)beta`.

    P0 = softmax(z0),       z0 = W_lm h
    Ps = softmax(zs),       zs = W_lm (h + delta)
    Pcopy = marginalized copy distribution

    alpha_raw = alpha_max * sigmoid(r_sem) * evidence
    g_route = stop_gradient(g)
    alpha = min(alpha_raw, max(0, 1 - g_route - generate_reserve))

    pi_copy = g
    pi_sem  = alpha
    pi_base = 1 - g - alpha
    P = pi_base P0 + pi_sem Ps + pi_copy Pcopy

`g` là copy gate legacy và bị ép bằng zero khi copy mask rỗng. `g_route` chỉ là
bản detached dùng cho router và cap semantic; `pi_copy` vẫn dùng `g` có
gradient; direct probability Jacobian theo `g` là `Pcopy-P0`, còn `dL/dg` thực
tế phải log riêng vì phân phối semantic đã thay đổi.
`generate_reserve` là phần generate được ưu tiên giữ lại khi còn đủ mass; nó
không phải floor tuyệt đối của `pi_base` khi `g > 1-generate_reserve`. Main
được gọi chính xác là **copy-mass-preserving capped simplex**: `pi_copy` giữ
giá trị `g` ở interface, còn semantic bị cap theo `g_route` đã detach. Ba điều
kiện `pi_copy=g`, `pi_base>=floor>0` và `pi_sem>0` không thể cùng đúng khi `g`
gần một; control `independent_capped_simplex` bên dưới dùng trade-off khác.
`evidence` main là nhị phân: `content_mask.any(dim=-1)[:, None, None].float()` có
shape `[B,1,1]`, broadcast thành `[B,T,1]`; không lấy từ copy mask `[B,W]`.
Chỉ null-slot variant mới dùng evidence liên tục `1-A_null`. Khi cả hai source
path rỗng, output bắt buộc là `P0`.

Nếu `Wo=0`, `delta=0`, nên `Ps=P0`; vì `pi_base+pi_sem=1-g`, forward đúng
bằng `new` ở cùng weights. C4 phải đối chiếu trực tiếp với kernel
`eviseq_new._mix_logits`, không dùng một oracle ba phép cộng khác; bitwise
identity không được giả định chỉ từ đại số. Đây là identity endpoint, không phải
frozen anchor sau training.

## 2. Trunk và semantic reader

`H0=bridge.value_memory` là value anchor; `M=bridge.memory` là AFMR retrieval
memory; `b` là source prior. Trunk giữ native encoder, AFMR và cross-attention ở
mọi decoder layer của `new/v2`. V5 phải yêu cầu `bridge.value_memory` tồn tại và
raise lỗi nếu state này là `None`; tuyệt đối không fallback ngầm sang `M`, vì
điều đó làm mất parity của copy và semantic reader với `new/v2`.

Semantic reader main:

    q   = Wq RMS(h)
    H0n = RMS(H0)
    Ks  = RMS(Wk(H0n))
    Vs  = Wv(H0n)
    A  = masked_softmax(q Ks^T / sqrt(r) + semantic_prior_scale * b)
    c  = A Vs
    u  = RMS(c)
    d0 = Wo u
    delta = smooth_relative_RMS_cap(d0, h, rho=0.10)
    hs = h + delta

Rank là `r=128`, một head. Không có hierarchy, region mass, planner, continuity,
hard top-k hoặc norm-restoring tangent projection. `semantic_prior_scale=1.0`
cho main; variant `AFMR-KM` phải ghi rõ scale (mặc định `0.0`) để không đếm
source prior hai lần. Nếu thử `r=256`, denominator là `sqrt(r)`, không cố định
`sqrt(128)`.

`K=M,V=H0` là variant `AFMR-KM`. Khi chạy variant này, giảm hoặc tắt source prior trong semantic score để đo nguy cơ double-count. Không dùng `M` làm value ở vòng đầu vì nó có thể làm AFMR drift tác động trực tiếp vào content và copy.

Semantic reader main không có scalar gate `.05` riêng. RMS cap `.10` là giới hạn
residual duy nhất. Cap phải được triển khai bằng công thức tái lập, không dùng
tên hàm mơ hồ:

```text
RMS(x) = sqrt(mean_hidden(x^2))                         # tính ở FP32
t      = max(rho * RMS(h), eps)
r      = RMS(d0)
x      = r / t
s      = tanh(x) / max(x, eps)                          # lim_{x→0} s = 1
delta  = s * d0
```

`eps=1e-6` và nhánh `r=0` phải cho `delta=0`. Công thức này cho
`RMS(delta) <= t` (sai số số học trong tolerance), hữu hạn tại residual zero và
có đạo hàm trơn ở vùng hoạt động; C5/C8 phải kiểm tra bound, finite gradient và
độ khớp dense/chunked. Null slot không nằm trong main; nếu cần, dùng value zero và evidence mass:

Control `legacy_semantic` phải bật lại đúng `semantic_gate_init=0.05` của v2
để C2 là endpoint thật; `inner_gate=false` chỉ áp dụng cho main AFMR. Trong
implementation, control này dùng `readout_mode=legacy_copy_mixture`,
`semantic_read.cap_mode=legacy`, `inner_gate=true` và `output_init=zero`;
cap legacy giữ đúng công thức v2, còn main dùng cap trơn ở trên.

Endpoint này phải tái tạo đúng graph v2, không chỉ đặt một flag tên tương tự:

```text
delta_v2 = sigmoid(gate_v2([q_v2, RMS(c_v2)])) * W_out_v2(RMS(c_v2))
P_semantic_v2 = softmax(W_lm(h + delta_v2))
P_v2 = (1-g) * P_semantic_v2 + g * Pcopy
```

Trong control, `g` và copy state phải lấy từ cùng legacy head, source path và
mask như v2; nếu v2 có `max_relative_rms` thì giữ đúng cap đó. C2 chỉ pass khi
logits/loss của control khớp implementation v2 trong tolerance đã đăng ký.

    K = [K_source, k_null]
    V = [V_source, 0]
    evidence = 1 - A_null
    delta = evidence * bounded_residual(...)

Không dùng RMS context một mình để đại diện evidence, vì RMS có thể xóa độ lớn của tín hiệu yếu.

## 3. Router và features

Router là linear projection trên các feature chỉ phụ thuộc source và observed prefix:

    [q, u, stop_gradient(g_raw_logit),
     stop_gradient(copy_entropy_norm)]

`g_raw_logit` là output tuyến tính trước sigmoid của legacy copy gate; không dùng
`logit(g)` khi source rỗng vì `g=0` tạo `-inf`. Router trả `r_sem`; không nhận
label, target/reference embedding, ROUGE, future prefix hay generated
candidates. Router không được tự chạy encoder/decoder.

Main defaults:

    alpha_max = 0.20
    generate_reserve = 0.05
    alpha_init ≈ 0.05
    hard_source_fallback = true

`alpha` là mixture mass, không phải factuality probability. Gating độc lập không có nghĩa semantic branch luôn đúng; diagnostics phải kiểm tra responsibility theo token và theo nguồn.

`copy_entropy_norm` được định nghĩa là entropy của attention trên các candidate
hợp lệ, chia cho `log(max(2, n_valid))`, rồi detach. Không dùng
`copy_support_mass`: tổng attention trên support hợp lệ thường bằng một và
không mang thông tin routing.

Một control tách biệt là **independent capped simplex**:

    l_base = log(clamp(1-g_route, eps, 1)) + r_base
    l_copy = log(clamp(g_route, eps, 1)) + r_copy
    l_sem  = log(eps_sem) + r_sem
    q = softmax(l_base, l_sem, l_copy)
    pi_base = base_floor + (1-base_floor) q_base
    pi_sem  = (1-base_floor) q_sem
    pi_copy = (1-base_floor) q_copy

Control này dùng `g_route` đã detached làm input và mask expert. `r_base`,
`r_sem`, `r_copy` là router residual khởi tạo zero; expert rỗng được mask bằng
`-inf`. `base_floor` chỉ thuộc control này và là floor thật. Không dùng
simplex tự do không floor. Dùng `log(g)` cho copy logit vì softmax của
`[log(1-g), log(g)]` khôi phục tỷ lệ legacy; `logit(g)` sẽ không khôi phục tỷ
lệ đó khi có nhánh semantic.

## 4. Initialization và gradient

Main candidate dùng `Wo=tiny_rms_1e-3`; `Wo=0` chỉ là endpoint/parity control.
Khi `delta=0`, `Ps=P0`, vì vậy
`∂L/∂alpha=0` và router không nhận gradient từ expert difference; Q/K/V cũng
chưa nhận gradient qua `u` vì `∂delta/∂u=Wo=0`. Ngược lại `Wo` vẫn nhận
gradient hữu hạn qua `zs=W_lm(h+Wo u)` trên batch nondegenerate. C5 không được
yêu cầu Q/K/V có gradient ở backward đầu tiên; chỉ kiểm tra điều đó sau
optimizer update.

Tiny được tạo bằng `Wo_raw` với seed cố định trên calibration batch không dùng
label, sau đó rescale để `RMS(Wo u)/RMS(h)=1e-3` (cho phép `[0.5e-3,2e-3]`).
Phép cộng `h+delta` thực hiện ở FP32 trước khi cast. Zero chỉ dùng cho parity và
smoke ngắn; tiny là candidate training đăng ký trước. Không suy luận tiny có
gradient mạnh hơn v2: v2 cũng có `W_out=0` và inner gate khoảng `0.05`, nên phải
đo theo optimizer step.

Q/K/V, AFMR, decoder và copy đều được phép nhận gradient qua shared trunk trong
training. `g_route` và các copy diagnostics đưa vào router đều detached; nhờ đó
semantic routing không tạo thêm đường trực tiếp vào copy gate. Shared trunk vẫn
có thể làm thay đổi copy qua các bước optimizer, nên claim chỉ là copy-mass
preserving ở output, không phải parameter isolation.

## 5. Mixture NLL

Training chỉ dùng gold CE trên phân phối cuối:

    l0 = z0[y] - logsumexp(z0)
    ls = zs[y] - logsumexp(zs)
    lc = logsumexp(copy_logprob[j] với token_id[j]=y)
    lp = logsumexp(log(pi_base)+l0,
                   log(pi_sem)+ls,
                   log(pi_copy)+lc)
    loss_sum = -sum(lp tại labels != -100)

Nếu y không copyable, `lc=-inf`; hai vocabulary routes vẫn được tính. Không dùng `pi_base*CE0 + pi_sem*CEs + pi_copy*CEcopy`, vì đó là objective khác.

Tính logits/normalizer/mixture ở FP32. Matmul có thể BF16 theo common recipe.
Dense và chunked dùng chung `_mixture_nll_from_targets`; chunked chỉ giữ route
state và target logits lazy, nên không tạo thêm objective hay nhánh routing.
Trong một logical accumulation window, đặt
`N_global=all_reduce(sum_r valid_target_tokens_r)` và dùng
`loss_r=world_size*local_loss_sum_r/N_global`; DDP average đúng một lần. Không
chia thêm theo accumulation hoặc world size. Clip được thực hiện sau
accumulate/synchronize và trước optimizer step; log norm trước/sau clip và hệ
số clip.

## 6. Output logits và decoding

Đặt:

    Z0 = logsumexp(z0)
    output_logits = log(P) + Z0

Cộng scalar `Z0` giúp `P=P0` khôi phục logits legacy trước processor. Offset cũ `(1-beta)Z0+beta Zs` chỉ là diagnostic. Vì repetition penalty phụ thuộc dấu và độ lớn logits, mọi endpoint phải kiểm tra cả trước và sau processors.

Greedy benchmark giữ generation settings hiện hành. Temperature và top-p chỉ áp lên `output_logits` cuối cùng trong sampled generation/candidate API; không sample từng expert rồi mới mix.

## 7. Shapes và chi phí

Với batch `B`, native source length `S`, decoder length `T`, hidden `D`, vocabulary `V` và rank `r=128`:

    H0, M: B×S×D
    copy ids/mask/bias: B×W
    h, hs: B×T×D
    q, u: B×T×r
    Ks, Vs: B×S×r
    z0, zs: B×T×V
    alpha, g: B×T×1

`W` là chiều rộng candidate copy sau alignment; không giả định `W=S`.

Hai vocabulary readouts làm tăng memory/compute output head gần gấp đôi khi dense. Chunked vocabulary được phép nhưng phải so với dense oracle; không gọi module 524k tham số là miễn phí. Profile train/eval VRAM và throughput cùng với ROUGE.

## 8. Cache và mask

Semantic K/V, copy state, source IDs/masks/bias và cross K/V có thể chuẩn bị một lần/source. `h`, `q`, `u`, `alpha` tính lại theo prefix. Cache không được chứa beta/query cũ hoặc detached training graph.

Native mask rỗng phải trả context zero an toàn; không softmax một hàng toàn `-inf`. Reorder/compaction phải index-select đồng thời self KV, cross KV, semantic K/V, copy IDs/masks và source prior. Cache reset khi source, weights, dtype/device hoặc alignment version đổi.

## 9. Các điều không được gọi là đóng góp

Probability mixture, pointer copy, sigmoid gate, zero-init và semantic attention đều có prior art. Đóng góp có thể nghiên cứu chỉ là decomposition cụ thể: bounded semantic correction cạnh tranh với base vocabulary trong khi copy responsibility được giữ riêng, cùng phân tích source-conditioned trade-off. Cần ablation với hidden interpolation và constant alpha để chứng minh mixture có giá trị riêng.
