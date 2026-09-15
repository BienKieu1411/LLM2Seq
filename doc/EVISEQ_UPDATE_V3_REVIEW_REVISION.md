# EviSeq v3: sửa kiến trúc theo Luna review

Ngày: 2026-09-09. Phạm vi: `src/eviseq_update_v3`.

Bản sửa xử lý hai hạn chế có probe tái hiện trong
[Luna review](EVISEQ_UPDATE_V3_LUNA_REVIEW.md): hard partition làm mất lựa chọn
nguồn của từng head, và context normalization làm mất phần lớn tác dụng giảm
đồng loạt head gain. Đồng thời sửa evaluator để predictions không bị gán nhầm
cho checkpoint khác. Chưa có kết quả training PubMed của bản sửa này.

## 1. Hai thay đổi kiến trúc

| Thành phần | Trước review | Sau review, mặc định mới |
|---|---|---|
| Semantic matching | 4 heads × 128 | Giữ 4 heads × 128, tổng rank512 |
| Vùng mỗi head được đọc | `region_id % 4 == head_id` | Mọi vùng nguồn hợp lệ |
| Head gain | Nhân context rồi RMS-normalize context | RMS-normalize context rồi nhân gain |
| Global semantic gate | Đọc query và context đã bị head gain tác động | Đọc query và context chuẩn hóa trước head gain |
| Region planner | Coverage/continuity học từ prefix bằng CE | Giữ nguyên công thức và causal scan |
| Fusion | Tangent, relative cap0.10, giữ norm | Giữ nguyên |
| Copy/backbone | Đường v2, copy nhận hidden trước fusion | Giữ nguyên |
| Objective | Gold-token mixture CE | Giữ nguyên; không thêm loss |

### 1.1. Mọi head được đọc mọi vùng nguồn

Với mỗi head h và timestep t, region attention vẫn có dạng:

```text
π[t,h,r] = softmax_r(q[t,h] · k_region[r,h] / sqrt(128)
                    + source_prior[r] + planner_bias[t,r])
c[t,h]   = Σ_r π[t,h,r] Σ_(s in r) a[t,h,s|r] v[s,h]
```

Chỉ bỏ mask sở hữu vùng modulo ở công thức đầu. Mask source/padding, chuẩn hóa
token trong từng vùng và đường gradient còn nguyên. Các head vẫn có projection
khác nhau và tính matching trong không gian128 chiều riêng.

Lý do: nếu nguồn có hai vùng, hard partition khiến hai head chỉ có đúng một
lựa chọn và hai head không có vùng. Region softmax của head có một lựa chọn luôn
bằng1; cộng coverage hay continuity không thay đổi context của head đó. Đây là
hạn chế biểu diễn, không phải gradient bị detach. Trường hợp nguồn dài hơn cũng
có rủi ro thông tin quan trọng nằm ngoài tập vùng được cấp cho một head.

Sau sửa, mọi head có thể so sánh hai vùng trên nguồn128 tokens/region64. Tests
can thiệp values của vùng đầu thấy cả bốn head thay đổi context; tăng coverage
của vùng đó giảm trọng số của nó, tăng recent-state thì tăng trọng số.

**Đánh đổi:** bỏ đảm bảo support rời nhau của hard partition. Các head có thể
học attention trùng nhau; không claim đã giải quyết head collapse hoặc fact
coverage. Với chỉ một vùng hợp lệ, region selection vẫn không có lựa chọn;
token attention trong vùng vẫn hoạt động. Block64 tokens vẫn có thể cắt câu.

### 1.2. Head gate hoạt động sau context normalization

Gọi N là RMS normalization trên context ghép của các head, g là gain mỗi head,
và G(g) là gain được lặp theo128 chiều/head. Công thức cũ:

```text
g = 2 sigmoid(W_head N(hidden) + b_head)
u_old = N(G(g) ⊙ c)
δ_old = sigmoid(W_gate [q; u_old] + b_gate) · W_out u_old
```

Nếu giảm đồng loạt g thành0.1g, RMS normalization có thể đưa context gần về
magnitude cũ. Review đo correction norm ratio khoảng0.999 dù gain giảm10 lần.

Công thức mới:

```text
u = N(c)
g = 2 sigmoid(W_head N(hidden) + b_head)
α = sigmoid(W_gate [q; u] + b_gate)
δ = α · W_out(G(g) ⊙ u)
h_generation = norm_preserving_fuse(hidden, δ, relative_cap=0.10)
```

Không normalize lại context sau nhân gain. Global gate đọc evidence trước gain,
để một phép giảm gain không đồng thời đổi global gate và bù lại tác dụng giảm.
`W_out` không có bias. Với hidden/context/global gate cố định, scale đồng loạt
gain sẽ scale δ tương ứng trước fusion. Khi correction nhỏ và chưa chạm cap,
test giảm gain10 lần thấy correction sau fusion giảm xấp xỉ10 lần. Khi cap đã
bão hòa, không claim tỷ lệ này vẫn đúng sau fusion.

Gain khởi tạo bằng1 và `W_out` khởi tạo zero như cũ. Ban đầu nhánh semantic chưa
đổi hidden; CE mở `W_out` trước, sau đó các parameter bên trong nhận gradient
khác0. Đây là lịch mở nhánh do zero-init, không phải lỗi gradient. Tests kiểm
tra hai bước optimizer ở cả interface warmup và full finetune, gồm rank512.

## 2. Phần giữ từ v2 và v3 trước review

- AFMR với final-state value anchor; cross-attention trên backbone và đường
  copy giữ nguyên. Mặc định `query_cross_gate=false`.
- Copy attention, gate và cộng xác suất các vị trí cùng token ID vẫn dùng
  hidden trước semantic fusion. Với cùng weights/hidden/source state, sửa
  nhánh semantic không làm đổi copy distribution trong forward đó.
- Prefix planner không đọc target tương lai, bỏ prompt/padding khỏi usage;
  dense CE, chunk CE và cached generation dùng cùng quy ước prefix.
- Fusion bỏ phần radial của correction, chặn tangent rồi đưa hidden về norm
  cũ; cap0.10 giữ nguyên. Thứ tự logits vẫn có thể thay đổi.
- Chỉ gold-token mixture CE. Không contrastive, R-Drop, NEFTune, teacher,
  self-improve hoặc sinh candidates trong training. Temperature/top-p vẫn ở
  API sampling ngoài training; benchmark greedy.
- FP32 parameter/gradient/AdamW, BF16 autocast khi train CUDA; clip1.0 theo
  PubMed recipe. Không đổi LR, stages hay objective để tránh lẫn nguyên nhân.

Các shared weights vẫn được CE cập nhật, nên copy probability sau training và
ROUGE-2 vẫn có thể đổi. Mục tiêu là gỡ hạn chế đọc evidence và kiểm soát semantic
correction; chưa thể nói giữ được mức R2 hay tăng R1/L trên PubMed.

Hai thay đổi không thêm tham số, không tăng số forward, giữ nguyên shape
attention và semantic cache của v3 trước review. Không suy ra throughput CUDA
từ điều đó. Chi phí4×128 so với v2 1×128 vẫn tồn tại; eval CUDA hiện giữ semantic
K/V FP32, riêng hai tensor đó ở batch64/source4096/rank512 là khoảng1GiB.

## 3. Sửa tính hợp lệ của eval

Trước đây evaluator chỉ kiểm tra ID/reference prefix. File hoàn tất được trả
metrics trước khi đọc checkpoint; file dở có thể trộn prediction từ hai model.

Evaluator mới tạo `predictions.jsonl.manifest.json` trước khi sinh. Identity
gồm SHA256 nội dung checkpoint, toàn bộ JSONL split đang eval, cấu hình ảnh
hưởng suy luận, tokenizer và prompt IDs, implementation Python của package,
version torch/transformers/tokenizers và loại device. Hash streaming tránh
đọc cả checkpoint thành một buffer RAM.

Kiểm tra manifest diễn ra trước cả nhánh trả metrics cho file đã hoàn tất.
Checkpoint không tồn tại báo lỗi; đổi weights dù giữ filename, source dù giữ
ID/reference, config hoặc tokenizer đều không được dùng chung predictions.
File predictions có nội dung nhưng thiếu manifest sẽ bị từ chối resume; dùng
output mới để eval lại. Không tự xóa hoặc ghi đè predictions cũ.

Resume đúng run vẫn cho đổi batch size hoặc mở rộng `max_examples`, vì đây là
tiến độ và cách chia batch. Batch/kernel/thiết bị vật lý có thể gây khác biệt
số học nhỏ; manifest không phải chứng nhận bitwise reproducibility. File JSONL
được hash theo bytes nên đổi format dù giữ dữ liệu cũng yêu cầu output mới.
Implementation/library fingerprint cố ý bảo thủ: sửa package hoặc nâng thư
viện có thể yêu cầu eval lại. Manifest không chống chỉnh sửa thủ công file.

Hash chỉ chạy một lần đầu mỗi lệnh eval; không hash mỗi batch, không thêm vào
training. Trên network storage, đọc checkpoint lớn có thể tăng thời gian khởi
động eval. Không đổi công thức Python ROUGE hay ROUGE155 wrapper.

## 4. Config, checkpoint và ablation

Recipe mới đã áp dụng trong `configs/afmr_base.yaml` và `run_pubmed_pair.sh`:

```yaml
semantic_read:
  rank: 512
  num_heads: 4
  attention: hierarchical_coverage
  head_gate_position: post_norm
  fusion: norm_preserving
  max_relative_rms: 0.10
  planner:
    region_size: 64
    partition_heads: false
    use_coverage: true
    use_continuity: true
```

Tên run chứa gate position và partition để không dùng nhầm output mặc định cũ.
Nếu tự đặt `RUN_ROOT`, người chạy cần dùng thư mục riêng cho từng ablation.
Checkpoint spec chặn load giữa hai graph khác gate position/partition dù
shape bằng nhau. Thiếu field mới trong resolved config cũ tương đương
`pre_norm`; checkpoint cũ vẫn eval bằng đúng graph cũ. Không coi train tiếp
checkpoint cũ là đối chứng train-from-pretrained của graph mới.

| Run | Biến môi trường thêm vào script |
|---|---|
| A: v3 trước review | `PARTITION_HEADS=true SEMANTIC_HEAD_GATE_POSITION=pre_norm` |
| B: chỉ bỏ hard partition | `PARTITION_HEADS=false SEMANTIC_HEAD_GATE_POSITION=pre_norm` |
| C: chỉ sửa gate | `PARTITION_HEADS=true SEMANTIC_HEAD_GATE_POSITION=post_norm` |
| D: cả hai sửa đổi | Mặc định: `PARTITION_HEADS=false SEMANTIC_HEAD_GATE_POSITION=post_norm` |

Giữ pretrained weights, data, seed, global batch, số update/stages, LR, clip,
source/target limits, decoding và ROUGE155 flags giống nhau cho A–D. Dùng
validation để chọn, rồi chốt cấu hình trước test. Không lấy độ lệch nhỏ ở một
seed làm bằng chứng chắc chắn vượt model khác.

Từ repo root, chạy PPLX trên2GPU, chọn bằng validation:

```bash
RUN_ENCODERS=pplx EVAL_SPLIT=validation bash src/eviseq_update_v3/scripts/run_pubmed_pair.sh
```

Mặc định48/GPU ×2GPU ×accum1 = global96;1 warmup +3 full epochs. Để dùng84/GPU,
đặt `BATCH_SIZE=84 GRADIENT_ACCUMULATION_STEPS=1` và giữ global168 cho đối chứng.
Đường dẫn model/data và `ROUGE155_SCRIPT` cần trỏ đúng tài nguyên máy chạy.

## 5. Kiểm chứng và hạn chế còn lại

Kiểm tra local ngày2026-09-09, Python trong `bienkieu_env`:

- **318 tests passed** trong71.25s, gồm DDP2processes Gloo và đối chiếu với
  một process; hai cảnh báo deprecation SWIG không làm fail tests.
- Smoke train → save → load → eval báo `status=ok`, sinh2examples; đúng
  `partition_heads=false`, `head_gate_position=post_norm`,4heads. Tiny loss
  từ4.490708 xuống4.476885; con số này chỉ xác nhận integration smoke, không
  đại diện cho PubMed hoặc ROUGE.
- Ruff0.15.20 lint toàn repo passed; formatter check báo276files đã đúng format.
- `bash -n scripts/run_pubmed_pair.sh` và `git diff --check` passed.

Phạm vi kiểm tra gồm: oracle forward/backward hierarchical attention cho cả
hai routing modes; source ngắn1–4 vùng; tác dụng giảm gain; zero-init và nguồn
rỗng; dense/chunk loss và gradient FP32/BF16; planner cập nhật qua optimizer;
cached/dense/compaction và prompt lengths khác nhau; copy/trunk invariants;
checkpoint compatibility; shell-generated configs; eval resume; DDP2processes.

Chưa chạy full PubMed, nhiều seed, ROUGE155 thực tế hoặc NCCL/CUDA trên máy này.
Các sửa đổi này không tự tạo bằng chứng novelty học thuật, giảm hallucination
hay vượt T5Gemma. Cơ sở phương pháp kế thừa được ghi trong
[report coverage](EVISEQ_UPDATE_V3_COVERAGE_REVISION.md); lần sửa này xuất phát
từ kiểm tra công thức và regression, không thêm một phương pháp training mới.

Hai edge cases API của review chưa thuộc lần sửa kiến trúc này: caller tự tái
sử dụng cross-cache nhưng đổi memory mà không reset; repetition penalty với
non-chat prompt dài ngắn khác nhau và pad trùng EOS. Runner PubMed chuẩn dùng
cache lifecycle và chat prompt hiện hành; không diễn giải việc pass suite
thành đã sửa mọi API edge case được nêu trong review.
