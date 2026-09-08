# EviSeq Update v2: semantic read riêng và residual có giới hạn

Package `eviseq_update` phát triển từ `src/eviseq_new` tại commit `a1ce056`.
Distribution/console command là `eviseq-update`; scripts dùng package trong thư mục
này. Output mặc định nằm trong `runs/eviseq_update/` của project này.

V2 dùng **attention riêng trên các vị trí encoder** để cấp context cho LM và giới
hạn RMS của residual bổ sung. Nhánh copy vẫn dùng alignment theo ký tự. Encoder,
AFMR value-anchor và cross-attention từng decoder layer giữ nguyên công thức.
Bản v1 dùng chung copy attention vẫn có thể chạy để đối chứng.

Đây là ứng viên sửa kiến trúc sau kết quả update thấp hơn new; **chưa có ROUGE của
v2**. Hai run đã báo còn khác batch, số bước cập nhật và clipping. Xem dữ kiện,
giới hạn kết luận và kế hoạch đối chứng trong
[báo cáo regression](../../Technical_Report/EVISEQ_UPDATE_REGRESSION_2026_09_08.md).
Ý tưởng v1 trước đó nằm trong [báo cáo nghiên cứu](../../Technical_Report/AFMR_FOUNDATION_WORLD_MODEL_NEXT_STEPS.md).

Recipe PubMed dùng **CE trên gold reference**, LR warmup + cosine. Training
chỉ dùng source/reference, một lượt teacher forcing mỗi microbatch.
Cơ sở paper, công thức, chi phí và cách kiểm
chứng claim nằm trong [báo cáo training](../../Technical_Report/EVISEQ_EVIDENCE_CONTRASTIVE_DESIGN.md).

## Computational graph

```text
encoder → final-state value memory H0 → native semantic keys / values
       ↘ AFMR memory/prior → cross-attention → decoder hidden h

h → semantic query → native source attention → context → bounded residual → h' → LM → P_LM
h → copy query → aligned contextual + lexical keys → scatter token IDs → P_copy

P(v) = (1-g_copy) P_LM(v) + g_copy P_copy(v)
loss = token CE của P
```

Decoder hidden đã nhận context qua cross-attention. Nhánh bổ sung đọc trực tiếp
memory encoder, không phụ thuộc cách tokenizer decoder chia source thành token copy:

```text
K = RMSNorm(W_key RMSNorm(H0)), V = W_value RMSNorm(H0)
q_t = W_query RMSNorm(h_t)
a_t = softmax(mask(q_t K^T / sqrt(rank) + source_prior))
c_t = a_t V
beta_t = sigmoid(W_beta [q_t ; RMSNorm(c_t)] + b_beta)
raw_t = beta_t W_out RMSNorm(c_t)
cap_t = rho * RMS(h_t), rho = 0.10
delta_t = raw_t * cap_t / sqrt(cap_t^2 + RMS(raw_t)^2 + 1e-12)
h'_t = h_t + delta_t
```

`RMS(delta_t) <= 0.10 * RMS(h_t)` trước khi làm tròn dtype. Đây là giới hạn residual
trong forward, khác với gradient clipping. `rho=0.10` là hyperparameter khởi điểm,
chưa được tối ưu bằng validation. Sigmoid gate đơn thuần không giới hạn độ lớn
projection đã học; v2 bổ sung ràng buộc này bằng phép tính trơn, không detach.

Query/key semantic không dùng chung với copy. CE cuối vẫn truyền qua cả hai nhánh,
backbones và source prior chung; không có cam kết loại bỏ mọi xung đột gradient.
V2 thêm một lượt attention, cỡ `O(B*T*S*rank)`, và cache native keys/values. Nó tăng
compute/memory so với v1; không gọi thêm encoder hay Transformer decoder.

`W_out=0` khi khởi tạo, beta bắt đầu ở 0.05. Với cùng seed, module chung và logits
ban đầu khớp copy-only. Backward đầu học W_out; sau đó mới mở gradient tới W_value,
W_key, W_query và beta. Module mới nằm trong optimizer group `cross_attention`,
được học cả trong interface warm-up và full fine-tuning. Không detach context.

Semantic attention chỉ đọc các vị trí nguồn nhìn thấy, bỏ prompt/padding bằng
content mask. Copy dùng sparse character-overlap alignment và cộng xác suất các
token ID trùng nhau. Nếu nguồn semantic rỗng, residual bằng 0. Nếu không có token
copy hợp lệ nhưng vẫn có nguồn semantic, head dùng LM đã được condition bằng nguồn.
Reference không được dùng để tạo source memory.

Ở hidden width 1024, copy key rank 128 và semantic rank 128, nhánh v2 có **524,545**
tham số (bốn projections và gate), tổng grounded head **918,018** tham số. V1 có
262,401 tham số semantic. RMSNorm không có tham số. Semantic keys/values có shape
`[B,S,rank]`; mask/prior có shape `[B,S]`. Cache được chọn lại khi loại hàng đã sinh xong.

## Config, precision và checkpoint

Các task recipes kế thừa lựa chọn v2 từ `configs/afmr_base.yaml`:

```yaml
decoder:
  grounded_copy:
    enabled: true
    key_dim: 128
    gate_init: 0.05
    semantic_read:
      enabled: true
      rank: 128
      gate_init: 0.05
      attention: independent_source
      max_relative_rms: 0.10
```

Copy-only: đặt `semantic_read.enabled: false`. Nếu tắt grounded copy thì cũng phải
tắt semantic read. Config thiếu `semantic_read` giữ graph copy cũ.

V1: đặt `attention: shared_copy`, `max_relative_rms: null`. Hai thuộc tính mới vắng
mặt trong **resolved config cũ** cũng giữ graph v1. Recipes kế thừa base hiện tại
chọn v2; phải ghi đè cả hai thuộc tính để chạy v1.

Checkpoint v2 lưu attention mode, semantic rank và residual cap trong
`architecture_spec`; đổi graph/rank/cap bị từ chối khi resume/evaluate. Checkpoint
còn lưu `training_spec`: batch, accumulation, số GPU, batch hiệu dụng, clipping và
các thiết lập training để đối chiếu các run sau. **V2 cần train từ pretrained
backbones; không resume checkpoint v1 vào v2.** Eval checkpoint cũ bằng resolved
config gốc. Sửa YAML không tạo ra trọng số đã học cho các projections mới.

Tham số, gradient và AdamW states giữ FP32; CUDA autocast mặc định BF16. Mixture
likelihood/CE được tổng hợp FP32. Dense logits và chunked loss đều dùng h' trước
LM head; chunked training chỉ tạo vocabulary logits tại vị trí có supervised labels.

Recipe PubMed dùng `training.max_grad_norm: 1.0` để khớp baseline đã báo. Base cho các
task khác vẫn để `null`. Norm ghi trong log là norm **trước clip**, sau đồng bộ
gradient; NaN/Inf khiến training dừng trước optimizer update. Vì vậy log `grad=2`
vẫn có thể đi cùng clip 1.0 đang hoạt động.

## Train và eval PubMed

Sửa đường dẫn model trong `configs/afmr_pubmed.yaml` trước khi dùng recipe trực tiếp.
Các split là JSONL có `id`, `text`, `summary`. Nên trỏ tới đúng dữ liệu đã chuẩn bị
của baseline. Config bundled resolve đường dẫn tương đối từ thư mục project;
config ngoài `configs/` resolve từ thư mục chứa chính config đó.

PubMed mặc định **1 epoch warm-up + 3 epoch full**, batch hiệu dụng **96**, clip **1.0**.
Một GPU: batch 48, accumulation 2. Hai GPU: batch 48/GPU, accumulation 1.
Training dùng embedding gốc và một lượt teacher forcing, không thêm input noise.
Mỗi stage có 5% bước LR warmup rồi cosine decay; các LR đỉnh từng nhóm giữ nguyên.

```bash
cd src/eviseq_update
# Một GPU
CUDA_VISIBLE_DEVICES=0 PYTHON=python3 \
  bash scripts/run_afmr.sh train configs/afmr_pubmed.yaml

# Hai GPU
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 GRADIENT_ACCUMULATION_STEPS=1 PYTHON=python3 \
  bash scripts/run_afmr.sh train configs/afmr_pubmed.yaml \
  --output-dir runs/eviseq_update/pubmed_copy_read_v2_cosine_2gpu
```

Script dùng `torchrun`/NCCL/DDP. Loss được cân theo **tổng supervised tokens trên mọi
rank và cả cửa sổ accumulation**. Chỉ đồng bộ backward cuối cửa sổ, rồi clip và update.
Batch cuối không bỏ/tính trùng mẫu; rank thiếu mẫu chạy placeholder có labels `-100`
và trọng số loss 0. Validation CE cũng tổng hợp theo token toàn cục.

Reducer DDP được tạo lại khi mở backbone sau warm-up. Chỉ rank 0 ghi metrics/config/
checkpoint. Checkpoint giữ RNG từng rank, weights không có tiền tố DDP và eval được
trên một GPU. Resume cùng số GPU, batch và accumulation để giữ lịch update; dùng
`--resume-checkpoint /path/to/v2/last.pt`. Mỗi GPU chứa đầy đủ model/optimizer;
DDP không gộp VRAM. Throughput/VRAM thực cần đo trên GPU.

Queue PPLX rồi Qwen3-Embedding dùng các đường dẫn server sẵn trong script:

```bash
bash scripts/run_pubmed_pair.sh
```

Mỗi run mặc định dùng hai GPU; train xong mới eval test bằng một GPU. Chỉ chạy PPLX
v2 rồi chấm thêm Perl ROUGE155, khi đã cài backend:

```bash
RUN_ENCODERS=pplx ROUGE155_SCRIPT="$PWD/../rouge155/evaluate_rouge.py" \
  bash scripts/run_pubmed_pair.sh
```

Các biến `PPLX_ENCODER`, `QWEN_ENCODER`, `DECODER_MODEL`, `PROCESSED_DATA_DIR` hoặc
`PUBMED_SOURCE_DIR` điều khiển đường dẫn model/data. `PYTHON`, `EVAL_BATCH_SIZE`,
`BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `MAX_GRAD_NORM` điều khiển tài nguyên/
training. Đặt `MAX_GRAD_NORM=null` nếu chủ động thử ablation không clip. Queue in batch
hiệu dụng, stage epochs và clipping, đồng thời lưu các giá trị vào config sinh ra.

Một GPU với batch hiệu dụng 96; queue tự chọn accumulation 2 nếu không override:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 RUN_ENCODERS=pplx \
  bash scripts/run_pubmed_pair.sh
```

`AFMR_SEMANTIC_VARIANT` chọn graph để đối chứng, cùng protocol mặc định:

| Giá trị | Attention | Residual cap |
|---|---|---|
| `shared_v1` | Copy attention dùng chung như update cũ | Không |
| `shared_bounded` | Copy attention dùng chung | 0.10 |
| `independent_unbounded` | Native source attention riêng | Không |
| `independent_bounded` (mặc định) | Native source attention riêng | 0.10 |

Ví dụ `RUN_ENCODERS=pplx AFMR_SEMANTIC_VARIANT=shared_v1 bash scripts/run_pubmed_pair.sh`.
`AFMR_SEMANTIC_READ=false` chạy copy-only. V2 ghi vào
`runs/eviseq_update/pubmed_pair_afmr_value_anchor_copy_read_independent_bounded_cosine/pplx`.
Các graph và training recipes có directory riêng; `RUN_ROOT`/`LOG_DIR` cho phép đổi nơi lưu.
Dùng directory mới khi đổi protocol/seed. Queue không tự ghi đè run có sẵn;
`OVERWRITE_OUTPUT_DIR=true` là yêu cầu reset run có chủ đích.

Eval epoch 3 trên test cho run recipe trực tiếp một GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python3 bash scripts/run_afmr.sh evaluate \
  runs/eviseq_update/pubmed_value_anchor_copy_read_v2_cosine/resolved_config.yaml \
  runs/eviseq_update/pubmed_value_anchor_copy_read_v2_cosine/epoch_003.pt \
  runs/eviseq_update/pubmed_value_anchor_copy_read_v2_cosine/epoch003_test_predictions.jsonl \
  --split test --batch-size 32
```

Runtime báo Python ROUGE 1.0.0. Muốn so Perl ROUGE 1.5.5, chấm predictions bằng cùng
wrapper/tokenization/flags của baseline. Queue chỉ chấm Perl nếu có `ROUGE155_SCRIPT`.
Chọn graph/hyperparameters/checkpoint bằng validation; test dùng cho đánh giá đã chốt.

## Training recipes và sampling

Chọn bằng `AFMR_TRAINING_RECIPE=... bash scripts/run_pubmed_pair.sh`:

| Recipe | LR | Full-stage regularization | Lượt model / microbatch | Batch × GPU × accum mặc định |
|---|---|---|---|---|
| `ce` | Linear, không LR warmup | Không | 1 | 48 × 2 × 1 |
| `cosine` (mặc định) | 5% warmup + cosine | Không | 1 | 48 × 2 × 1 |
| `dropout` | 5% warmup + cosine | Cross-attention dropout 0.1 | 1 | 48 × 2 × 1 |
| `evidence` | 5% warmup + cosine | Query–source contrastive on copy + semantic reads | 1 | 48 × 2 × 1 |

Log ghi `ce`, LR, gradient norm và throughput. Checkpoint lưu recipe và chặn
resume khi đổi scheduler hoặc LR warmup. NEFTune đã bị loại bỏ: config cũ có
`neftune_noise_alpha` không được chấp nhận; checkpoint đã train với alpha khác 0
không thể tiếp tục như một run CE tương đương. Bắt đầu run CE riêng để đối chứng.
Weights cũ vẫn cùng graph và có thể đánh giá bằng bản sao config bỏ trường đã xóa;
không sửa provenance/resolved config gốc. Config thiếu scheduler dùng CE + linear.

Recipe `evidence` chuẩn bị một sidecar JSONL trước khi train cho từng encoder. Với
mỗi từ/number xuất hiện ở nhiều vị trí source, cache giữ source context phù hợp làm
positive và context cùng từ nhưng khác nghĩa làm hard negative. Train vẫn chỉ có
một encoder pass và một decoder teacher-forcing pass: query/key từ copy read và
native semantic read nhận gradient trực tiếp qua set-based multi-positive InfoNCE.
Không có candidate generation, target encoder hay pseudo-reference. Cache kiểm tra
hash source/target token IDs, config/tokenizer fingerprint và checksum trước train;
đổi tokenizer, prefix, length hay mining config thì phải build lại.

```bash
cd src/eviseq_update
RUN_ENCODERS=pplx AFMR_TRAINING_RECIPE=evidence \
  bash scripts/run_pubmed_pair.sh
```

Queue tạo `runs/.../evidence/pplx/evidence.jsonl`, ghi manifest cùng thư mục, sau
đó mới `torchrun` train. Có thể chuẩn bị cache riêng để audit trước:

```bash
python scripts/prepare_evidence.py --config /path/to/resolved_config.yaml \
  --split train --output-dir /path/to/evidence/pplx
```

`--audit-size 200` vẫn build cache cho toàn split, đồng thời ghi 200 hàng đầu vào
`audit_examples.jsonl`; `summary.json` ghi coverage, số unit và các lý do bị lọc.

`AFMR_EVIDENCE_MODE=copy|semantic|both` chọn ablation; `both` là bản chính và lấy
trung bình hai nhánh. `AFMR_EVIDENCE_MAX_WEIGHT` mặc định `0.05`,
`AFMR_EVIDENCE_RAMP_RATIO` mặc định `0.10`; lambda chỉ tăng trong full fine-tune,
warm-up interface vẫn là CE. Metrics ghi `evidence_copy`, `evidence_semantic`,
`evidence_lambda` và `evidence_units` cạnh CE. Ví dụ đặt batch 84 mỗi GPU và
không accumulation: `BATCH_SIZE=84 GRADIENT_ACCUMULATION_STEPS=1`.

Temperature/top-p đã có **Python API riêng** cho sinh candidates khi cần:

```python
import torch
from eviseq_update.evaluation.generate import generate_sampled

# model, batch, tokenizer đã được nạp; batch ở cùng device với model.
generator = torch.Generator(device=next(model.parameters()).device).manual_seed(42)
texts, token_ids = generate_sampled(
    model, batch, tokenizer, temperature=0.7, top_p=0.9,
    generator=generator, max_new_tokens=512, min_new_tokens=32,
    repetition_penalty=1.05, no_repeat_ngram_size=3,
)
```

Training không gọi API này; CLI `evaluate` và queue vẫn greedy, không sampling.
Queue mặc định chấm `last.pt` trên test sau train, không chọn checkpoint bằng test.
Để chọn `best.pt`, chạy evaluate trên validation trước và chốt tiêu chí cho tất cả
baselines. `best.pt` được lưu theo validation CE, chưa phải best ROUGE/factuality.

## Xác minh offline

Sau khi cài dependencies trong `requirements.txt`:

```bash
PYTHON=python3 bash scripts/run_afmr.sh smoke
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -m pytest -q -p no:cacheprovider
```

Smoke mặc định CE + cosine, dùng tiny Qwen ngẫu nhiên, tắt mạng, bật copy và semantic read; chạy hai stages,
checkpoint round-trip, dense/chunked CE, greedy eval và prediction resume. Dữ liệu
nằm trong `tests/fixtures`, output trong temporary directory tự xóa. Kết quả phải có
`"status": "ok"`, `"semantic_read": true`. Fixture `afmr_smoke.yaml` tắt copy/semantic
để hỗ trợ tests LM-only; lệnh `smoke` chủ động bật cả hai. Đây không phải đo ROUGE thực.

Tests kiểm tra initialization parity, residual bound khi projection lớn/hidden bằng 0,
nhánh semantic không dùng lexical copy attention, thay đổi tokenization copy, padding/
nguồn rỗng, gradient từng tham số, cập nhật FP32 dưới BF16 autocast, cache compaction
và compatibility checkpoint. Test hai tiến trình CPU/Gloo so gradient trước optimizer
step và weights sau update với một tiến trình ở cả warm-up/full, bốn chế độ LM/copy/
v1/v2, target dài/ngắn, accumulation dư và rank không có labels. Nó cũng kiểm tra
AdamW resume, RNG từng rank và eval checkpoint DDP bằng một tiến trình.
Tests kiểm tra một lượt forward không sinh text, gradient BF16/FP32 và resume
hai tiến trình với dropout để xác minh RNG khi có phép tính ngẫu nhiên.

Test queue chạy config generator thật nhưng giả lập CUDA/train/eval; không chứng minh
queue đã train trên GPU ở máy này. Smoke model thực trên GPU, có directory riêng:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python3 \
  bash scripts/smoke_a100.sh configs/afmr_pubmed.yaml
```

Đọc phân cấp, long-source encoding, BRIO và world-model predictor vẫn là các hướng
nghiên cứu riêng; chưa ghép thêm vào lần sửa này.

Thiết kế objective contrastive theo cơ chế copy/read, các tiền lệ cần trích dẫn và
ablation dự kiến nằm trong
[đề xuất training](../../Technical_Report/EVISEQ_EVIDENCE_CONTRASTIVE_DESIGN.md).
Objective đó đang ở mức thiết kế; recipe thực thi hiện tại là CE.
