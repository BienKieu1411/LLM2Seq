# EviSeq Update: context nguồn cho cả LM và copy

Bản này phát triển từ `src/eviseq_new` tại commit `a1ce056`. Package Python riêng là
`eviseq_update`, distribution/console command là `eviseq-update`. Các scripts chạy
từ source tree dùng package trong thư mục này. Thư mục output mặc định là
`runs/eviseq_update/` bên trong project mới.

Thay đổi kiến trúc: dùng **cùng attention của grounded copy head** để đọc thêm
source values và đưa context vào LM qua một gated residual. Encoder, AFMR
value-anchor, cross-attention từng decoder layer và copy gate giữ nguyên công thức.
Đây là bản triển khai ứng viên ở mục 10 của
[báo cáo nghiên cứu](../../Technical_Report/AFMR_FOUNDATION_WORLD_MODEL_NEXT_STEPS.md).

## Computational graph

```text
encoder → final-state value memory H0 → semantic value projection → overlap pooling → U
       ↘ AFMR memory/prior → cross-attention → decoder hidden h

copy attention a(h, source keys, prior)
    ├── scatter theo source token IDs → P_copy
    └── a @ U → RMSNorm → gated residual → h' → LM head → P_LM

P(v) = (1-g_copy) P_LM(v) + g_copy P_copy(v)
loss = token CE của P
```

Decoder hidden đã được condition bởi cross-attention; nhánh mới thêm đường từ
lượt đọc của output copy head tới vocabulary distribution:

```text
U_j = overlap_pool_j(W_value RMSNorm(H0))
c_t = sum_j a_tj U_j
beta_t = sigmoid(W_beta [q_t ; RMSNorm(c_t)] + b_beta)
h'_t = h_t + beta_t W_out RMSNorm(c_t)
```

`beta_t` điều khiển residual; `g_copy` vẫn điều khiển mixture. Hai gate học độc lập.
Chỉ tính attention scores một lần cho mỗi forward của output head; không gọi thêm
encoder hoặc Transformer decoder. Attention-value multiplication và source values
vẫn tăng compute/memory, với phần đọc thêm cỡ `O(B*T*J*rank)`.

`W_out=0` khi khởi tạo, beta bắt đầu ở 0.05. Với cùng seed, các module chung và
logits ban đầu khớp bản copy-only. Backward đầu học W_out; các bước sau mới mở
gradient qua đường mới tới W_value, beta và attention. Không detach context hoặc
attention trong đường CE. Tất cả module mới thuộc optimizer group `cross_attention`,
được học ngay trong interface warm-up và tiếp tục trong full fine-tuning.

Source values dùng cùng sparse character-overlap alignment với copy keys; chỉ dùng
phần source nhìn thấy. Không dùng reference để tạo source memory. Duplicate token
IDs cộng xác suất copy như trước. Hàng không có source token hợp lệ dùng LM chính xác,
kể cả khi residual đã được học.

Ở hidden width 1024, copy key rank 128 và semantic rank 128, nhánh mới có **262,401**
tham số (hai projections và gate), tổng grounded head là **655,874** tham số. RMSNorm
của head không có tham số. Cached `CopyState.semantic_values` có shape `[B,J,rank]`;
được chọn lại cùng các tensor còn lại khi finished-row compaction.

## Config và checkpoint

Các task recipes kế thừa cấu hình bật nhánh mới từ `configs/afmr_base.yaml`:

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
```

Đối chứng copy-only: đặt `semantic_read.enabled: false` và dùng output directory
khác. Nếu tắt cả grounded copy thì cũng phải tắt semantic read. Config thiếu
`semantic_read` giữ graph copy cũ để hỗ trợ đối chứng.

Checkpoint lưu graph `shared_attention_residual_v1` và semantic rank trong
`architecture_spec`. Bật/tắt nhánh hoặc đổi rank không được resume/evaluate bằng
checkpoint không tương ứng. Bản mới cần train từ pretrained backbones; sửa YAML
của checkpoint cũ không tạo ra weights đã học cho nhánh mới. Copy-only checkpoint
cũ chỉ tương thích khi nhánh mới tắt và các thông số cấu trúc còn lại khớp.

Tham số, gradient và AdamW states trong training được giữ FP32. CUDA autocast mặc
định BF16; mixture likelihood/CE được tổng hợp FP32. Loss chia chunk và dense logits
đều dùng h' trước LM head. Training chia chunk chỉ tạo vocabulary logits tại các
vị trí có supervised labels. Dtype/batch size/source length giữ theo recipe gốc.
`training.max_grad_norm: null` tắt gradient clipping theo yêu cầu. Norm vẫn được
ghi sau khi đồng bộ gradient; NaN/Inf khiến training dừng trước optimizer update.
Đặt lại `1.0` để bật clip. Config resolved từ run trước cần sửa trường này riêng.

## Chạy thử offline

Sau khi môi trường đã có dependencies trong `requirements.txt`:

```bash
cd src/eviseq_update
PYTHON=python3 bash scripts/run_afmr.sh smoke
```

Smoke dùng tiny Qwen ngẫu nhiên, tắt mạng, bật cả copy và semantic read. Nó kiểm tra
gradient lúc khởi tạo và sau training, hai training stages, checkpoint round-trip,
dense/chunked CE, greedy evaluation và prediction resume. Dữ liệu thử nằm trong
`tests/fixtures`; checkpoints/predictions smoke nằm trong temporary directory tự xóa.
Kết quả cuối phải có `"status": "ok"` và `"semantic_read": true`.

Fixture `configs/afmr_smoke.yaml` giữ copy/semantic read tắt để chạy các regression
tests LM-only kế thừa; lệnh `smoke` chủ động bật cả hai. Đây không phải config để
đo chất lượng task thực.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 -m pytest -q -p no:cacheprovider
```

## Train và eval PubMed

Sửa đường dẫn `model.encoder_name`, `model.decoder_name` trong
`configs/afmr_pubmed.yaml`. Các split cần là JSONL với `id`, `text`, `summary`.
Có thể trỏ `data.train_file`, `validation_file`, `test_file` bằng đường dẫn tuyệt đối
tới dữ liệu đã chuẩn bị ở `eviseq_new`, để hai bản dùng đúng cùng split.
Đường dẫn tương đối của config bundled được resolve từ thư mục `eviseq_update`;
config ngoài thư mục `configs/` resolve từ thư mục chứa chính config đó.

```bash
cd src/eviseq_update
CUDA_VISIBLE_DEVICES=0 PYTHON=python3 \
  bash scripts/run_afmr.sh train configs/afmr_pubmed.yaml
```

### Train trên hai GPU

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 GRADIENT_ACCUMULATION_STEPS=4 PYTHON=python3 \
  bash scripts/run_afmr.sh train configs/afmr_pubmed.yaml \
  --output-dir runs/eviseq_update/pubmed_copy_read_2gpu
```

Script dùng `torchrun`, mỗi tiến trình chạy một GPU qua NCCL/DDP. Recipe có
batch 4 trên mỗi GPU: `4 × 2 GPU × accumulation 4 = 32` mẫu/update, bằng run
một GPU với accumulation 8. Nếu giảm batch mỗi GPU, tăng accumulation tương ứng.
Override accumulation được lưu trong `resolved_config.yaml` của run.

Loss được cân theo **tổng token được giám sát trên cả hai rank và toàn bộ cửa sổ
accumulation**. DDP chỉ đồng bộ backward cuối cửa sổ, sau đó mới đo norm và cập nhật
trọng số. Dữ liệu được chia theo global batch; batch cuối không bỏ hoặc tính trùng
mẫu. Rank thiếu mẫu chạy placeholder có toàn bộ labels `-100` và trọng số loss 0.
Validation CE cũng tổng hợp numerator/token count trên hai GPU.

Reducer DDP được tạo lại sau khi chuyển warm-up sang full fine-tuning để nhận cả
backbone vừa được mở gradient. Chỉ rank 0 ghi metrics/config/checkpoint; checkpoint
lưu RNG từng rank và weights không có tiền tố DDP, nên có thể eval một GPU. Resume
với cùng số GPU, batch và accumulation để giữ lịch update; ví dụ thêm
`--resume-checkpoint runs/eviseq_update/pubmed_copy_read_2gpu/last.pt` vào lệnh trên.

Mỗi GPU chứa một bản đầy đủ của model và optimizer; DDP không gộp VRAM hai card.
Tốc độ thực tế phụ thuộc độ dài mẫu và kết nối GPU. Greedy eval/ROUGE vẫn dùng một
tiến trình: chạy `evaluate` như bình thường, không dùng `torchrun` cho eval.

Queue hai encoder mặc định dùng GPU `0,1`, hai DDP workers và accumulation 4.
Sau khi đường dẫn model/data đã đúng, chỉ cần chạy:

```bash
bash scripts/run_pubmed_pair.sh
```

Hai encoder được train lần lượt, mỗi run dùng cả hai GPU. Các đường dẫn model/data
của queue được cấu hình bằng các biến môi trường mô tả bên dưới. Script kiểm tra
số GPU CUDA/NCCL trước khi chuẩn bị dữ liệu và lưu accumulation vào config tạo ra.
Nếu cần một GPU, override cả worker count và accumulation để giữ batch 32:

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 GRADIENT_ACCUMULATION_STEPS=8 \
  bash scripts/run_pubmed_pair.sh
```

Recipe PubMed dùng 1 epoch interface warm-up + 3 epoch full fine-tuning, như bản gốc.
Checkpoint là `epoch_001.pt`, ..., `last.pt`. Eval epoch 3 trên test:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python3 bash scripts/run_afmr.sh evaluate \
  configs/afmr_pubmed.yaml \
  runs/eviseq_update/pubmed_value_anchor_copy_read/epoch_003.pt \
  runs/eviseq_update/pubmed_value_anchor_copy_read/epoch003_test_predictions.jsonl \
  --split test --batch-size 32
```

Runtime báo Python ROUGE 1.0.0. Để so với mốc T5Gemma Perl ROUGE 1.5.5, chấm lại
hai bộ predictions bằng cùng wrapper/tokenization/flags đã dùng cho baseline.
Chọn config/epoch bằng validation; test dành cho đánh giá đã chốt.

Queue PPLX rồi Qwen3-Embedding vẫn có ở `scripts/run_pubmed_pair.sh`. Nó mặc định
bật semantic read và ghi vào `pubmed_pair_afmr_value_anchor_copy_read`. Có thể đặt
`PROCESSED_DATA_DIR` thành đường dẫn tuyệt đối tới ba split đã chuẩn bị, hoặc đặt
`PUBMED_SOURCE_DIR` để chuẩn bị từ raw. Các biến `PPLX_ENCODER`, `QWEN_ENCODER`,
`DECODER_MODEL`, `PYTHON`, `EVAL_BATCH_SIZE` điều khiển đường dẫn/tài nguyên.
`AFMR_SEMANTIC_READ=false` chạy copy-only trong directory khác. Queue không tự
ghi đè run có sẵn; `OVERWRITE_OUTPUT_DIR=true` là yêu cầu reset run có chủ đích.

Smoke model thực trên GPU, có giới hạn số mẫu và directory riêng:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python3 \
  bash scripts/smoke_a100.sh configs/afmr_pubmed.yaml
```

## Phạm vi xác minh

Các tests kiểm tra initialization parity, CE/gradient parity khi nhánh đã hoạt động,
gradient cho target không copy được, masked/empty source, BF16 autocast với FP32
updates, sparse alignment, cache compaction, checkpoint compatibility và runtime
train/resume/eval. Test hai tiến trình CPU/Gloo đối chiếu gradient từng tham số
trước optimizer step và weights sau update với một tiến trình, với target dài/ngắn
khác nhau, accumulation dư, rank có zero labels, cả hai stages và ba chế độ
LM-only/copy/semantic read. Nó kiểm tra cả AdamW resume, RNG từng rank và eval
checkpoint DDP trên một tiến trình. Đây là kiểm tra correctness trên model nhỏ; throughput/VRAM GPU
và ROUGE của bản update cần được đo bằng run thực.

Triển khai này chỉ thêm shared context read. Đọc phân cấp theo vùng, chunked
long-source encoding, BRIO và world-model predictor là các ứng viên nghiên cứu khác,
chưa được ghép vào để giữ phép so sánh có thể xác định tác dụng của thay đổi này.
