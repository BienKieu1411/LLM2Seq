# EviSeq Update v3 — 4×128 đọc toàn nguồn, gate sau normalization

Bản hiện tại dùng **4 semantic heads × 128 chiều**, mỗi head được đọc mọi vùng
nguồn, theo dõi usage của prefix và giữ norm khi fusion với LM hidden. Head gate
đặt sau context normalization để giữ được tác dụng giảm đóng góp. Đây là bản sửa
theo Luna review ngày 2026-09-09; chưa có ROUGE PubMed để xác nhận vượt v2/new/T5Gemma.

Package độc lập `eviseq_update_v3`; `src/eviseq_update` cũ được giữ nguyên.
Xem [report phương pháp hiện hành](../../Technical_Report/EVISEQ_UPDATE_V3_REVIEW_REVISION.md).
[Report coverage trước review](../../Technical_Report/EVISEQ_UPDATE_V3_COVERAGE_REVISION.md) và
[Report v3 ban đầu](../../Technical_Report/EVISEQ_UPDATE_V3_ARCHITECTURE.md) được
giữ làm lịch sử thiết kế, không mô tả mặc định hiện tại.

## Bốn thay đổi

1. **Matching đủ128 chiều/head, đọc toàn nguồn.** Tổng rank512; mỗi head được
   đọc mọi vùng64 content tokens qua attention region→token. Bỏ mask modulo
   giúp head không bị bỏ trống hoặc chỉ còn một vùng để chọn khi nguồn ngắn.
   Head gate đặt sau context normalization, trước output projection; không có
   normalization tiếp theo triệt tiêu hệ số giảm đóng góp trước projection.
2. **Coverage và continuity từ prefix.** Một tracker học bằng CE ước lượng vùng
   đang được diễn đạt; cumulative usage làm giảm ưu tiên vùng đã dùng, tín hiệu
   vùng gần nhất hỗ trợ tiếp nối. Prompt/padding không tăng coverage.
3. **Fusion giữ norm.** Correction được chiếu vuông góc với hidden, giới hạn
   relative RMS0.10 rồi chuẩn hóa về norm hidden gốc. Nhánh mới thay hướng hidden,
   không dùng tăng magnitude của LM input như một cách thay scale logits.
4. **Giữ đường copy v2.** Mặc định tắt query gate mới trong backbone; gate theo
   head đặt trong semantic read. Copy nhận hidden trước fusion, giữ cùng công
   thức attention/gate/marginalization. Shared weights vẫn học qua CE nên không
   có cam kết ROUGE-2 bất biến sau training.

Coverage là ước lượng học được, không phải kiểm chứng fact hoặc sentence planner
biết chắc các ý đã hoàn thành. Block64 tokens vẫn có thể cắt câu. Các head được
đọc chung nguồn nên có thể học trùng; chưa có bảo đảm đa dạng fact. Khi chỉ có
một vùng, coverage không thể thay đổi lựa chọn vùng. Hiệu quả thực cần ablation.

## Objective và tính toán

Chỉ dùng gold-token CE của mixture LM/copy. Không contrastive, evidence mining,
R-Drop, NEFTune, teacher hay sinh thêm dữ liệu trong training. Temperature/top-p
chỉ có trong API sinh candidates ngoài training.

Một encoder/decoder forward như cũ. Prefix tracker tính song song với cumulative
sum, không autoregressively sinh thêm một summary. CE chunk nhận đúng prefix
state xuyên ranh giới chunk. Generation lưu coverage/recent-region state và
compact cùng self/cross/source caches.

Với PubMed: FP32 parameters/gradients/AdamW, BF16 autocast CUDA, clip1.0. Gradient
checkpointing tính lại activations trong backward như cũ. Tất cả semantic/planner
parameters thuộc optimizer group `cross_attention`, train ở cả hai stages.

## Config chính

```yaml
decoder:
  query_cross_gate: false
  grounded_copy:
    enabled: true
    key_dim: 128
    gate_init: 0.05
    semantic_read:
      enabled: true
      rank: 512          # Tổng rank = 4 × 128
      num_heads: 4
      attention: hierarchical_coverage
      fusion: norm_preserving
      max_relative_rms: 0.10
      gate_init: 0.05
      head_gate_position: post_norm
      planner:
        region_size: 64
        partition_heads: false
        use_coverage: true
        use_continuity: true
        coverage_scale: 8.0
        coverage_init: 0.2
        coverage_max: 2.0
        continuity_init: 0.2
        continuity_max: 2.0
```

`rank:128,num_heads:4` nghĩa là4×32, không phải4×128.
Checkpoint spec ghi các thay đổi graph, kể cả vị trí gate dù shape trọng số
không đổi. Không resume checkpoint graph cũ vào config mới. Muốn eval checkpoint
cũ dùng đúng resolved config cũ; thiếu `head_gate_position` được hiểu là
`pre_norm`. Train các đối chứng từ cùng pretrained backbones.

## Cài đặt và chạy

Từ thư mục `src/eviseq_update_v3`:

```bash
python -m pip install -e '.[dev]'
RUN_ENCODERS=pplx EVAL_SPLIT=validation bash scripts/run_pubmed_pair.sh
```

Mặc định script chạy PPLX rồi Qwen-Embedding tuần tự, mỗi run dùng2GPU. Lệnh trên
chỉ chạy PPLX và eval validation để chọn kiến trúc. Với benchmark cấu hình đã chốt,
dùng `EVAL_SPLIT=test` (mặc định). Eval dùng `last.pt`, greedy, mộtGPU.

Đường dẫn model/data có mặc định theo server cũ; có thể đặt lại:

```bash
PPLX_ENCODER=/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/path/to/Qwen3-0.6B \
PROCESSED_DATA_DIR=/path/to/prepared/pubmed \
RUN_ENCODERS=pplx EVAL_SPLIT=validation \
bash scripts/run_pubmed_pair.sh
```

- 2GPU: `BATCH_SIZE=48` mỗiGPU, accum1 ⇒ global96.
- 1GPU: `CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1`, mặc định48×accum2 ⇒96.
- 84/GPU: `BATCH_SIZE=84 GRADIENT_ACCUMULATION_STEPS=1` ⇒ global168 trên2GPU;
  cần đối chứng cùng protocol, không coi như tái lập global96.
- `EVAL_BATCH_SIZE=64`, `MAX_GRAD_NORM=1.0`.
- 1 epoch interface warmup +3 epochs full finetune; linear decay từng stage.
  Interface warmup là freeze backbone, không phải LR warmup.
- Dùng lại dữ liệu canonical JSONL `id/source/target` đã chuẩn bị cho v2 bằng
  `PROCESSED_DATA_DIR`; không cần evidence cache.
- Output/log trong `runs/eviseq_update_v3`, `logs/eviseq_update_v3`; tên run ghi
  variant, rank, heads, fusion, vị trí head gate và planner flags. `resolved_config.yaml` lưu
  cấu hình thực. Đặt `RUN_ROOT` riêng khi đổi seed/batch/data.

## Ablation

| Đối chứng | Thiết lập thêm |
|---|---|
| V2 control1×128 | `AFMR_SEMANTIC_VARIANT=independent_bounded SEMANTIC_RANK=128 SEMANTIC_HEADS=1` |
| Free heads4×32 | `AFMR_SEMANTIC_VARIANT=independent_bounded SEMANTIC_RANK=128` |
| Free heads4×128 | `AFMR_SEMANTIC_VARIANT=independent_bounded` |
| V3 trước review | `PARTITION_HEADS=true SEMANTIC_HEAD_GATE_POSITION=pre_norm` |
| Chỉ bỏ phân công vùng cố định | `SEMANTIC_HEAD_GATE_POSITION=pre_norm` |
| Chỉ sửa vị trí head gate | `PARTITION_HEADS=true` |
| V3 sau review đầy đủ | Mặc định: `PARTITION_HEADS=false SEMANTIC_HEAD_GATE_POSITION=post_norm` |
| Không coverage | `USE_COVERAGE=false` |
| Không continuity | `USE_CONTINUITY=false` |
| Fusion cộng residual cũ | `SEMANTIC_FUSION=residual` |

`CROSS_QUERY_GATE=false` mặc định. Muốn đối chứng v3 ban đầu:
free heads4×32 với `CROSS_QUERY_GATE=true`.

Script chọn fusion cũ cho independent/shared variants, fusion giữ norm cho
hierarchical_coverage. `SEMANTIC_FUSION` cho phép override. Cần giữ cùng
data/model/seed/global batch/LR/stages/clip/limits/decoding và chọn bằng validation.

## ROUGE và sampling

Benchmark giữ `num_beams=1,do_sample=false`. Config validation từ chối sampling
trong benchmark. API `evaluation.generate.generate_sampled` giữ `temperature`,
`top_p`, `generator`; training không gọi API đó.

Evaluator mặc định báo Python `rouge==1.0.0`. Để so các score ROUGE155 đã báo,
đặt `ROUGE155_SCRIPT=/path/to/rouge155_wrapper.py`; runner gọi:

```bash
python /path/to/rouge155_wrapper.py predictions.jsonl --output scores.rouge155.json
```

Giữ wrapper/Perl/resources/flags và detokenization giống baseline. Nếu không đặt
wrapper, script ghi rõ ROUGE155 bị bỏ qua; không coi Python score là Perl score.

Mỗi file predictions có `.manifest.json` chứa fingerprint checkpoint, split,
config suy luận, tokenizer, implementation và môi trường thư viện. Eval kiểm tra
manifest trước cả nhánh trả metrics từ file đã hoàn tất. Đổi checkpoint/source/
config mà dùng lại predictions cũ sẽ báo lỗi; chọn output mới để eval lại.
File cũ không có manifest vẫn có thể chấm ROUGE riêng, nhưng không được resume
bằng evaluator mới. Resume đúng run cho phép đổi batch size và tăng prefix limit.
Hash đọc checkpoint/dataset một lần đầu mỗi lệnh eval; không thêm bước vào training.

## Kiểm chứng và chi phí

```bash
PYTHONPATH=. python -m pytest -q -p no:cacheprovider
PYTHONPATH=. python scripts/smoke_test.py
```

Tests kiểm tra oracle hierarchy, cả free heads và partition ablation, prefix scan/incremental
parity, prompt/padding/future exclusion, FP32/BF16 gradients, dense/chunk CE,
fusion norm/bound, copy isolation, optimizer2stages, checkpoint và DDP2tiến trình
Gloo. Thêm regression cho head gate sau norm, source1–4 vùng và eval provenance.
Smoke dùng tiny4heads với rank thu nhỏ, bật graph sau review.

Kết quả kiểm tra lần sửa này được ghi trong
[report sau review](../../Technical_Report/EVISEQ_UPDATE_V3_REVIEW_REVISION.md).
Chưa chạy full PubMed hoặc NCCL/CUDA trong môi trường này.

Ở hidden1024: semantic+planner có2,627,592 tham số, so với524,545 của v2.
Semantic K/V rộng512 thay128 nên riêng cache này tăng4 lần. Thêm tracker theo vùng
và scatter/cumsum; chưa có đo throughput/VRAM CUDA. Không có cơ sở nói tổng train
chậm4 lần, cũng chưa có cơ sở gán một phần trăm overhead nhỏ cố định.
Hai sửa đổi sau review không thêm tham số và giữ nguyên shape attention/cache
so với v3 trước review. Eval CUDA vẫn giữ semantic K/V FP32: riêng hai tensor
này ở batch64 × source4096 × rank512 chiếm khoảng1GiB.

Các tests không thay thế full PubMed training hoặc chứng minh tăng R1/R2/L.
