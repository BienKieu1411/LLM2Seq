# EviSeq Update v3: query-gated cross-attention + multihead semantic read

V3 là package độc lập `eviseq_update_v3`, scaffold từ bản CE-only v2 ở commit
`381ebe7`. Không cần import hoặc cài package `eviseq_update`. Bản cũ được giữ nguyên.

Kết quả người dùng cung cấp: new **49.626 / 21.901 / 45.895** và v2
**49.488 / 21.953 / 45.776** (ROUGE-1/2/L). Chênh lệch nhỏ này chưa chứng minh
nguyên nhân hồi quy, ý nghĩa thống kê hay hiệu quả v3. **V3 chưa có kết quả PubMed.**

## Thay đổi kiến trúc

1. **Cross gate theo target token/head.** Mỗi layer dùng
   `2 * sigmoid(W_g * query_states + b_g)` sau SDPA, trước output projection.
   Gate mới zero-init nên hệ số ban đầu bằng 1; static layer gate và các
   projections sao chép từ pretrained decoder giữ nguyên.
2. **Semantic read 4 head, tổng rank 128.** Mỗi head đọc native source riêng
   với 32 chiều. Keys chuẩn hóa RMS theo head; contexts nối lại rồi đi qua
   semantic gate/output/residual như v2. Số tham số và số phần tử cache của
   nhánh semantic bằng bản 1 head cùng rank.

Giữ lexical copy, AFMR value anchor, source prior, semantic output zero-init,
scalar semantic gate 0.05 và residual cap 0.10. Không đổi post-norm fusion,
nới cap hoặc tăng rank cùng lúc. Không thêm encoder/decoder forward.

Cơ sở: [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need)
và [Gated Attention, NeurIPS 2025](https://arxiv.org/abs/2505.06708).
Gate `2*sigmoid` giữ function ban đầu là adaptation ở đây; không phải công thức
nguyên bản của paper, cũng không tự chứng minh novelty hoặc tăng ROUGE PubMed.

Xem [report phương pháp và kế hoạch đối chứng](../../Technical_Report/EVISEQ_UPDATE_V3_ARCHITECTURE.md).

## Objective và gradient

Chỉ tối ưu gold-token CE của distribution trộn LM/copy. Không có contrastive,
evidence mining/cache, R-Drop, NEFTune, teacher, self-improve hoặc sinh candidates
trong training. Reference chỉ dùng theo teacher forcing và CE.

Cross gate học từ bước đầu. Semantic output zero-init học trước; Q/K/V và
semantic gate nhận gradient có ích sau khi output projection khác 0. Tất cả
tham số mới thuộc nhóm optimizer `cross_attention`, được cập nhật trong cả
interface warmup và full finetune.

Với PubMed: FP32 parameters/gradients/AdamW, BF16 autocast trên CUDA, gradient
clipping 1.0. CE chuẩn hóa theo số gold tokens thực trên tất cả ranks và các
microbatch trong một optimizer update. Gradient checkpointing tính lại activations
khi backward; đó không phải một objective hay một lượt sinh dữ liệu bổ sung.

## Cài đặt và dữ liệu

Chạy từ thư mục `src/eviseq_update_v3`:

```bash
python -m pip install -e '.[dev]'
```

Dữ liệu canonical là JSONL `id/source/target`; script có thể chuẩn bị dữ liệu từ
`train.label.jsonl`, `val.label.jsonl`, `test.label.jsonl`. Có thể dùng lại đúng
bản dữ liệu đã chuẩn bị cho v2 bằng `PROCESSED_DATA_DIR`; không cần evidence cache.

## Chạy PubMed

`scripts/run_pubmed_pair.sh` mặc định chạy PPLX rồi Qwen-Embedding **tuần tự**;
mỗi run train trên 2 GPU, sau đó greedy eval `last.pt` trên tập test bằng 1 GPU.
Để chỉ chạy PPLX:

```bash
RUN_ENCODERS=pplx bash scripts/run_pubmed_pair.sh
```

Đường dẫn model/data có mặc định theo server cũ; nếu server khác, đặt:

```bash
PPLX_ENCODER=/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/path/to/Qwen3-0.6B \
PROCESSED_DATA_DIR=/path/to/prepared/pubmed \
RUN_ENCODERS=pplx \
bash scripts/run_pubmed_pair.sh
```

- 2 GPU mặc định: batch 48/GPU × 2 × accum 1 = **96 examples/update**.
- 1 GPU: `CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1`, mặc định batch 48 × accum 2 = 96.
- Muốn batch 84/GPU: `BATCH_SIZE=84 GRADIENT_ACCUMULATION_STEPS=1`; trên 2 GPU
  thành global batch 168, cần chạy đối chứng cùng setting để so sánh kiến trúc.
- `EVAL_BATCH_SIZE=64` mặc định; eval batch không đổi global batch training.
- `MAX_GRAD_NORM=1.0`, 1 epoch interface warmup + 3 epochs full finetune.
  LR decay **linear theo từng stage**, giữ recipe CE-only ở commit v2 nói trên.
  Interface warmup là giai đoạn freeze backbone, không phải LR warmup.
- Kết quả v2 vừa báo chưa có resolved config để xác nhận protocol.
  Không mặc nhiên coi cấu hình hiện tại là cấu hình của run đó.

Output/log nằm dưới `runs/eviseq_update_v3` và `logs/eviseq_update_v3`.
Tên output phân biệt query gate và số head. `RUN_ROOT` cho phép chỉ định thư mục
riêng, ví dụ khi đổi seed hoặc batch. `resolved_config.yaml` lưu cấu hình thực chạy.

## Ablation chỉ thay kiến trúc

Mặc định `CROSS_QUERY_GATE=true SEMANTIC_HEADS=4`. Chạy từng đối chứng cùng
data, seed, global batch, số epochs, LR, clip, source/target limits và decoding:

| Ablation | CROSS_QUERY_GATE | SEMANTIC_HEADS |
|---|---|---|
| Graph CE-only v2 | false | 1 |
| Chỉ gate mới | true | 1 |
| Chỉ semantic nhiều head | false | 4 |
| V3 đầy đủ | true | 4 |

Ví dụ control v2:

```bash
RUN_ENCODERS=pplx CROSS_QUERY_GATE=false SEMANTIC_HEADS=1 \
bash scripts/run_pubmed_pair.sh
```

Head count được ghi vào checkpoint architecture spec dù shapes của weights
giống nhau. Không resume v2 checkpoint vào v3 đầy đủ. Control gate-off/head1 giữ
spec và arithmetic v2; mỗi run kiến trúc mới nên train từ cùng pretrained base.

Runner giữ last-epoch test eval cho benchmark cố định trước. Khi chọn ablation
hoặc điều chỉnh hyperparameters, dùng validation trước, chốt cấu hình rồi mới
đánh giá test; không chọn kiến trúc dựa trên việc thử đi thử lại test.

## Greedy evaluation và sampling

Benchmark eval giữ `num_beams=1, do_sample=false`, không dùng temperature/top-p.
Config validation chủ động từ chối bật sampling trong benchmark.

API `eviseq_update_v3.evaluation.generate.generate_sampled` giữ
`temperature`, `top_p` và `generator` riêng cho sinh candidates khi cần.
Training không gọi API này. Chọn `temperature > 0`, `0 < top_p <= 1`.

`evaluate` mặc định báo Python `rouge==1.0.0`. Để đối chiếu các điểm người dùng
báo bằng **ROUGE-1.5.5**, đặt `ROUGE155_SCRIPT=/path/to/rouge155_wrapper.py`.
Runner gọi wrapper theo giao diện:

```bash
python /path/to/rouge155_wrapper.py predictions.jsonl --output scores.rouge155.json
```

Wrapper/Perl/ROUGE resources phải có sẵn trên máy chạy. Giữ tokenizer,
detokenization và các flags ROUGE giống baseline; không so trực tiếp hai backend.

## Kiểm chứng

```bash
PYTHONPATH=. python -m pytest -q -p no:cacheprovider
PYTHONPATH=. python scripts/smoke_test.py
```

Test dùng tiny models offline: attention oracle, gradient CE FP32/BF16,
dense/chunk parity, source masks, zero-label batch, optimizer hai stage,
checkpoint compatibility, cached prefix/compaction và DDP hai tiến trình Gloo.
Smoke bật gate mới và 4 semantic heads, train 2 stage rồi checkpoint/resume/eval.

Kết quả local 2026-09-08: **253 tests passed**, smoke passed. Control gate-off/head1
đã được đối chiếu trực tiếp với commit CE v2: logits, CE và gradients khớp chính
xác trên tiny model ở FP32/BF16, kể cả khi semantic residual đã khác zero.

Những kiểm tra này không thay thế phép đo tốc độ/VRAM trên CUDA, full PubMed
training hoặc chứng minh v3 vượt new/T5Gemma. QKV/cache semantic giữ nguyên tổng
chiều; softmax có thêm heads. SDPA có thể dùng fused kernels, nhưng thời gian và
VRAM thực còn tùy kernel, batch và độ dài dữ liệu.
