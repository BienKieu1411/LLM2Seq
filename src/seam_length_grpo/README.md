# SEAM length-conditioned GRPO

Phase GRPO sau SFT, đặt riêng trong `src/seam_length_grpo`. Không thay kiến trúc,
copy head hoặc vòng training của `eviseq_new`. Thư mục này tái sử dụng core của
`../seam_grpo` và model/SEAM vLLM ở `../eviseq_new`; khi chuyển lên server cần có
cả ba thư mục cùng phiên bản code.

## Cơ chế

Mỗi source có một yêu cầu **W từ**. Encoder nhận hướng dẫn đọc để chọn thông tin
phù hợp W; decoder nhận hướng dẫn viết bản tóm tắt khoảng W từ. Cả hai nhận cùng
một W. Decoder vẫn dùng `decoder_prefix` và chat template của checkpoint SFT.
Prompt decoder cũng đi vào controller/bridge theo cơ chế sẵn có của SEAM.

Ngân sách source và prompt tách riêng:

- `source_content_tokens`: token dành cho nội dung source.
- `encoder_prompt_reserve`: phần dự phòng cho toàn bộ prefix và special tokens.
- `decoder_prompt_reserve`: giới hạn prompt decoder sau chat template/prefix.
- `training.max_new_tokens`: trần **token sinh**, khác W **từ** mong muốn.

Source được cắt theo offset tokenizer trước khi thêm hướng dẫn. Đổi W không
làm mất thêm nội dung source. Code từ chối prompt vượt reserve hoặc tổng input
vượt khả năng vị trí mà backbone công bố. Copy alignment được dựng lại bằng
prefix thật của từng sample; chỉ copy từ source, không copy phần hướng dẫn.

Reward của mỗi candidate:

```
Q = 0.25 * R1_F1 + 0.50 * R2_F1 + 0.25 * RL_F1
cost = abs(generated_words - W) / error_budget_words
R = Q - length_weight * cost + source_reward.weight * source_score
loss = TRL Dr. GRPO loss + ce_weight * gold_CE
```

Trước GRPO có một đoạn **warm-up chỉ CE** để thích nghi với prompt có số từ.
Mỗi sample vẫn có W riêng, lấy từ `target_words` hoặc số từ gold nếu thiếu.
Warm-up dùng cùng renderer encoder/decoder, source budget, decoder prefix và
copy alignment như GRPO. Không rollout, không sinh thêm dữ liệu, không dùng
reward; CE ở giai đoạn này có hệ số 1. Sau đó GRPO tiếp tục với CE nhỏ 0.05.

```yaml
ce_warmup:
  fraction: 0.05
  max_steps: 0
```

Mặc định số update warm-up bằng 5% số update của **một lượt** train, làm tròn
lên, tối thiểu một update. Đây là thiết lập khởi đầu để thử nghiệm, chưa được
chứng minh tối ưu. `max_steps>0` ghi đè fraction; đặt cả hai bằng 0 để bỏ qua.
Warm-up dùng learning rate từng component và cosine riêng. Đây là phase CE,
khác `training.warmup_ratio` là đoạn tăng learning rate trong mỗi lịch.

Checkpoint `length_ce_warmup.pt` cùng `ce_warmup_metrics.json` ghi lại đoạn
thích nghi. GRPO dùng trọng số đã thích nghi nhưng tạo optimizer/cosine mới,
global step GRPO bắt đầu từ 0. Actor được đồng bộ sang vLLM sau warm-up, trước
rollout đầu tiên. Teacher và KL reference vẫn nạp từ SFT gốc, không lấy trọng
số đang thích nghi. Các bước warm-up không tính vào `training.max_steps` GRPO.

`length_weight=0.10`, `ce_weight=0.05` là điểm khởi đầu, chưa phải hệ số đã được
thực nghiệm chứng minh tối ưu. Đây là penalty cố định, không phải thuật toán
Lagrange multiplier thích nghi. Mục tiêu tăng ROUGE và đạt chỉ tiêu length cần
được kiểm chứng trên validation.

Đếm từ khớp `src/evaluation/evaluate_length_vietnamese.py`: chuẩn hóa NFKC rồi
đếm Unicode word bằng regex, không đếm token subword hay từ đã tách bằng mô hình
phân đoạn tiếng Việt. ROUGE trong reward là proxy Unicode/whitespace nhanh của
`seam_grpo`, **không phải Perl ROUGE-1.5.5**. Báo cáo cuối vẫn chạy evaluator Perl
hiện tại trên predictions.

## Data

### Ghi số từ vào từng sample trước training

Chạy trên JSONL mixed **sau bước gộp data**, rồi trỏ `train_file`/`validation_file`
vào output tương ứng. Mỗi dòng được thêm `target_words` riêng:

```bash
python3 src/seam_length_grpo/prepare_data.py \
  --input /path/mixed/train.jsonl \
  --output /path/mixed/train_with_length.jsonl \
  --config src/seam_length_grpo/configs/grpo.yaml
```

`--config` dùng cùng `target_field`, list separator và chế độ detokenize như
dataloader SFT. Chỉ đọc YAML, không cần thư mục model/checkpoint tồn tại và
không tải tokenizer. Nếu không dùng config:

```bash
python3 src/seam_length_grpo/prepare_data.py \
  --input /path/sds.jsonl --output /path/sds_with_length.jsonl \
  --target-field abstract_sum --detokenize
```

Hỗ trợ trường target dạng text, list text hoặc đường dẫn lồng nhau như
`answer.summary`. Giữ nguyên source, target và các trường khác; detokenize chỉ
dùng để tính số từ, không sửa nội dung lưu. Không lọc/giảm số sample, không cắt
token. Dòng lỗi hoặc target không có từ sẽ báo số dòng và không công bố output
dở dang. Input không bị ghi đè. Output đã có cần `--overwrite`; nếu trường
length cũ khác kết quả đếm cần `--replace-length` để chủ động tính lại.
Script in tổng sample và min/max/mean số từ ra terminal.
Length do script này thêm được lấy từ **gold**, phù hợp để prepare train và
validation. Nếu áp dụng cho test, đó là giao thức được cung cấp độ dài reference
(oracle), không phải yêu cầu số từ độc lập từ người dùng; phải ghi rõ khi báo cáo.

Dùng JSONL đã prepare, với tên trường source/target theo `base_config`. Ví dụ
khi base dùng `text` và `summary`:

```json
{"id":"sample_1","text":"Văn bản nguồn...","summary":"Một bản tóm tắt ngắn.","dataset":"vietnews","language":"vi","target_words":5}
```

Trong train và validation, thiếu `target_words` thì lấy số từ của summary đã
detokenize. Nếu bật CE, W phải bằng số từ của gold; gold không được bị cắt mất
token. Code báo lỗi thay vì lặng lẽ học một target trái yêu cầu. Nếu reference
dài hơn `data.max_target_length`, hãy tăng giới hạn đó trong bản config SFT
dùng cho phase này hoặc chuẩn bị tập train phù hợp. Không tự bỏ sample.

`dataset` phải khớp chính xác key trong `dataset_error_budgets`. Nếu file mixed
cũ không còn metadata dataset, code dùng `default_dataset` và
`default_error_budget_words`; không tự suy đoán domain từ nội dung. Muốn áp
ngân sách từng bộ thì cần giữ trường dataset trong input. Mỗi dòng cũng có thể
ghi đè bằng `length_error_budget_words`.

Ngân sách mặc định trong config mẫu: WikiLingua 43, Vietnews 38, XLSum 40,
Legal 72, Book 59, ViM 98, VLSP22 34 từ sai lệch tuyệt đối. Đây là chỉ tiêu MAE
cấp dataset, không phải bắt buộc từng sample có sai số nhỏ hơn ngân sách.

## Source reward tận dụng encoder-decoder

`source_reward.weight: 0.05` mặc định để reward có đủ ba thành phần ROUGE,
length và source. Đặt `0.0` nếu chủ động làm ablation bỏ nhánh teacher:

1. Đóng băng **toàn bộ SEAM SFT**, gồm encoder, bridge, decoder và copy.
2. Chấm chính candidate vừa sinh với source đúng và source TRAIN không liên quan.
3. Giữ nguyên candidate, W, prompt decoder, sampling token cap và số token được chấm.
4. Với source sai, dựng lại encoder states, bridge memory/bias và copy state.
5. `source_score = tanh((mean_logp_true - mean_logp_negative) / tau)`.

Mean log-prob chỉ tính token action thật, gồm EOS nếu được sinh; temperature
chấm teacher là 1. Teacher không cập nhật, không sinh thêm candidate, không
dùng test. Negative được chọn xác định từ pool hash giới hạn **mỗi
dataset/ngôn ngữ**, xét độ dài và Jaccard trên phần source thật sự nhìn thấy.
Thiếu negative phù hợp thì báo lỗi; tăng `reservoir_size` hoặc kiểm tra data.
Nếu thiếu `language`, chỉ có nhãn `unspecified`, không có bộ nhận diện ngôn ngữ.

Đây là tín hiệu phân biệt source mang tính thử nghiệm, không phải thước đo
factuality, không phải CPMI gốc và chưa có kết quả cho thấy nó tăng ROUGE.
Nhánh này cần thêm một bản full SFT trên mỗi GPU và hai lượt teacher scoring
cho mỗi candidate. Nếu bật `beta>0` với cùng SFT, reference KL và teacher dùng
chung bản frozen. Không cache encoder states qua các lần cập nhật actor.

## Chạy train

Yêu cầu môi trường của `eviseq_new`, **TRL 1.16.0**. Nếu thiếu TRL:

```bash
python3 -m pip install -r src/seam_length_grpo/requirements.txt
```

Sửa `configs/grpo.yaml`: `base_config` trỏ vào resolved config SFT,
`sft_checkpoint` trỏ `best.pt`/`last.pt`, `output_dir` là thư mục mới.
Đường dẫn tương đối tính từ file YAML. Model backbone phải là thư mục local.
Có thể ghi đè `train_file` và `validation_file` ở đầu YAML. Encoder/decoder
family nào native SEAM hỗ trợ thì phase này dùng lại native loader đó.

Từ root repository:

```bash
# Chỉ kiểm tra config/path; không tải hoặc load trọng số model.
python3 src/seam_length_grpo/run_grpo.py \
  --config src/seam_length_grpo/configs/grpo.yaml --dry-run

# 1 GPU
CUDA_VISIBLE_DEVICES=0 bash src/seam_length_grpo/scripts/run_grpo.sh \
  --config src/seam_length_grpo/configs/grpo.yaml

# 2 GPU, mỗi GPU giữ một full actor, gradient đồng bộ bằng DDP
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash src/seam_length_grpo/scripts/run_grpo.sh \
  --config src/seam_length_grpo/configs/grpo.yaml
```

`prompt_batch_size` là số source/GPU/microbatch; mỗi source có `group_size`
candidate. Tổng source/optimizer update bằng
`prompt_batch_size × gradient_accumulation_steps × số GPU`. Candidate giữ
nguyên source/W/negative trong nhóm. Đuôi dataset được lặp thêm tối đa
`update_sources-1` dòng để đủ update, không bỏ sample; số gốc và số sau padding
ghi trong `implementation.json`.

Native rollout ghép các candidate liên tiếp có cùng độ rộng prompt thành
microbatch tối đa `rollout_micro_batch_size`, dùng KV cache khi decode.
`score_micro_batch_size` điều khiển bộ nhớ chấm likelihood. Khi training,
encoder states được tính lại với autograd để encoder/bridge cũng nhận gradient.
Copy vẫn hoạt động nếu checkpoint SFT bật copy.

Mỗi update sinh bằng policy phiên bản k, chấm exact mixture likelihood,
backward/cập nhật đầy đủ actor rồi mới sinh bằng k+1. Sampling dùng temperature
dương, full support (`top_p=1`, không top-k/repetition penalty) để không làm sai
importance ratio. Rollout không mang gradient; policy scoring và CE có gradient.
Dropout tắt ở cả hai bước. Learning rate dùng cosine decay.

### SEAM vLLM rollout

Đặt `training.rollout_backend: seam_vllm` với vLLM **0.25.1** và plugin SEAM
trong checkout. Chạy cùng script 1/2 GPU bên trên. Đây là engine nội bộ mỗi
rank, không gọi HTTP host và không dùng causal-only trainer vLLM mặc định.
Adapter hiện chỉ hỗ trợ decoder tương thích mapping Qwen3 của SEAM vLLM;
native có phạm vi backbone rộng hơn.

Mỗi phiên bản actor đồng bộ đầy đủ encoder, bridge, decoder/cross-attention,
copy sang engine trước rollout. Multimodal processor cache tắt để không tái
dùng states từ trọng số cũ. Mỗi request mang source, prompt encoder và token
prompt decoder thực tế riêng. Có kiểm tra sai lệch log-prob native/vLLM trước
backward; vượt tolerance thì dừng. Cần kiểm tra trên GPU server trước khi tin
vào throughput/parity. Có thể cần giảm memory utilization hoặc microbatch vì
actor, optimizer và engine cùng nằm trên mỗi GPU.

## Checkpoint và validation

Lưu `last.pt`, `epoch_XXX.pt` và `best.pt` theo định dạng SEAM hiện có, kèm
`resolved_config.yaml`, `length_grpo_config.yaml`, `best_selection.json`, logs
và prediction validation. Best ưu tiên tất cả domain có mean length cost ≤1,
rồi chọn macro Q cao nhất. Nếu chưa có checkpoint nào đạt length thì chọn
worst-domain violation nhỏ nhất; `best_selection.json` ghi `feasible:false`.
Không gọi fallback đó là đã đạt chỉ tiêu. Baseline trước GRPO cũng được xét:
đó là `length_ce_baseline` sau warm-up, hoặc `sft_baseline` nếu tắt warm-up.
Việc chọn best này không tự chứng minh vượt checkpoint SFT gốc trước thích nghi;
cần đánh giá checkpoint gốc riêng dưới cùng giao thức yêu cầu số từ.

Đây không phải cơ chế resume chính xác optimizer/RNG/rollout của TRL. Có thể
dùng trọng số export làm khởi đầu của **run mới**, lịch cosine và step bắt đầu
lại; không mô tả điều đó là tiếp tục nguyên trạng training cũ.

## Evaluate có yêu cầu số từ

Dùng entrypoint riêng để bảo đảm prompt khi eval khớp lúc train. Mỗi dòng test
có `target_words` thì không cần tùy chọn thêm:

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/seam_length_grpo/evaluate.py \
  --config runs/seam_length_grpo/dual_prompt/length_grpo_config.yaml \
  --checkpoint runs/seam_length_grpo/dual_prompt/best.pt \
  --output runs/seam_length_grpo/dual_prompt/test_predictions.jsonl \
  --split test --batch-size 8
```

Thêm `--target-words 100` nếu request bên ngoài là 100 từ cho mọi sample. Có thể
dùng `--input /path/test_requests.jsonl` để đổi file test mà không sửa YAML.
Nếu benchmark cung cấp số từ tham chiếu và bạn chủ động đánh giá theo giao thức
đó, thêm **`--oracle-reference-length`**; prediction ghi rõ nguồn W là
`reference_length`. Mặc định test thiếu W sẽ báo lỗi, không ngầm lấy từ gold.

Để chia test trên hai GPU, thay entrypoint bằng
`CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash src/seam_length_grpo/scripts/evaluate.sh`
và giữ các arguments. Mỗi GPU sinh một shard, sau đó merge theo đúng thứ tự
input. Ghi/flush sau mỗi sample đã hoàn tất batch, không đợi hết dataset.
Không ghi đè prediction đã có trừ khi truyền `--overwrite`; chưa có resume
prediction. `.metrics.json` chứa MAE/ROUGE proxy theo dataset. Mang JSONL cuối
sang evaluator ROUGE-1.5.5 và length chính thức để đối chiếu chỉ tiêu.

## Kiểm thử và cơ sở thiết kế

Adapter dùng cùng [conditional-policy contract](../seam_grpo/README.md#verification-and-evidence)
với bản GRPO cơ bản: likelihood đã trộn copy, gradient qua toàn bộ actor,
reference tính lại encoder/bridge riêng. Kiểm thử TRL còn bật đồng thời
`beta=0.01`, source teacher và warm-up CE để xác nhận KL reference và teacher
có thể chia sẻ bản SFT đóng băng, trong khi actor tiếp tục cập nhật độc lập.

Không cần tải backbone để chạy checks. Trên Mac dùng đúng môi trường của dự án:

```bash
/Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q src/seam_length_grpo/checks
```

Xem `VERIFICATION.md` cho phạm vi đã kiểm chứng và giới hạn CUDA/vLLM.
Cơ sở nghiên cứu, phân biệt phương pháp đã có với phần đề xuất cho SEAM, và
các nguồn được lưu tại `doc/SEAM_LENGTH_RESEARCH_20261017/`.
