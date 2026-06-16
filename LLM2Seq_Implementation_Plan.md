# Kế hoạch triển khai chi tiết: LLM2Vec Encoder + Lightweight Decoder

## 0. Tên tạm thời

**LLM2Seq**: Converting LLM2Vec Encoders into Lightweight Encoder-Decoder Generators.

Mục tiêu là xây dựng một mô hình encoder-decoder mới, trong đó:

- **Encoder**: dùng LLM2Vec hoặc một LLM đã được chuyển sang dạng encoder để hiểu input.
- **Adaptor**: chuyển hidden states của encoder sang đúng kích thước decoder.
- **Decoder nhỏ**: tự thiết kế Transformer decoder nhẹ để sinh output.
- **MTP**: có thể bật/tắt để tăng tốc sinh nhiều token.
- **Distillation**: có thể bật/tắt để học theo teacher LLM hoặc teacher model mạnh hơn.

Ý tưởng cốt lõi:

```text
Input x
  -> LLM2Vec Encoder: H_enc
  -> Adaptor: H_dec_memory
  -> Lightweight Decoder: P(y_t | y_<t, x)
  -> Optional MTP Heads / MTP Modules
  -> Output y
```

---

## 1. Mục tiêu nghiên cứu

### 1.1. Bài toán chính

Thay vì dùng decoder-only LLM để vừa đọc input vừa sinh output, ta tách mô hình thành hai phần:

1. **LLM2Vec Encoder** chịu trách nhiệm hiểu input.
2. **Small Decoder** chịu trách nhiệm sinh output từng token.

Thiết kế này phù hợp với các bài toán dạng sequence-to-sequence:

- Machine Translation.
- Summarization.
- Title generation.
- Post-editing.
- Code/documentation generation ngắn.
- Any text-to-text task cần output không quá dài.

### 1.2. Giả thuyết chính

Mô hình encoder-decoder với encoder mạnh và decoder nhỏ có thể:

- Giữ chất lượng gần với LLM teacher.
- Giảm chi phí decoding vì decoder nhỏ hơn LLM gốc.
- Giảm KV-cache trong quá trình sinh.
- Dễ thêm MTP vì decoder nhỏ dễ sửa kiến trúc hơn full LLM.
- Dễ kiểm soát ablation nhờ có flag bật/tắt MTP và Distillation.

---

## 2. Kiến trúc tổng thể

### 2.1. Sơ đồ hệ thống

```text
                +----------------------+
Input tokens -> |   LLM2Vec Encoder    |
                | token hidden states  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |       Adaptor        |
                | Layer fusion / MLP   |
                | Optional EncStack    |
                +----------+-----------+
                           |
                           v
Target prefix ->+----------------------+
y_<t            | Lightweight Decoder  |
                | Causal self-attn     |
                | Cross-attn to memory |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Main LM Head         |
                | Optional MTP Heads   |
                +----------+-----------+
                           |
                           v
                        Output y
```

### 2.2. Encoder

Encoder nhận input `x` và trả về token-level hidden states:

```python
H_enc = encoder(input_ids=x, attention_mask=x_mask, output_hidden_states=True)
```

Yêu cầu quan trọng:

- Không chỉ lấy sentence embedding cuối cùng.
- Phải lấy **token-level representations** để decoder cross-attention vào từng vị trí input.
- Nếu encoder trả về nhiều tầng hidden states, có thể dùng layer fusion.

Output mong muốn:

```text
H_enc: [batch_size, src_len, d_enc]
```

### 2.3. Adaptor

Adaptor chuyển hidden states từ encoder sang decoder memory.

Cấu hình đơn giản:

```text
H_adapt = MLP(LayerNorm(H_enc))
```

Cấu hình đầy đủ:

```text
H_fuse  = LayerFusion(hidden_states_from_encoder)
H_mlp   = MLP(H_fuse)
H_mem   = OptionalEncStack(H_mlp)
```

Kích thước:

```text
H_enc: [B, S, d_enc]
H_mem: [B, S, d_dec]
```

Gợi ý ban đầu:

- `d_dec = 768` nếu decoder nhỏ.
- `num_decoder_layers = 6`.
- `num_attention_heads = 12`.
- `ffn_dim = 3072`.
- `dropout = 0.1`.

### 2.4. Lightweight Decoder

Decoder là Transformer decoder chuẩn:

Mỗi layer gồm:

1. Causal self-attention trên target prefix `y_<t`.
2. Cross-attention vào `H_mem`.
3. FFN.
4. Residual + LayerNorm/RMSNorm.

Công thức:

```text
H_mem = Adaptor(LLM2VecEncoder(x))
S_t = Decoder(y_<t, H_mem)
P_main(y_t | y_<t, x) = Softmax(W_out S_t)
```

### 2.5. Output Head

Có hai lựa chọn:

- **Shared tokenizer**: dùng tokenizer của LLM2Vec/LLM gốc cho cả encoder và decoder.
- **Separate tokenizer**: encoder và decoder dùng tokenizer khác nhau.

Khuyến nghị triển khai trước:

```text
Dùng chung tokenizer để đơn giản hóa training, distillation và evaluation.
```

---

## 3. Các chế độ bật/tắt

Toàn bộ hệ thống phải có thể chạy theo 4 cấu hình chính:

| Cấu hình | MTP | Distillation | Mục đích |
|---|---:|---:|---|
| Baseline | Tắt | Tắt | Đo chất lượng encoder-decoder cơ bản |
| KD-only | Tắt | Bật | Đo lợi ích từ teacher LLM |
| MTP-only | Bật | Tắt | Đo MTP có giúp tốc độ/chất lượng không |
| KD + MTP | Bật | Bật | Cấu hình đầy đủ |

### 3.1. Config YAML đề xuất

```yaml
project:
  name: llm2seq
  task: translation
  output_dir: runs/llm2seq_exp01

model:
  encoder_name: llm2vec-base
  encoder_trainable: false
  use_lora_for_encoder: false
  d_enc: 4096
  d_dec: 768

adaptor:
  type: mlp
  use_layer_fusion: true
  fuse_layers: [-1, -4, -8, -12]
  use_encstack: false
  encstack_layers: 2

small_decoder:
  num_layers: 6
  hidden_size: 768
  num_heads: 12
  ffn_size: 3072
  dropout: 0.1
  tie_embeddings: true

features:
  use_mtp: false
  use_distillation: false

mtp:
  type: cascaded        # options: parallel, cascaded
  num_heads: 4
  loss_weight: 0.3
  head_weights: [1.0, 0.8, 0.6, 0.4]
  use_mtp_at_inference: false
  inference_strategy: confidence_adaptive
  confidence_threshold: 0.9

knowledge_distillation:
  teacher_name: llama-or-qwen-teacher
  type: topk_kl         # options: logits_kl, topk_kl, sequence_kd
  temperature: 2.0
  loss_weight: 0.5
  top_k: 10000
  cache_teacher_logits: true
  detach_teacher: true

training:
  stage: baseline
  batch_size: 16
  grad_accum_steps: 8
  learning_rate: 2.0e-4
  decoder_lr: 2.0e-4
  adaptor_lr: 2.0e-4
  encoder_lr: 1.0e-5
  warmup_steps: 1000
  max_steps: 50000
  fp16: true
  bf16: false
  gradient_checkpointing: true

evaluation:
  metrics: [bleu, comet, rouge, latency, memory, tokens_per_second]
  eval_every_steps: 1000
  save_every_steps: 2000
```

---

## 4. Thiết kế MTP

MTP có hai hướng triển khai. Nên code cả hai dưới dạng module, nhưng ưu tiên chạy `parallel` trước vì dễ debug.

### 4.1. Parallel MTP Heads

Mỗi head dự đoán một token tương lai từ cùng hidden state decoder.

```text
Main head: dự đoán y_t
MTP head 1: dự đoán y_{t+1}
MTP head 2: dự đoán y_{t+2}
...
MTP head K: dự đoán y_{t+K}
```

Forward:

```python
main_logits = lm_head(decoder_states)
mtp_logits = []
for k in range(num_mtp_heads):
    mtp_logits_k = mtp_heads[k](decoder_states)
    mtp_logits.append(mtp_logits_k)
```

Loss:

```text
L_MTP = Σ_k α_k CE(P_k, y_{t+k})
```

Ưu điểm:

- Dễ code.
- Dễ bật/tắt.
- Dễ đo ablation.

Nhược điểm:

- Các token tương lai được dự đoán tương đối độc lập.
- Không giữ causal chain tốt bằng cascaded MTP.

### 4.2. Cascaded / Sequential MTP

Mỗi depth nhận thêm thông tin từ token tương lai trước đó, giúp giữ quan hệ nhân quả tốt hơn.

Ý tưởng:

```text
Depth 0: decoder state chính -> y_t
Depth 1: combine hidden depth 0 + embedding(y_t) -> y_{t+1}
Depth 2: combine hidden depth 1 + embedding(y_{t+1}) -> y_{t+2}
...
```

Pseudo-code:

```python
h_prev = decoder_states
mtp_logits = []
for k in range(num_mtp_heads):
    future_emb = target_embedding(shifted_target_ids[:, k:])
    h_input = concat_project(h_prev, future_emb)
    h_k = mtp_blocks[k](h_input)
    logits_k = shared_lm_head(h_k)
    mtp_logits.append(logits_k)
    h_prev = h_k
```

Loss:

```text
L_MTP = Σ_k α_k CE(P_mtp_k, y_{t+k})
```

Khuyến nghị:

- Phase MVP: dùng `parallel`.
- Phase paper-ready: dùng `cascaded`.

---

## 5. Thiết kế Distillation

Distillation có thể dùng theo ba mức.

### 5.1. Sequence-level Distillation

Teacher sinh output trước, sau đó student học theo output đó.

```text
x -> teacher -> y_teacher
student học (x, y_teacher)
```

Ưu điểm:

- Dễ triển khai.
- Không cần lưu full logits.
- Phù hợp nếu tài nguyên hạn chế.

Nhược điểm:

- Mất thông tin phân phối xác suất của teacher.

### 5.2. Logits KL Distillation

Teacher và student cùng chạy trên input-output prefix, sau đó tính KL giữa phân phối token.

```text
L_KD = KL(P_teacher(. | prefix), P_student(. | prefix))
```

Công thức:

```text
P_T = softmax(z_T / T)
P_S = log_softmax(z_S / T)
L_KD = T^2 * KL(P_T || P_S)
```

Trong đó `T` là temperature.

### 5.3. Top-k KL Distillation

Chỉ distill top-k logits của teacher để giảm bộ nhớ.

```python
topk_idx = teacher_logits.topk(k=top_k, dim=-1).indices
teacher_topk = gather(teacher_logits, topk_idx)
student_topk = gather(student_logits, topk_idx)
loss_kd = KLDiv(log_softmax(student_topk / T), softmax(teacher_topk / T))
```

Khuyến nghị:

- MVP: sequence-level distillation.
- Bản mạnh hơn: top-k KL distillation.
- Luôn detach teacher logits.
- Có thể cache teacher logits nếu dataset nhỏ.

---

## 6. Tổng loss và cơ chế bật/tắt

### 6.1. Loss tổng quát

```text
L_total = L_CE
        + λ_KD  * I_distill * L_KD
        + λ_MTP * I_mtp     * L_MTP
```

Trong đó:

```text
I_distill = 1 nếu use_distillation = true, ngược lại 0
I_mtp     = 1 nếu use_mtp = true, ngược lại 0
```

### 6.2. Pseudo-code tính loss

```python
loss = ce_loss(main_logits, labels)

if cfg.features.use_distillation:
    loss_kd = compute_kd_loss(
        student_logits=main_logits,
        teacher_logits=batch["teacher_logits"],
        temperature=cfg.knowledge_distillation.temperature,
        top_k=cfg.knowledge_distillation.top_k,
    )
    loss = loss + cfg.knowledge_distillation.loss_weight * loss_kd

if cfg.features.use_mtp:
    loss_mtp = compute_mtp_loss(
        mtp_logits=outputs.mtp_logits,
        labels=labels,
        head_weights=cfg.mtp.head_weights,
    )
    loss = loss + cfg.mtp.loss_weight * loss_mtp
```

---

## 7. Training pipeline

### Stage 0: Chuẩn bị dữ liệu

Dữ liệu chuẩn dạng JSONL:

```json
{"id": "0001", "source": "...", "target": "...", "task": "translation"}
{"id": "0002", "source": "...", "target": "...", "task": "summarization"}
```

Tiền xử lý:

1. Tokenize source bằng encoder tokenizer.
2. Tokenize target bằng decoder tokenizer.
3. Tạo `labels` bằng target ids shifted.
4. Nếu bật sequence distillation, thêm `teacher_target`.
5. Nếu bật cached logits, thêm đường dẫn tới file logits.

### Stage 1: Baseline encoder-decoder

Config:

```yaml
features:
  use_mtp: false
  use_distillation: false
model:
  encoder_trainable: false
```

Train:

- Freeze encoder.
- Train adaptor + decoder.
- Loss chỉ gồm CE.

Mục tiêu:

- Kiểm tra mô hình chạy đúng.
- So sánh với Transformer/NMT baseline.
- Đảm bảo decoder học được từ encoder memory.

### Stage 2: Thêm Distillation

Config:

```yaml
features:
  use_mtp: false
  use_distillation: true
knowledge_distillation:
  type: topk_kl
  loss_weight: 0.3
```

Train:

- Freeze encoder ở giai đoạn đầu.
- Train adaptor + decoder.
- Loss = CE + KD.

Mục tiêu:

- Tăng chất lượng generation.
- Giảm khoảng cách giữa small decoder và teacher LLM.

### Stage 3: Thêm MTP

Config:

```yaml
features:
  use_mtp: true
  use_distillation: false
mtp:
  type: parallel
  num_heads: 2
  loss_weight: 0.2
```

Train:

- Load checkpoint baseline từ Stage 1.
- Thêm MTP heads/modules.
- Train decoder + MTP heads.
- Có thể freeze main decoder vài nghìn step đầu để tránh làm hỏng main head.

Mục tiêu:

- Đo acceptance rate.
- Đo speedup khi dùng MTP inference.
- Đảm bảo main CE không giảm quá mạnh.

### Stage 4: Full model KD + MTP

Config:

```yaml
features:
  use_mtp: true
  use_distillation: true
mtp:
  type: cascaded
  num_heads: 4
  loss_weight: 0.3
knowledge_distillation:
  type: topk_kl
  loss_weight: 0.5
```

Train:

- Load checkpoint KD-only hoặc MTP-only tốt nhất.
- Bật cả KD và MTP.
- Fine-tune toàn bộ adaptor + decoder + MTP.
- Chỉ unfreeze encoder bằng LoRA nếu baseline đã ổn.

Mục tiêu:

- Cấu hình mạnh nhất.
- So sánh chất lượng và tốc độ với các baseline.

### Stage 5: Optional encoder LoRA fine-tuning

Config:

```yaml
model:
  encoder_trainable: true
  use_lora_for_encoder: true
training:
  encoder_lr: 1.0e-5
```

Chỉ nên làm khi:

- Adaptor + decoder đã học ổn.
- Có đủ GPU.
- Cần cải thiện domain-specific task.

---

## 8. Repo structure đề xuất

```text
llm2seq/
  configs/
    baseline.yaml
    kd_only.yaml
    mtp_only.yaml
    kd_mtp_full.yaml

  data/
    raw/
    processed/
    teacher_outputs/

  src/
    models/
      llm2seq_model.py
      encoder_wrapper.py
      adaptor.py
      decoder.py
      mtp_heads.py
      mtp_cascaded.py

    training/
      trainer.py
      losses.py
      kd_loss.py
      mtp_loss.py
      scheduler.py

    inference/
      generate.py
      generate_mtp.py
      speculative_verify.py
      confidence_adaptive.py

    data/
      dataset.py
      collator.py
      preprocess.py
      build_teacher_cache.py

    eval/
      eval_bleu.py
      eval_comet.py
      eval_latency.py
      eval_acceptance.py

  scripts/
    train_baseline.sh
    train_kd.sh
    train_mtp.sh
    train_full.sh
    eval_all.sh

  README.md
```

---

## 9. Interface model forward

### 9.1. Input

```python
batch = {
    "input_ids": Tensor[B, S_src],
    "attention_mask": Tensor[B, S_src],
    "decoder_input_ids": Tensor[B, S_tgt],
    "decoder_attention_mask": Tensor[B, S_tgt],
    "labels": Tensor[B, S_tgt],
    "teacher_logits": Optional[Tensor[B, S_tgt, K_or_V]],
    "teacher_topk_indices": Optional[Tensor[B, S_tgt, K]],
}
```

### 9.2. Output

```python
outputs = {
    "logits": Tensor[B, S_tgt, vocab_size],
    "loss": Optional[Tensor],
    "loss_ce": Optional[Tensor],
    "loss_kd": Optional[Tensor],
    "loss_mtp": Optional[Tensor],
    "mtp_logits": Optional[List[Tensor[B, S_tgt, vocab_size]]],
}
```

### 9.3. Forward pseudo-code

```python
class LLM2Seq(nn.Module):
    def forward(self, input_ids, attention_mask,
                decoder_input_ids, decoder_attention_mask,
                labels=None, teacher_logits=None,
                teacher_topk_indices=None):

        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        H_mem = self.adaptor(enc_out.hidden_states)

        dec_states = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=H_mem,
            encoder_attention_mask=attention_mask,
        )

        main_logits = self.lm_head(dec_states)

        mtp_logits = None
        if self.cfg.features.use_mtp:
            mtp_logits = self.mtp_module(
                decoder_states=dec_states,
                decoder_input_ids=decoder_input_ids,
            )

        loss_dict = {}
        if labels is not None:
            loss = compute_total_loss(
                main_logits=main_logits,
                labels=labels,
                mtp_logits=mtp_logits,
                teacher_logits=teacher_logits,
                teacher_topk_indices=teacher_topk_indices,
                cfg=self.cfg,
            )
            loss_dict = loss

        return {
            "logits": main_logits,
            "mtp_logits": mtp_logits,
            **loss_dict,
        }
```

---

## 10. Inference

### 10.1. Không dùng MTP

Sinh autoregressive bình thường:

```text
for t in range(max_len):
    logits = model(x, y_<t)
    y_t = sample_or_argmax(logits[-1])
    append(y_t)
```

### 10.2. Dùng MTP đơn giản

MTP heads đề xuất nhiều token tương lai.

```text
main head -> y_t
mtp head 1 -> y_{t+1}
mtp head 2 -> y_{t+2}
...
```

Có hai hướng accept:

1. **Static accept**: lấy cố định k token nếu confidence đủ cao.
2. **Confidence-adaptive**: chỉ nhận các token có xác suất cao hơn threshold.

Pseudo-code:

```python
tokens = []
while len(tokens) < max_len:
    outputs = model.forward_with_mtp(x, tokens)
    candidates = [main_token] + mtp_tokens

    accepted = []
    for token, conf in candidates:
        if conf >= cfg.mtp.confidence_threshold:
            accepted.append(token)
        else:
            break

    if len(accepted) == 0:
        accepted = [main_token]

    tokens.extend(accepted)
```

### 10.3. MTP có verification

Nếu muốn nghiêm ngặt hơn:

1. MTP module sinh draft tokens.
2. Main decoder verify lại draft tokens.
3. Chỉ accept prefix hợp lệ.

Mục tiêu là giữ chất lượng gần main decoder, đổi lại inference phức tạp hơn.

---

## 11. Evaluation plan

### 11.1. Chất lượng

Tùy task:

- Translation: BLEU, chrF, COMET.
- Summarization: ROUGE-1/2/L, BERTScore.
- Title generation: ROUGE-L, BLEU, human evaluation.
- General seq2seq: exact match, F1, task-specific metric.

### 11.2. Tốc độ

Cần đo:

- Latency trung bình / sample.
- Tokens per second.
- Peak GPU memory.
- KV-cache memory.
- Số decoding steps trung bình.
- Average accepted tokens per step.

### 11.3. MTP-specific metrics

```text
Acceptance Rate (AR): tỷ lệ token MTP được chấp nhận.
Cumulative Acceptance Rate (CAR): tỷ lệ chấp nhận liên tiếp đến head k.
Average accepted length: trung bình số token được sinh trong một decoding step.
Speedup ratio: tốc độ so với không dùng MTP.
```

### 11.4. Ablation bắt buộc

| ID | Encoder | Decoder | MTP | KD | Ghi chú |
|---|---|---|---|---|---|
| A0 | Transformer encoder | small decoder | off | off | NMT baseline |
| A1 | LLM2Vec frozen | small decoder | off | off | baseline chính |
| A2 | LLM2Vec frozen | small decoder | off | on | KD-only |
| A3 | LLM2Vec frozen | small decoder | on | off | MTP-only |
| A4 | LLM2Vec frozen | small decoder | on | on | full model |
| A5 | LLM2Vec LoRA | small decoder | on | on | full + encoder tuning |
| A6 | LLM teacher direct generation | none | none | none | teacher reference |

---

## 12. Lệnh chạy đề xuất

### 12.1. Baseline

```bash
python -m src.training.trainer \
  --config configs/baseline.yaml
```

### 12.2. Build teacher cache

```bash
python -m src.data.build_teacher_cache \
  --config configs/kd_only.yaml \
  --input data/processed/train.jsonl \
  --output data/teacher_outputs/train_topk_logits
```

### 12.3. KD-only

```bash
python -m src.training.trainer \
  --config configs/kd_only.yaml \
  --resume runs/baseline/best.pt
```

### 12.4. MTP-only

```bash
python -m src.training.trainer \
  --config configs/mtp_only.yaml \
  --resume runs/baseline/best.pt
```

### 12.5. Full KD + MTP

```bash
python -m src.training.trainer \
  --config configs/kd_mtp_full.yaml \
  --resume runs/kd_only/best.pt
```

### 12.6. Evaluation

```bash
python -m src.eval.eval_all \
  --checkpoint runs/kd_mtp_full/best.pt \
  --test_file data/processed/test.jsonl \
  --use_mtp true
```

---

## 13. Lộ trình triển khai

### Tuần 1: MVP chạy được

- Tạo repo.
- Implement dataset + collator.
- Implement encoder wrapper.
- Implement adaptor MLP.
- Implement small decoder.
- Train baseline CE trên dataset nhỏ.
- Generate được output đầu tiên.

Kết quả cần có:

```text
Baseline chạy end-to-end, loss giảm, output không rỗng.
```

### Tuần 2: Baseline nghiêm túc

- Train baseline đủ lâu.
- Đánh giá BLEU/ROUGE/metric chính.
- Đo latency và memory.
- So sánh với Transformer hoặc T5/mBART nhỏ.

Kết quả cần có:

```text
Bảng A0 vs A1.
```

### Tuần 3: Distillation

- Implement sequence KD trước.
- Sau đó implement top-k KL KD.
- Cache teacher outputs/logits.
- Train KD-only.

Kết quả cần có:

```text
Bảng A1 vs A2, chứng minh KD giúp hoặc không giúp.
```

### Tuần 4: MTP

- Implement parallel MTP.
- Train MTP-only.
- Implement confidence-adaptive inference.
- Đo AR/CAR/speedup.

Kết quả cần có:

```text
Bảng A1 vs A3 về quality, speed, accepted length.
```

### Tuần 5: Full model

- Bật KD + MTP.
- Chạy full ablation.
- Thử cascaded MTP nếu parallel chạy ổn.

Kết quả cần có:

```text
Bảng A1/A2/A3/A4 đầy đủ.
```

### Tuần 6: Viết báo cáo/paper draft

- Viết method.
- Vẽ architecture.
- Vẽ training stages.
- Tạo bảng ablation.
- Phân tích failure cases.

---

## 14. Rủi ro và cách xử lý

### 14.1. Decoder nhỏ học chậm

Nguyên nhân:

- Decoder random init.
- Encoder hidden states khó dùng trực tiếp.

Cách xử lý:

- Freeze encoder trước.
- Train adaptor + decoder lâu hơn.
- Dùng layer fusion.
- Thêm EncStack 1-2 layer.
- Khởi tạo decoder từ một LM nhỏ nếu có thể.

### 14.2. Chỉ dùng sentence embedding làm mất thông tin

Không nên:

```text
sentence_embedding -> decoder
```

Nên dùng:

```text
token_hidden_states -> cross-attention memory
```

### 14.3. Distillation quá tốn bộ nhớ

Cách xử lý:

- Dùng sequence distillation.
- Dùng top-k logits thay vì full vocab.
- Cache logits offline.
- Chỉ distill một phần batch.

### 14.4. MTP làm giảm main-head performance

Cách xử lý:

- Giảm `mtp.loss_weight`.
- Warmup MTP heads riêng.
- Freeze main decoder vài nghìn step đầu khi train MTP.
- Dùng head weights giảm dần: `[1.0, 0.8, 0.6, 0.4]`.
- Thêm KD để align MTP với main head/teacher.

### 14.5. MTP inference không tăng tốc thật

Nguyên nhân:

- Acceptance rate thấp.
- Overhead MTP lớn hơn lợi ích.
- Batch size/implementation chưa tối ưu.

Cách xử lý:

- Đo cả acceleration factor và wall-time latency.
- Bắt đầu với 2 MTP heads.
- Dùng confidence threshold cao.
- Chỉ bật MTP ở greedy/low-temperature decoding.

---

## 15. Tiêu chí thành công

### MVP thành công nếu:

- Baseline encoder-decoder train được.
- Loss giảm ổn định.
- Output sinh ra hợp lệ.
- Chạy được 4 cấu hình: baseline, KD-only, MTP-only, KD+MTP.

### Research result tốt nếu:

- KD-only cải thiện chất lượng so với baseline.
- MTP-only tăng tốc mà chất lượng giảm nhỏ.
- KD+MTP có trade-off tốt nhất.
- Mô hình full nhanh hơn decoder-only teacher trong decoding.
- Memory/KV-cache thấp hơn direct LLM generation.

### Paper-level novelty mạnh nếu chứng minh được:

```text
LLM2Vec không chỉ dùng cho embedding/retrieval, mà còn có thể làm token-level encoder cho generation.
Khi kết hợp với small decoder, distillation và optional MTP, mô hình đạt trade-off tốt giữa quality và inference efficiency.
```

---

## 16. Kết luận triển khai

Thứ tự nên làm:

1. Làm baseline LLM2Vec Encoder + MLP Adaptor + Small Decoder.
2. Bật Distillation trước vì dễ cải thiện chất lượng.
3. Bật MTP sau vì liên quan trực tiếp đến inference và acceptance rate.
4. Cuối cùng mới thử cascaded MTP và LoRA encoder.

Cấu hình nên dùng để bắt đầu:

```yaml
features:
  use_mtp: false
  use_distillation: false
model:
  encoder_trainable: false
adaptor:
  type: mlp
small_decoder:
  num_layers: 6
  hidden_size: 768
```

Cấu hình mục tiêu cuối:

```yaml
features:
  use_mtp: true
  use_distillation: true
mtp:
  type: cascaded
  num_heads: 4
knowledge_distillation:
  type: topk_kl
```
