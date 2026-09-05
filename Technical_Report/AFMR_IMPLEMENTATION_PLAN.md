# Kế hoạch triển khai EviSeq-AFMR

**Tên kiến trúc:** EviSeq-AFMR — Adaptive Full-Memory Residual Bridge

**Trạng thái:** đặc tả đã chốt để triển khai

**Ngày:** 2026-09-05

**Quyết định cập nhật cuối:** token-wise depth readout, CE-only. Đã loại bỏ hoàn toàn allocation loss, positive mining, metadata và diagnostic reductions không dùng khỏi code/config. Các phần allocation trong mục 1–27 là lịch sử thiết kế, không còn là yêu cầu triển khai. Xem mục 28 và README hiện tại để chạy.

**Mục tiêu chính:** ghép một encoder pretrained và một decoder pretrained thành mô hình tóm tắt tổng quát, giữ tổng tham số thấp hơn T5Gemma2-1B-1B và vượt T5Gemma trên ít nhất một trong ba chỉ số ROUGE-1/2/L trong đánh giá greedy công bằng.

Tài liệu này là hợp đồng giữa thiết kế, code, thí nghiệm và bài báo. Nếu code khác tài liệu này, phải sửa code hoặc cập nhật đặc tả bằng một quyết định có lý do; không thêm nhánh, loss hay chế độ tương thích cũ theo kiểu chắp vá.

---

## 1. Quyết định cuối cùng

Kiến trúc mới được chốt theo một câu chuyện duy nhất:

> Encoder và decoder pretrained đã có năng lực biểu diễn mạnh. Nút thắt của việc ghép hai LLM không phải là thiếu thêm một encoder, memory bank hay contrastive head, mà là phân bổ đúng ngân sách thông tin tại interface. AFMR giữ nguyên toàn bộ source memory và chỉ học ba phép điều chỉnh residual bị chặn: chọn độ sâu encoder, hiệu chỉnh không gian đặc trưng, và phân bổ ưu tiên theo nhiều độ phân giải nguồn.

Ba trục thích nghi là:

1. **Depth:** mỗi token cần đọc representation ở tầng encoder nào, kết hợp context của tài liệu.
2. **Channel:** những chiều nào cần hiệu chỉnh khi chuyển từ không gian encoder sang decoder.
3. **Span:** mức chi tiết nào của nguồn — local, đoạn trung bình hay vùng dài — cần được ưu tiên.

Đường chính luôn là final encoder state → full source memory → cross-attention của decoder. Mọi khối mới đều là residual nhỏ, normalized, bounded và có thể trở về gần baseline lúc khởi tạo.

### 1.1. Những thứ có trong cấu hình chính

- Một encoder pretrained.
- Một decoder pretrained.
- Một source memory duy nhất, giữ mọi token nguồn còn visible sau truncation.
- Token-wise depth-selective residual readout trên bốn hidden states cuối của encoder.
- Identity-preserving cross-space feature residual.
- Prompt-, source- và budget-conditioned multi-scale focus prior.
- Một bias nguồn dùng chung cho mọi decoder cross-attention layer.
- Main loss chỉ là token cross-entropy; allocation loss trên prior chỉ bật trong ablation riêng.
- Một epoch interface warm-up và bốn epoch full fine-tuning.
- Greedy decoding, một lần chạy encoder, KV cache chuẩn.

### 1.2. Những thứ cố ý không có

- Không dual source memory, HiRoute bank hay per-layer router.
- Không evidence slot cố định, không CoVeR/PCEB slot semantics.
- Không decoder probe để tạo query evidence.
- Không hard top-k hoặc bỏ token nguồn.
- Không document-level InfoNCE head tách khỏi generation path.
- Không salience CE, diversity, alignment, source-swap, geometry và nhiều auxiliary loss chồng lên nhau.
- Không candidate generation/ranking phase.
- Không KD, RL, DPO, MTP, EAGLE hay MoE trong kiến trúc chính.
- Không beam search và không sampling khi báo kết quả chính.
- Không fingerprint, checksum hay logic khóa split dựa trên hash.
- Không trường benchmark, claim, footprint hoặc reporting trong YAML huấn luyện.

---

## 2. Cơ sở từ các technical report mới

AFMR đã học các nguyên lý phù hợp nhất từ các report trong thư mục này, nhưng không bê nguyên khối của model lớn. Bản tổng hợp nguồn đầy đủ nằm ở [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md).

| Technical report | Quan sát gốc | Chuyển thành AFMR | Không chuyển trực tiếp |
|---|---|---|---|
| Kimi K3 | Attention Residuals cho phép đọc representation theo chiều sâu thay vì residual accumulation cố định | Depth-selective residual readout chạy một lần ở interface, có final-state anchor | KDA, MLA, NoPE, AttnRes xuyên toàn backbone, LatentMoE |
| DeepSeek-V4 | mHC bảo vệ main path bằng residual mixing có ràng buộc; CSA dùng coarse retrieval trước fine attention | Gate bounded, normalized residual; coarse-to-fine trở thành soft multi-scale prior | mHC đa stream, CSA/HCA sparse decoder, hard top-k, MoE, MTP |
| Qwen3.8-Flash-Next | GR cho thấy feature-wise read hữu ích; QSA cho thấy coarse-to-fine retrieval; sparse branch read có thể giữ loss nhưng giảm downstream quality | Feature-wise residual correction và soft source prior trong khi vẫn giữ full memory | GDN/QSA native, sparse source pruning, GR đa branch toàn decoder, n-gram table |
| Instella-MoE | Gated MLA dùng input-conditioned channel gate và normalized attention output | Channel-wise gate cho cross-space residual | MLA/MoE/FarSkip, speculative hoặc system-only optimization |

### 2.1. Cách phát biểu trong paper

Được phép viết:

> AFMR is motivated by recent advances in depth-selective access, constrained residual routing, and coarse-to-fine retrieval. We adapt these principles at the encoder–decoder interface while retaining a single full source memory.

Không được viết:

- “EviSeq implements Kimi Attention Residuals.”
- “EviSeq uses DeepSeek mHC.”
- “EviSeq adopts QSA/CSA.”
- “Các report chứng minh AFMR tăng ROUGE.”

Report là cơ sở hình thành giả thuyết. Chỉ ablation của EviSeq mới là bằng chứng cho hiệu quả tóm tắt.

---

## 3. Vấn đề mà kiến trúc phải giải

Kết quả PubMed gần nhất cho thấy khoảng cách đã nhỏ:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| T5Gemma2-1B-1B | 49.580 | 21.990 | 45.463 |
| Legacy dual-bridge EviSeq | 49.176 | 21.590 | 45.283 |
| Khoảng cách | -0.404 | -0.400 | -0.180 |

Khoảng cách này không biện minh cho việc thêm một hệ nhiều bank hoặc nhiều loss. Ba giả thuyết về hạn chế của interface cần được kiểm chứng bằng ablation, không thể suy ra nguyên nhân từ ROUGE đơn lẻ:

1. **Representation depth bị cố định hoặc trộn không có anchor rõ.** Final layer có thể quá task-specific cho encoder gốc, còn mean nhiều layer làm loãng thông tin.
2. **PPLX-space và Qwen-space chưa được hiệu chỉnh có kiểm soát.** Một full projection tự do dễ phá representation; identity thuần lại bắt decoder tự gánh mismatch.
3. **Evidence routing cũ không trùng với phép tính quyết định generation.** Một contrastive head có thể giảm loss hoặc tăng accuracy mà không làm decoder cross-attend đúng hơn. Slot cố định cũng đưa giả định domain-specific vào kiến trúc.

AFMR được thiết kế để kiểm nghiệm ba giả thuyết trên; chưa có bằng chứng thực nghiệm rằng mỗi thành phần mới cải thiện ROUGE.

---

## 4. Bất biến hệ thống

Các điều kiện sau phải được kiểm tra bằng assertion hoặc test, không chỉ ghi trong README:

1. Source memory đưa vào decoder có chiều dài đúng bằng số token nguồn visible; không có top-k pruning.
2. Chỉ có một tensor memory `[batch, source_length, decoder_hidden]`.
3. Bridge chạy đúng một lần sau mỗi encoder forward.
4. Source bias chạy đúng một lần và được dùng chung ở mọi cross-attention layer.
5. Forward tạo memory và bias không đọc target tokens, target length, reference summary hoặc evidence labels.
6. Evidence supervision chỉ đi vào loss ở train/validation; thay đổi label của test record không được làm thay đổi logits/prediction.
7. Teacher forcing và greedy inference dùng cùng encoder, controller, bridge, memory và source bias.
8. Official generation là `num_beams=1`, `do_sample=false` và argmax từng bước.
9. `last.pt` là checkpoint mặc định cho đánh giá chính; không tự chọn best theo test.
10. Không có model phụ hoặc reranker tại inference.

---

## 5. Ký hiệu và tensor contract

| Ký hiệu | Shape | Ý nghĩa |
|---|---|---|
| \(X\) | `[B,T]` | encoder token IDs |
| \(M\) | `[B,T]` | memory validity mask, gồm content và special/prefix tokens hợp lệ |
| \(C\) | `[B,T]` | source-content mask; article tokens bằng 1, prefix/EOS/padding bằng 0 |
| \(H^{(l)}\) | `[B,T,d_e]` | hidden state encoder tại layer \(l\) |
| \(P\) | `[B,T_p,d_d]` | embedding các decoder prompt tokens |
| \(M_p\) | `[B,T_p]` | decoder prompt validity mask |
| \(p\) | `[B,d_d]` | masked mean của prompt embeddings |
| \(N\) | `[B,1]` | số article tokens visible, \(N=\sum_t C_t\) |
| \(K\) | `[B,1]` | requested output budget, biết ở cả train và inference |
| \(c\) | `[B,d_c]` | controller state |
| \(Z\) | `[B,T,d_d]` | full source memory trong decoder space |
| \(b\) | `[B,T]` | additive source prior cho cross-attention |

Giá trị khởi đầu:

- `d_c = 256`
- `depth_taps = 4`
- `depth_rank = 128`
- `feature_rank = 256`
- `focus_hidden = 256`
- `focus_windows = [32, 128, 512]`
- overlap giữa các vùng là 50%.

Các giá trị này là hypothesis ban đầu, không phải kết luận từ technical report. Chúng phải xuất hiện trong ablation hoặc sensitivity pilot trước khi viết thành lựa chọn cuối cùng.

---

## 6. Computational graph cuối cùng

```text
source ids ──> pretrained encoder ──> final state H_L ───────────────┐
                         │                                           │
                         └── last four states ─> depth residual ─────┤
                                                                     v
decoder prompt ─> prompt pool ─┐                            cross-space residual
source pool ───────────────────┼─> controller c                       │
source length + output budget ─┘                                      v
                                                            full memory Z [B,T,d]
                                                                     │
                         multi-scale pools 32/128/512 <───────────────┤
                                    │                                │
                                    v                                │
                         shared token prior b [B,T]                   │
                                    │                                │
                                    └──────┬─────────────────────────┘
                                           v
                              every decoder cross-attention
                                           │
                                           v
                                  greedy summary tokens
```

Không có nhánh nào tạo summary memory thứ hai. Multi-scale route chỉ sinh relative bias; decoder vẫn đọc mọi token trong `Z`.

Pseudocode chuẩn của một teacher-forced forward:

```python
encoder_state = encoder(
    input_ids,
    attention_mask,
    source_content_mask,
)
prompt_embeddings = decoder.embed_tokens(decoder_prompt_ids)
bridge_state = bridge(
    encoder_state,
    prompt_embeddings,
    decoder_prompt_mask,
    output_budget,
)
decoder_output = decoder(
    decoder_input_ids,
    decoder_attention_mask,
    memory=bridge_state.memory,
    memory_mask=bridge_state.memory_mask,
    source_bias=bridge_state.source_bias,
    labels=labels,
)
loss_ce = decoder_output.loss
loss_alloc = allocation_loss(
    bridge_state.source_bias,
    allocation_target,
    allocation_valid,
    bridge_state.content_mask,
)
loss = loss_ce + allocation_weight * loss_alloc
```

Greedy generation chạy phần encoder/prompt/bridge đúng một lần, sau đó lặp
decoder với cache. Không có forward thứ hai để reroute memory.

---

## 7. Đặc tả toán học

### 7.1. Document-, prompt- và budget-conditioned controller

Nguồn và prompt thuộc hai representation space khác nhau, vì vậy không cộng trực tiếp. Mỗi phía có projection riêng về controller space:

\[
\bar h = \operatorname{MaskedMean}(H^{(L)},C),
\qquad
p = \operatorname{MaskedMean}(P,M_p).
\]

\[
c_h=W_h\operatorname{RMSNorm}(\bar h),
\qquad
c_p=W_p\operatorname{RMSNorm}(p).
\]

Budget features chỉ dùng thông tin có sẵn ở inference, với
\(N=\sum_t C_t\):

\[
u_b=\left[
\log(1+N),
\log(1+K),
K/\max(N,1)
\right],
\qquad c_b=\operatorname{MLP}_b(u_b).
\]

\[
c=\operatorname{RMSNorm}(c_h+c_p+c_b).
\]

Quy tắc parity bắt buộc:

- `K` là requested generation budget từ task config hoặc metadata input hợp lệ.
- Không dùng độ dài reference thật để tạo `K` lúc train.
- Nếu task không có budget theo sample, `K = generation.max_new_tokens` ở cả train và inference.

Khởi tạo `W_h`, `W_p` theo Xavier; MLP budget có output scale nhỏ. Decoder
prompt vector chỉ được lấy bằng embedding lookup và masked pooling trên
instruction/prefix cố định; tuyệt đối không chạy decoder probe và không đưa
target suffix vào controller. Controller được tính một lần cho mỗi document.

### 7.2. Bounded depth-selective residual readout

Với bốn hidden states cuối:

\[
e_{t,j}=u_d^\top\operatorname{RMSNorm}_j(H_t^{(L-4+j)})+(W_{depth}c)_j,
\qquad
\pi_{t,:}=\operatorname{Softmax}_j(e_{t,j}),
\qquad
D_t=\sum_{j=1}^{4}\pi_{t,j}\operatorname{RMSNorm}_j(H_t^{(L-4+j)}).
\]

Shared scorer `u_d` đọc trực tiếp representation của từng candidate token. Context thêm preference theo depth, không thay thế thông tin token. `u_d` và `W_depth` khởi tạo zero; depth weights ban đầu uniform. Tensor weights có shape `[B,T,J]`, không materialize thêm `[B,T,J,d_e]`. Softmax chỉ theo chiều depth. Đây là readout sau encoder, không sửa computation bên trong backbone.

Không thay final state bằng \(D\). Chỉ học delta so với final-state anchor:

\[
\Delta^d_t=B_d\operatorname{SiLU}
\left(A_d[D_t-\operatorname{RMSNorm}(H_t^{(L)})]\right),
\]

\[
\alpha_d(c)=\alpha_{d,max}\sigma(w_d^\top c+a_d),
\qquad
H_t^\star=H_t^{(L)}+\alpha_d(c)\Delta^d_t.
\]

Khởi tạo:

- \(A_d\): Kaiming/Xavier bình thường.
- \(B_d=0\): output ban đầu đúng final encoder state.
- `depth_gate_init = 0.02`.
- `depth_gate_max = 0.15`.
- Depth logits khởi tạo đều; không hard-code tầng thắng.

Lưu ý gradient: vì \(B_d=0\), ở backward đầu tiên gradient có thể chỉ đi vào
\(B_d\); sau một optimizer update gradient mới đi sâu vào \(A_d\), depth
router và gate. Test phải phản ánh đúng tính chất zero-init này.

### 7.3. Identity-preserving cross-space feature residual

Đường chính:

\[
Z_t^0=P_0H_t^\star.
\]

- Nếu \(d_e=d_d\), \(P_0\) là identity cố định.
- Nếu \(d_e\ne d_d\), \(P_0\) là linear projection khởi tạo semi-orthogonal và trainable.

Đường hiệu chỉnh:

\[
\Delta^f_t=B_f\operatorname{SiLU}
\left(A_f\operatorname{RMSNorm}(H_t^\star)\right),
\]

\[
\gamma_f(c)=\gamma_{max}\sigma(W_\gamma c+a_\gamma),
\qquad
Z_t=Z_t^0+\gamma_f(c)\odot\Delta^f_t.
\]

Trong đó \(\gamma_f\in\mathbb{R}^{d_d}\) là feature-wise gate. Khởi tạo:

- \(B_f=0\), vì vậy memory ban đầu vẫn đúng đường chính.
- `feature_gate_init = 0.02`.
- `feature_gate_max = 0.20`.
- Không thêm loss bắt projection phải giống identity. Anchor và zero-init bảo toàn đường pretrained lúc khởi tạo; gate bị chặn nhưng norm của delta học được không có chặn cứng. Không được suy ra bảo đảm ổn định hay tăng ROUGE chỉ từ bound của gate.

Đây là phần học mismatch PPLX-space → Qwen-space. Nó không được gọi là Gated MLA hay GR; nó chỉ chuyển nguyên lý feature-wise gated read thành một cross-space residual nhỏ.

### 7.4. Multi-scale full-memory focus

AFMR không phụ thuộc sentence splitter trong forward. Nó tạo các vùng token overlap ở ba scale:

\[
\mathcal W=\{32,128,512\},
\qquad stride(w)=w/2.
\]

Masked mean pooling cho vùng \(j\), scale \(w\). Grid index chỉ đếm
content tokens; encoder prefix, EOS và padding không làm dịch boundary của
article:

\[
U_j^{(w)}=
\frac{\sum_{t\in R_j^{(w)}}C_tZ_t}
{\max(1,\sum_{t\in R_j^{(w)}}C_t)}.
\]

Một scorer dùng chung giữa các scale, cộng scale embedding \(e_w\):

\[
s_j^{(w)}=v^\top\operatorname{SiLU}
\left(W_u\operatorname{RMSNorm}(U_j^{(w)})+W_c c+e_w\right).
\]

Mỗi vector score được center và RMS-squash độc lập theo document và scale.
Không dùng z-score `sqrt(var + eps)` vì scorer zero-init có thể làm gradient
ban đầu bị khuếch đại bởi \(1/\sqrt{\epsilon}\):

\[
\delta s_j^{(w)}=
s_j^{(w)}-\operatorname{MaskedMean}(s^{(w)}),
\qquad
\tilde s_j^{(w)}=
\frac{\delta s_j^{(w)}}
{\sqrt{1+\operatorname{MaskedMean}((\delta s^{(w)})^2)}}.
\]

Score vùng được lift về token bằng overlap-add rồi chia số vùng phủ token.
Một scale chỉ khả dụng nếu có ít nhất hai vùng non-empty; scale một vùng
không thể phân biệt vị trí và phải trả score 0. Controller chọn mixture trên
các scale khả dụng:

\[
\omega(c)=\operatorname{MaskedSoftmax}(W_\omega c,A_{\mathcal W}),
\qquad
s_t=\sum_{w\in\mathcal W}\omega_w(c)
\operatorname{Lift}(\tilde s^{(w)})_t.
\]

Nếu không scale nào có hai vùng, đặt \(s=0\), \(b=0\), và bỏ sample đó khỏi
allocation loss. Mọi phép mean/variance/log-sum-exp của focus path được tính
ở FP32 rồi cast bias về dtype của cross-attention.

Focus strength và temperature đều bị chặn:

\[
\lambda(c)=\lambda_{max}\sigma(w_\lambda^\top c+a_\lambda),
\]

\[
\tau(c)=\tau_{min}+(\tau_{max}-\tau_{min})
\sigma(w_\tau^\top c+a_\tau),
\]

\[
r_t=\lambda(c)\tanh(s_t/\tau(c)).
\]

Giá trị khởi đầu:

- `focus_strength_init = 0.10`.
- `focus_strength_max = 1.0`. Vì \(r_t\in[-\lambda,\lambda]\), cap này
  giới hạn odds ratio lớn nhất do riêng prior tạo ra ở
  \(\exp(2)\approx7.39\); route có thể ưu tiên mạnh nhưng không dễ áp đảo QK.
- `temperature_init = 1.0`.
- `temperature_min = 0.5`.
- `temperature_max = 2.0`.
- Vector output \(v=0\), do đó source bias bằng 0 ở initialization. Bước
  optimizer đầu tiên cập nhật \(v\); các layer scorer/controller phía trước
  bắt đầu nhận gradient sau khi \(v\) khác 0.

Prior được chuẩn hóa trên content tokens:

\[
b_t=
\begin{cases}
r_t-\log\left(\frac{1}{N}\sum_{k:C_k=1}e^{r_k}\right),&C_t=1,\\
0,&M_t=1,\ C_t=0.
\end{cases}
\]

Padding vẫn bị chặn bởi memory mask trong cross-attention. Phép trừ trên cố
định scale của multiplicative prior \(e^b\) trên content tokens và giữ prior
trung bình của content ngang mức bias 0 của prefix/EOS. Nó không kiểm soát
chính xác attention mass cuối vì content logits từ QK vẫn tham gia. Giá trị
quan trọng là chênh lệch tương đối giữa các \(b_t\).

### 7.5. Decoder cross-attention

Mỗi decoder layer đọc cùng `Z` và `b`, nhưng query của từng layer và output position vẫn khác:

\[
A_{l,q}=\operatorname{Softmax}
\left(\frac{Q_{l,q}K_l(Z)^\top}{\sqrt{d_h}}+b+\log M\right).
\]

\[
h_l'=h_l+g_l\operatorname{CrossAttn}
(\operatorname{RMSNorm}(h_l),Z,b,M).
\]

- Cross-attention nằm ở mọi decoder layer.
- Q/K/V/O được copy từ self-attention khi shape tương thích.
- \(g_l\) là scalar bounded riêng từng layer, khởi tạo `0.10`, max `1.0`.
- Không đặt mục tiêu ép `cross_res` phải lớn; gate và residual ratio là diagnostic, không phải quality objective.
- `b` được ghép vào additive SDPA mask. Không materialize thủ công full attention matrix nếu PyTorch SDPA hỗ trợ kernel fused cho shape hiện tại.
- Cross K/V của source phải được cache một lần mỗi layer trong greedy generation.
- Giữ helper đã kiểm chứng cho trainable bias head alignment và SDPA
  log-sum-exp backward. Một unit test phải xác nhận gradient của `b` khác 0
  dưới đúng dtype/kernel dùng trên B200; nếu kernel rơi về math path, profiling
  phải báo rõ thay vì che latency.

### 7.6. Output contract

Forward trả một dataclass ổn định:

```python
AFMROutput(
    logits,
    loss_ce,
    loss_allocation,
    loss,
    source_memory,
    source_bias,
    diagnostics,
)
```

`diagnostics` chỉ chứa tensor detached cần cho log; không trả hidden states hoặc attention maps lớn mặc định.

### 7.7. Initialization contract

Mọi bounded sigmoid gate dùng cùng quy tắc:

\[
g(x)=g_{\max}\sigma(Wx+a),
\qquad
W=0,\quad
a=\operatorname{logit}(g_{\mathrm{init}}/g_{\max}).
\]

Temperature dùng phép nội suy bounded tương tự, với bias được tính từ vị trí
của `temperature_init` trong khoảng
`[temperature_min, temperature_max]`. Depth/scale router weights khởi tạo 0
để cho phân phối đều trên lựa chọn hợp lệ. Output factors \(B_d,B_f\) và focus
vector \(v\) khởi tạo 0. Cross-attention gate có raw parameter riêng từng
layer và dùng cùng công thức bounded.

Acceptance test tại step 0:

- depth output bằng final encoder state;
- feature memory bằng đường \(P_0H^{(L)}\);
- source bias bằng 0;
- không có NaN/Inf trong FP32 hoặc BF16;
- AFMR memory/bias khớp exact A0; decoder graph, gồm copied
  cross-attention, không đổi giữa A0 và full AFMR tại initialization.

---

## 8. Objective và đường gradient

### 8.1. Main CE-only; một auxiliary loss cho ablation

Main configuration dùng `allocation_weight=0.0`, nên không cần positive metadata và không mining nhãn. Depth, feature và focus đều nhận gradient từ generation CE. Công thức tổng quát dưới đây giữ cho ablation bật allocation, không phải mặc định main.

\[
\mathcal L=\mathcal L_{CE}+\lambda_a\mathcal L_{alloc}.
\]

Token CE là objective chính. Allocation loss giám sát đúng score tạo source bias lúc inference, thay vì huấn luyện một contrastive head tách rời:

\[
\hat q_t=\operatorname{MaskedSoftmax}(b,C)_t,
\qquad
\mathcal L_{alloc}=
D_{KL}(\operatorname{stopgrad}(q^*)\Vert\hat q).
\]

KL được tính riêng từng sample ở FP32 rồi mean chỉ trên
`allocation_valid=true`. Nếu một batch không có sample hợp lệ,
\(\mathcal L_{alloc}\) là một scalar zero trên đúng device/dtype và không làm
thay đổi reduction của CE. `MaskedSoftmax` đặt probability prefix/EOS/padding
bằng 0, không chỉ gán logit nhỏ rồi vẫn để chúng trong denominator.

Khởi đầu:

- Main: `allocation_weight = 0.0`; allocation ablation: `0.10`.
- Ramp tuyến tính từ 0 đến 0.10 trong 10% số optimizer steps đầu của interface warm-up.
- Trong allocation ablation, giữ 0.10 ở full fine-tuning; main luôn bằng 0.
- Chỉ đổi weight sau pilot một-biến-thay-đổi; không tăng vì thấy allocation loss còn cao.
- Log \(\lambda_a\mathcal L_{alloc}/\mathcal L_{CE}\). Nếu median ratio vượt
  0.30 sau ramp và validation CE xấu hơn A3, chạy đúng một sensitivity pilot
  với weight 0.05; không dùng automatic loss balancing trong main model.

### 8.2. Xây \(q^*\)

Evidence target chỉ tồn tại cho train/validation:

1. Chia source thành các source units trong bước prepare, không trong hot path.
2. Nếu raw `text` là list, mỗi phần tử là một unit.
3. Nếu raw `text` là string, sentence segmentation chỉ dùng để tạo supervision metadata; forward AFMR vẫn dùng token windows.
4. Chọn tối đa **ba** positive units bằng greedy marginal coverage trên ROUGE-1 + ROUGE-2 đối với training reference.
5. Mỗi positive unit nhận tổng probability mass bằng nhau; token trong cùng unit chia đều mass của unit.
6. Sau source truncation, chỉ giữ phần positive visible. Nếu không còn positive visible, sample không đóng góp allocation loss.

Miner phải deterministic: normalize Unicode, case-fold cho metric mining,
tokenize theo một Unicode-aware word tokenizer cố định, chọn unit có marginal
\(R_1+R_2\) lớn nhất, break tie bằng source index nhỏ hơn, và dừng khi không
còn gain dương. Nó lưu version dạng plain string trong prepared metadata,
không lưu hash. Với ngôn ngữ không có word boundary ổn định, allocation loss
phải được ablate/tắt hoặc miner riêng phải được khai báo; không giả rằng
English word ROUGE là supervision tổng quát cho mọi ngôn ngữ.

Với \(J\) positive units còn visible và unit \(j\) chứa \(n_j\) tokens:

\[
q_t^*=\frac{1}{J}\frac{1}{n_j}
\quad\text{nếu }t\text{ thuộc positive unit }j.
\]

Cách phân phối này tránh unit dài tự động nhận nhiều mass hơn. Trường `label` có sẵn trong dataset không được dùng mặc định; có thể là một ablation riêng, không là dependency của kiến trúc.

### 8.3. Toàn bộ negative set

Không cần `evidence_hard_negatives=4/8`. Softmax denominator của \(\hat q\)
chứa mọi content token visible, nên mọi vùng ngoài positive set là negative
và vùng có score cao nhận gradient âm mạnh hơn tự nhiên. Điều này bỏ sampling
noise và vòng lặp hard-negative chậm của implementation cũ.

Nếu sau ablation có bằng chứng listwise full-memory quá loãng, hard-negative reweighting phải là một thí nghiệm riêng; không được lén thêm vào main objective.

### 8.4. Gradient contract

Đường CE:

```text
CE
 -> decoder logits
 -> decoder cross-attention
 -> source bias b and source memory Z
 -> focus scorer/controller and feature residual
 -> depth residual
 -> encoder
```

Đường allocation:

```text
allocation KL
 -> source bias b
 -> multi-scale focus scorer
 -> scale mixture, temperature, focus strength
 -> controller
 -> pooled source/prompt representations
```

Yêu cầu test:

- Khi `allocation_weight=0`, CE vẫn tạo gradient cho focus scorer và controller qua live source bias.
- Khi decoder/encoder bị freeze trong warm-up, AFMR và cross-attention vẫn có gradient.
- Với zero-init output matrices, test hai optimizer steps: bước đầu cập nhật output factor, bước thứ hai xác nhận input factor/router nhận gradient.
- Không có `.detach()` trên `Z`, `b`, depth mixture hoặc controller trong path CE.
- Chỉ `q*` được stop-gradient.

---

## 9. Train–inference parity

Đây là acceptance gate, không phải tối ưu phụ.

| Thành phần | Teacher forcing | Greedy inference |
|---|---|---|
| Encoder input | source | cùng source |
| Prompt representation | decoder instruction/prefix | cùng instruction/prefix |
| Budget \(K\) | configured requested budget | cùng configured budget |
| Depth readout | AFMR | cùng AFMR |
| Cross-space residual | AFMR | cùng AFMR |
| Multi-scale prior | AFMR | cùng AFMR |
| Evidence labels/reference | chỉ trong loss | không dùng |
| Source memory | full | full |
| Source prior | shared \(b\) | shared \(b\) |
| Decoder mode | teacher-forced tokens | cached argmax tokens |

Một test phải chạy cùng source/prompt/budget qua hai entry points và xác nhận `source_memory` cùng `source_bias` bằng nhau trong tolerance.

---

## 10. Cấu trúc source code mục tiêu

Không giữ thư mục `core` hỗn hợp. Tên project directory và import package phải khác nhau để tránh kiểu `eviseq/eviseq` khó đọc:

```text
src/eviseq_new/
├── README.md
├── pyproject.toml
├── run.py
├── eviseq_afmr/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── prepare.py
│   │   ├── dataset.py
│   │   └── collate.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── outputs.py
│   │   ├── encoder.py
│   │   ├── controller.py
│   │   ├── afmr.py
│   │   ├── cross_attention.py
│   │   ├── decoder.py
│   │   └── model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── optimizer.py
│   │   ├── checkpoint.py
│   │   └── engine.py
│   └── evaluation/
│       ├── __init__.py
│       ├── generate.py
│       ├── metrics.py
│       └── rouge155.py
├── configs/
│   ├── base.yaml
│   ├── pubmed.yaml
│   ├── arxiv.yaml
│   ├── cnndm.yaml
│   ├── wikilingua.yaml
│   ├── smoke.yaml
│   └── ablations/
│       ├── final_layer_only.yaml
│       ├── no_feature_residual.yaml
│       ├── no_focus_prior.yaml
│       ├── single_scale_128.yaml
│       ├── no_prompt_conditioning.yaml
│       ├── no_budget_conditioning.yaml
│       └── no_allocation_loss.yaml
├── scripts/
│   └── run.sh
└── tests/
    ├── test_config.py
    ├── test_data.py
    ├── test_afmr_shapes.py
    ├── test_afmr_identity.py
    ├── test_focus_prior.py
    ├── test_gradients.py
    ├── test_train_infer_parity.py
    ├── test_decoder_cache.py
    ├── test_checkpoint.py
    └── test_evaluation_resume.py
```

`eviseq_afmr` là import namespace công khai. `core`, `modeling/architecture.py` khổng lồ và `training/objectives.py` chứa nhiều objective cũ phải biến mất sau migration.

### 10.1. Giới hạn trách nhiệm từng module

| File | Trách nhiệm duy nhất |
|---|---|
| `config.py` | load một base + một task override, validate schema và invariants |
| `data/schema.py` | canonical record dataclasses và field validation |
| `data/prepare.py` | chuyển raw JSONL, tạo unit spans và ba positive units cho train/valid |
| `data/dataset.py` | streaming/indexed JSONL access, không tạo evidence online |
| `data/collate.py` | tokenize, map char spans sang token mask, pad batch |
| `modeling/encoder.py` | backend adapter, trả final state + đúng bốn taps + source mask |
| `modeling/controller.py` | tạo `c` từ source, prompt, lengths và budget |
| `modeling/afmr.py` | depth residual, feature residual, multi-scale focus và source bias |
| `modeling/cross_attention.py` | SDPA cross-attention với additive source prior |
| `modeling/decoder.py` | Qwen decoder wrapper, copied cross-attention, cache contract |
| `modeling/model.py` | nối encoder → AFMR → decoder; không tự tính metric |
| `training/losses.py` | CE + allocation KL |
| `training/optimizer.py` | parameter groups, freeze policy, scheduler |
| `training/checkpoint.py` | atomic save/load/resume và structural compatibility |
| `training/engine.py` | warm-up/full loops, teacher-forced validation và compact logs |
| `evaluation/generate.py` | batched greedy decode, append/resume JSONL, ETA |
| `evaluation/metrics.py` | Python diagnostic metrics và length/repetition statistics |
| `evaluation/rouge155.py` | isolated Perl ROUGE-1.5.5 adapter |

Không file modeling nào vượt khoảng 500 dòng. Không file training nào vượt khoảng 600 dòng. Nếu vượt, phải tách theo trách nhiệm thay vì tiếp tục thêm condition.

---

## 11. Public API và dataclasses

### 11.1. Encoder contract

```python
@dataclass
class EncoderState:
    final: torch.Tensor
    taps: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor
    content_mask: torch.Tensor
```

```python
class EncoderAdapter(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> EncoderState: ...
```

- `len(taps)` phải bằng `architecture.depth_taps` khi depth residual bật.
- `taps[-1]` tương ứng final layer.
- Native bidirectional encoder giữ nguyên attention implementation pretrained.
- Causal Qwen encoder chỉ được hỗ trợ qua backend riêng với số upper bidirectional layers khai báo rõ; đây là encoder ablation, không bị trộn vào AFMR.

### 11.2. AFMR contract

```python
@dataclass
class BridgeState:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    content_mask: torch.Tensor
    source_bias: torch.Tensor
    controller: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
```

```python
class AdaptiveFullMemoryResidualBridge(nn.Module):
    def forward(
        self,
        encoder_state: EncoderState,
        prompt_embeddings: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_budget: torch.Tensor,
    ) -> BridgeState: ...
```

Không nhận `labels`, `reference`, `positive_indices` hay decoder hidden states.

### 11.3. Model contract

```python
class EviSeqAFMR(nn.Module):
    def encode_source(
        self,
        input_ids,
        attention_mask,
        source_content_mask,
        decoder_prompt_ids,
        decoder_prompt_mask,
        output_budget,
    ) -> BridgeState: ...

    def forward(
        self,
        input_ids,
        attention_mask,
        source_content_mask,
        decoder_prompt_ids,
        decoder_prompt_mask,
        decoder_input_ids,
        decoder_attention_mask=None,
        labels=None,
        allocation_target=None,
        allocation_valid=None,
        output_budget=None,
    ) -> AFMROutput: ...

    def generate_greedy(...) -> GenerationOutput: ...
```

`decoder_prompt_ids` chỉ chứa instruction/prefix có sẵn trước token summary
đầu tiên. `forward` chuyển `allocation_target` thẳng sang loss function;
không đưa nó vào bridge. `generate_greedy` bắt buộc gọi `encode_source` đúng
một lần cho mỗi batch, rồi chỉ lặp decoder.

---

## 12. Config tối giản

Chỉ có tối đa một `base.yaml` và một task override. Không có chuỗi template → task → model → corrected → strong. Mọi field cũ không thuộc schema mới phải gây lỗi, không silently ignore.

```yaml
experiment:
  name: pubmed_pplx_qwen_afmr
  output_dir: runs/afmr/pubmed_pplx_qwen

model:
  encoder_name: /path/to/pplx-embed-v1-0.6b
  decoder_name: /path/to/Qwen3-0.6B
  tokenizer_use_fast: true
  dtype: bfloat16
  gradient_checkpointing: true
  attention_implementation: sdpa

encoder:
  backend: pretrained_native
  upper_bidirectional_layers: 0

architecture:
  name: afmr_v1
  controller_dim: 256
  depth_taps: 4
  depth_rank: 128
  depth_gate_init: 0.02
  depth_gate_max: 0.15
  feature_rank: 256
  feature_gate_init: 0.02
  feature_gate_max: 0.20
  focus_hidden: 256
  focus_windows: [32, 128, 512]
  focus_overlap: 0.5
  focus_strength_init: 0.10
  focus_strength_max: 1.0
  temperature_init: 1.0
  temperature_min: 0.5
  temperature_max: 2.0

decoder:
  cross_attention_every: 1
  initialize_cross_from_self: true
  cross_gate_init: 0.10
  cross_gate_max: 1.0

objective:
  allocation_weight: 0.0
  allocation_ramp_ratio: 0.10
  max_positive_units: 3

training:
  interface_warmup_epochs: 1
  full_finetune_epochs: 4
  batch_size: 48
  gradient_accumulation_steps: 2
  warmup_bridge_lr: 1.0e-4
  warmup_cross_attention_lr: 1.0e-4
  full_encoder_lr: 1.0e-5
  full_decoder_lr: 1.0e-5
  full_bridge_lr: 3.0e-5
  full_cross_attention_lr: 5.0e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  seed: 42
  log_every_steps: 10
  save_each_epoch: true

data:
  train_file: datasets/pubmed/train.jsonl
  validation_file: datasets/pubmed/validation.jsonl
  test_file: datasets/pubmed/test.jsonl
  source_field: text
  target_field: summary
  id_field: id
  list_separator: "\n"
  encoder_prefix: "Summarize the article faithfully and concisely.\nArticle:\n"
  decoder_prompt: "Summarize the article faithfully and concisely.\nSummary:\n"
  max_source_length: 4096
  max_target_length: 512

generation:
  batch_size: 96
  max_new_tokens: 512
  min_new_tokens: 16
  repetition_penalty: 1.0
  no_repeat_ngram_size: 0
  num_beams: 1
  do_sample: false
```

### 12.1. Field bị loại khỏi schema

Xóa và reject các key sau:

```text
native_attention.variant
implementation_revision
memory_bank_count
coverage_slot_count
evidence_source_probe_layers
evidence_hard_negatives
use_contrastive
source_swap
geometry
salience_ranking_weight
alignment_weight
online_kd
ranking_phase
benchmark
reporting
target_total_footprint_approx
fingerprint
```

### 12.2. Structural compatibility

Checkpoint lưu riêng một `architecture_spec` gồm các field ảnh hưởng shape/graph. Khi load:

- Phải match `architecture.name`, hidden sizes, ranks, taps, windows, decoder layer count và cross-attention layout.
- Không yêu cầu match data paths, model location, batch size, worker count hoặc generation batch size.
- Cho phép model folder chuyển vị trí nếu config/model shape tương thích.
- Không dùng `resolved_config.yaml` cũ làm nguồn cấu hình ngầm.
- File snapshot config trong run directory chỉ để audit; CLI config hiện tại luôn là nguồn runtime rõ ràng.

---

## 13. Data preparation

### 13.1. Canonical JSONL

Sau prepare, mỗi train/validation record có dạng:

```json
{
  "id": "PMC...",
  "source": "...",
  "target": "...",
  "unit_char_spans": [[0, 120], [121, 284]],
  "positive_unit_indices": [1, 7, 12]
}
```

Test record chỉ cần:

```json
{
  "id": "PMC...",
  "source": "...",
  "target": "..."
}
```

`target` ở test chỉ được evaluator đọc sau generation. Model/dataset collator
không tạo allocation target cho test. `encoder_prefix` được thêm trong
collator, không ghi lẫn vào `source`; nhờ đó collator tạo được
`source_content_mask` chính xác.

### 13.2. Prepare flow

```text
raw JSONL
 -> validate id/text/summary
 -> normalize list-or-string fields
 -> preserve source unit boundaries
 -> choose top-3 units by greedy marginal ROUGE-1 + ROUGE-2
 -> store source string + char spans + positive indices
 -> write canonical JSONL atomically
```

Prepare chạy trước khi load model, có progress và ETA. Training không tính greedy labels lại và không giữ một evidence cache lớn trong RAM.

### 13.3. Token mapping

Collator tạo `encoder_prefix + source`, tokenizes toàn chuỗi bằng
`return_offsets_mapping=True`, rồi sinh hai mask khác nhau:

- `attention_mask`: mọi token hợp lệ mà decoder được phép đọc;
- `source_content_mask`: chỉ token có offset giao với source article, không
  gồm prefix, BOS/EOS hoặc padding.

Sau đó collator map positive char spans sang source-token mask. Sau truncation:

- Bỏ token ngoài `max_source_length`.
- Bỏ positive unit không còn token visible.
- Renormalize \(q^*\) trên positive units còn visible.
- Mark `allocation_valid=false` nếu không còn positive nào.

Không tokenize từng unit rồi nối IDs vì BPE tại boundary có thể khác tokenization của full source.
Main implementation yêu cầu fast tokenizer có offset mapping và fail sớm với
thông báo rõ nếu backend không hỗ trợ; không âm thầm dùng một mask xấp xỉ.

### 13.4. Split policy

- Train và validation được phép tạo positive metadata từ reference của chính split đó.
- Test không tạo hoặc sử dụng positive metadata trong generation.
- Không có hard fail vì duplicate/cross-split content trong training CLI.
- Nếu cần audit duplicate, cung cấp command read-only riêng; không dùng fingerprint/hashing trong run artifacts.

---

## 14. Training protocol

### 14.1. Stage 1 — interface warm-up

Thời lượng mặc định: 1 epoch.

Trainable:

- AFMR controller.
- Depth residual.
- Feature residual.
- Multi-scale focus scorer.
- Decoder cross-attention và cross gates.

Frozen:

- Encoder backbone.
- Decoder embeddings, self-attention, MLP, norms và LM head.

Loss vẫn là CE + allocation trên đúng graph deployed. Không thay dynamic route bằng static teacher trong warm-up.

### 14.2. Stage 2 — full fine-tuning

Thời lượng mặc định: 4 epochs.

- Unfreeze encoder và decoder.
- Dùng LR nhỏ cho backbone, LR lớn hơn cho AFMR/cross-attention.
- Giữ cùng graph và cùng objective.
- Không reset optimizer moments của AFMR/cross-attention khi chuyển stage; tạo optimizer mới nhưng carry state theo parameter name cho các parameter tiếp tục train.
- Scheduler mới cho stage 2, tính theo optimizer steps thật sau accumulation.

Cho phép `interface_warmup_epochs=0`. Khi bằng 0, bỏ hẳn stage 1, không tạo scheduler chia cho zero và bắt đầu full fine-tuning trực tiếp.

### 14.3. Effective batch

\[
B_{effective}=B_{physical}\times accumulation\times world\_size.
\]

PubMed single-B200 khởi đầu `48 × 2 = 96`. Khi dùng DDP hai GPU, phải điều chỉnh để không vô tình nhân effective batch thành 192 nếu experiment muốn giữ 96.

### 14.4. Optimizer groups

Mỗi parameter phải thuộc đúng một group:

1. `encoder`
2. `decoder`
3. `bridge`
4. `cross_attention`

Norm/bias/gate scalar không weight decay. Matrix projection dùng weight decay. Manifest được kiểm tra lúc build; unclassified hoặc duplicate parameter phải raise error.

### 14.5. Checkpoint

Mỗi checkpoint chứa:

- model state dict;
- optimizer và scheduler states;
- scaler nếu dùng FP16;
- stage, epoch, global step, stage step;
- RNG state Python/NumPy/PyTorch/CUDA;
- structural `architecture_spec`;
- minimal task metadata.

Ghi atomically qua temporary file rồi `os.replace`. Rank 0 là process duy nhất ghi. Tạo:

```text
epoch_001.pt
epoch_002.pt
...
last.pt
```

Không tạo `best.pt` trong main recipe để tránh nhầm protocol. Nếu cần model selection study, làm experiment riêng; official test dùng `last.pt`.

### 14.6. Validation

- Mỗi epoch chạy teacher-forced validation CE và allocation diagnostics.
- Không autoregressive decode toàn validation trong training.
- Không dùng test split trước khi full run kết thúc.
- Sau training, evaluator load `last.pt` và generate test một lần.

---

## 15. Logging tối giản

Mỗi `log_every_steps` chỉ in một JSON line:

```text
stage, epoch, step,
loss, ce, alloc, alloc_ce_ratio,
alloc_pos_mass, alloc_gap,
depth_gate, depth_entropy,
feature_residual_rms,
focus_strength, focus_temperature, scale_weights,
source_bias_rms, cross_residual_ratio,
grad_norm, step_s, samples_s, lr
```

Không log các metric legacy như `evi_acc`, `evi_queries`, `prompt_ctx`, `bridge_fuse`, `dyn_clip`, `source_swap`, `geo`, `slot_diversity` hoặc hàng chục field luôn bằng zero.

Định nghĩa quan trọng:

- `alloc_pos_mass`: tổng \(\hat q\) trên positive tokens.
- `alloc_gap`: mean source bias positive trừ mean source bias negative.
- `alloc_ce_ratio`: \(\lambda_a\mathcal L_{alloc}/\mathcal L_{CE}\), chỉ là
  diagnostic để phát hiện auxiliary loss lấn CE.
- `depth_entropy`: entropy của \(\pi\), không dùng để tối ưu.
- `feature_residual_rms`: RMS của \(\gamma_f\odot\Delta^f\) chia RMS đường chính.
- `cross_residual_ratio`: RMS cross-attention residual chia RMS decoder hidden trước residual.

Không đặt threshold cứng “càng lớn càng tốt”. Chỉ ROUGE/factuality/length ở evaluation mới quyết định model tốt hơn.

---

## 16. Evaluation protocol

### 16.1. Generation

- Load `last.pt` mặc định.
- Split mặc định cho official run là `test`.
- Greedy argmax, `num_beams=1`, `do_sample=false`.
- Dùng Transformers `Cache` API phù hợp version; không truyền tuple `past_key_values` deprecated.
- Precompute encoder, AFMR, source bias và cross K/V một lần mỗi batch.
- Append prediction JSONL sau mỗi batch và `flush` ngay.
- Progress log mỗi batch: completed/total, docs/s, elapsed, ETA, peak VRAM.
- `--resume` đọc IDs đã hoàn tất và bắt đầu từ record chưa có tiếp theo; không generate lại 0…N-1.
- Resume chỉ chấp nhận existing records tạo thành một prefix liên tục của split
  hiện tại. So khớp trực tiếp `id` và `reference` theo từng record; không dùng
  fingerprint/hash. Nếu có hole, duplicate hoặc mismatch thì fail trước khi
  nạp model.
- Nếu OOM, optional batch fallback giảm `B → B/2`; kết quả generation không được thay đổi vì batch size.

Output record:

```json
{"id":"...","prediction":"...","reference":"..."}
```

### 16.2. Metrics

Hai tầng metric:

1. `rouge==1.0.0` chỉ là diagnostic nhanh.
2. Perl ROUGE-1.5.5 qua pyrouge là số cuối cùng cho comparison/paper.

Luôn báo cùng:

- ROUGE-1/2/L F1 × 100.
- prediction/reference length mean và length ratio.
- empty/too-short/too-long rate.
- repeated trigram rate.
- latency, docs/s và peak VRAM.

Nếu chênh dưới 0.5 ROUGE, chạy paired bootstrap trên cùng prediction/reference IDs trước khi claim improvement.

---

## 17. Hiệu năng code

### 17.1. Điều bắt buộc

- Không loop Python qua source tokens, windows hoặc decoder layers trong phần có thể vectorize.
- Region pooling dùng masked `avg_pool1d`/`unfold`; lift dùng overlap-add tensor operation.
- Không `.item()`, `.cpu()` hoặc logging sync trong hot path mỗi microbatch.
- Chỉ giữ bốn encoder taps cần thiết; không bật
  `output_hidden_states=True` cho toàn bộ backbone. Capture bằng local forward
  hooks trên đúng bốn layer, remove hooks ngay sau encoder forward và giải
  phóng taps sau bridge. Không lưu capture trong persistent module state.
- Khi gradient checkpointing bật, dùng non-reentrant checkpointing nếu backend
  hỗ trợ và có test end-to-end hai optimizer steps; backend không thể expose
  bốn taps có gradient phải fail rõ hoặc dùng ablation `depth_taps=1`, không
  lặp final state giả làm nhiều depth.
- Tái sử dụng prompt embedding theo task khi decoder embeddings frozen; khi decoder unfreeze, compute trong graph mỗi batch.
- SDPA cho cross-attention; không tự tạo attention probability tensor chỉ để log.
- BF16 trên B200, TF32 cho FP32 matmul khi phù hợp.
- Gradient checkpointing chỉ cho backbone; AFMR nhỏ không checkpoint.
- `pin_memory`, persistent workers và prefetch có config, nhưng benchmark trước khi tăng workers.
- Evaluation chạy `torch.inference_mode()`.

### 17.2. Complexity budget

- AFMR phải tuyến tính theo source length ngoài cross-attention vốn có.
- Không thêm \(O(T^2)\) source attention.
- Bridge parameters mới, không tính decoder cross-attention, mục tiêu dưới khoảng 3M tại hidden size 1024.
- Main model giữ một source memory, không nhân VRAM theo số bank/scale; region tensors phải nhỏ hơn source memory.
- Báo exact parameter count từ code runtime, không hard-code footprint trong YAML.

### 17.3. Profiling gate

Trước full PubMed:

1. Profile 50 train steps sau 20 warm steps.
2. Báo mean/p50/p95 `step_s`, examples/s, peak allocated/reserved VRAM.
3. Profile encode, AFMR, decoder forward, backward và optimizer riêng bằng CUDA events.
4. AFMR overhead mục tiêu dưới 10% so với final-layer single-memory baseline.
5. Nếu vượt, sửa vectorization trước khi tăng batch hoặc chạy full.

---

## 18. Test plan

### 18.1. Offline unit tests

Không tải model từ Hugging Face. Dùng tiny fake encoder/decoder hoặc local config-only modules.

#### Config

- Valid main config pass.
- Unknown legacy key fail.
- Invalid gate range fail.
- `focus_windows` không tăng dần hoặc stride không nguyên fail.
- `num_beams != 1` hoặc `do_sample=true` fail cho official recipe.
- Warm-up bằng zero pass.

#### Shape và mask

- Variable source lengths và padding.
- Memory shape `[B,T,d_d]`, bias `[B,T]`.
- Padded/prefix/EOS positions không góp region mean hoặc focus normalization.
- Prefix/EOS vẫn còn trong full memory nếu `attention_mask=1`, với source bias bằng 0.
- Mọi valid source token còn trong memory.
- Scale có window lớn hơn source vẫn tạo đúng một valid region.

#### Identity và boundedness

- Với zero-init deltas, `H_star == H_final` và `Z == P0(H_final)` trong FP32 tolerance.
- Mọi gate nằm trong configured bounds.
- Depth weights và scale weights không âm, tổng bằng một.
- `r` nằm trong `[-focus_strength, +focus_strength]`.
- Mean `exp(b)` trên content tokens bằng một trong tolerance; prefix/EOS bias bằng 0.

#### Allocation target

- Tối đa ba positive units.
- Mỗi positive unit có equal total mass bất kể length.
- Truncation renormalize đúng.
- Sample không còn positive bị bỏ khỏi mean loss, không sinh NaN.

#### Gradient

- Allocation KL tạo nonzero gradient cho focus scorer/controller.
- CE-only tạo gradient cho source bias route và cross-attention.
- Ở optimizer step đầu, zero-init focus vector \(v\) có nonzero gradient; ở
  step thứ hai, `W_u`, `W_c` và scale router bắt đầu có gradient. Test này
  ngăn centered normalization làm chết focus branch.
- Sau hai optimizer steps, depth/feature input factors và gates có gradient.
- Frozen backbone không có grad trong warm-up.
- Mỗi full-finetune parameter trainable đều thuộc một optimizer group.

#### Parity và leakage

- Encode path của teacher forcing và greedy trả cùng memory/bias.
- Thay target/evidence metadata nhưng giữ source/prompt/budget không đổi thì memory/bias không đổi.
- Thay test `label` không đổi generated IDs.
- Padding thêm ở cuối không đổi logits valid trong tolerance.

#### Decoder/cache

- Cached và non-cached next-token logits khớp cho sequence ngắn.
- Cross K/V chỉ project một lần mỗi layer/batch.
- Shared source bias tới tất cả cross-attention layers.
- BF16 forward/backward finite.

#### Checkpoint/evaluation

- Save/load roundtrip logits exact trong tolerance.
- Resume training giữ stage/epoch/global step/RNG.
- Model path thay đổi nhưng architecture match vẫn load.
- Generation resume từ first missing ID.
- Prediction được flush sau batch.

### 18.2. Smoke test trên server

Smoke chỉ kiểm tra plumbing, không dùng để chọn architecture:

- WikiLingua: 100 train, 20 validation, 100 test.
- Một warm-up epoch và một full epoch.
- `last.pt`, per-epoch checkpoint, greedy predictions và ROUGE diagnostic phải được tạo.
- Chạy một batch với allocation labels và một batch không có valid positive.
- Chạy BF16, gradient checkpointing và KV cache.
- Không OOM, NaN, missing gradient hoặc graph mismatch.

### 18.3. Pilot PubMed

- 2,000 train / 500 validation trước full run.
- So sánh AFMR và exact single-memory baseline cùng seed, batches và generation.
- Không tiếp tục full nếu AFMR có bug parity, overhead >10%, allocation path không có gradient hoặc validation CE xấu đi rõ rệt.
- Pilot không dùng test để tune.

---

## 19. Ablation bắt buộc

| ID | Variant | Câu hỏi |
|---|---|---|
| A0 | Final layer + base projection + full-memory cross-attention | Baseline ghép hai LLM là bao nhiêu? |
| A1 | A0 + depth residual | Depth selection có giúp không? |
| A2 | A1 + feature residual | Cross-space correction có giúp không? |
| A3 | A2 + multi-scale focus, CE only | Prior có học trực tiếp từ generation không? |
| A4 | Full AFMR | Allocation supervision có chuyển thành ROUGE không? |
| A5 | Full AFMR, single 128-token scale | Multi-scale có cần thiết không? |
| A6 | Full AFMR, no prompt conditioning | Prompt có góp gì ngoài source context? |
| A7 | Full AFMR, no budget conditioning | Adaptive budget có góp gì không? |

Không cần chạy mọi ablation full 5 epochs ngay. Quy trình:

1. Pilot tất cả A0–A7 trên train/validation subset, một seed.
2. Loại variant lỗi hoặc thua rõ.
3. Chạy A0, A3, A4 và component có tín hiệu tốt nhất full PubMed.
4. Xác nhận main và baseline với ba seeds nếu tài nguyên cho phép.
5. Chỉ sau khi main ổn trên PubMed mới mở rộng CNNDM, WikiLingua và ArXiv.

PubMed và ArXiv cùng domain scientific nên không đủ để claim general-purpose. Ít nhất cần một news dataset và một procedural dataset hoặc phân tích rõ giới hạn claim.

---

## 20. Implementation phases

### 20.1. Map từ implementation hiện tại sang AFMR

Review code hiện tại xác nhận không nên chồng AFMR lên CoVeR. AFMR là clean
graph break; chỉ port utility đã được kiểm chứng:

| File hiện tại | Giữ/port có chọn lọc | Viết lại hoặc xóa |
|---|---|---|
| `core/modeling/bridge.py` | RMSNorm, identity/zero-init linear, output dataclass | thay `EvidenceBridge` bằng `AdaptiveFullMemoryResidualBridge`; bỏ unit salience và mọi `reroute*` |
| `core/modeling/encoder.py` | `PretrainedNativeEncoder`, freeze/unfreeze, cơ chế lấy vài final states | bỏ evidence head, `unit_logits`, `unit_ids`; encoder chỉ trả `EncoderState` |
| `core/modeling/attention.py` | helper căn shape bias head và bảo toàn backward của trainable SDPA bias | bỏ unit pooling, evidence-key và dual-mask helpers khỏi main backend |
| `core/modeling/architecture.py` | không port nguyên file | tách thành `model.py` + `afmr.py`; bỏ Coverage/PCEB head, neutral decoder probe và per-layer route |
| `core/modeling/decoder.py` | copied cross-attention, cross gate, KV cache | chỉ nhận shared `[B,T]` bias; reject `[B,L,T]` route |
| `core/training/objectives.py` | không port generic objective maze | thay bằng `losses.py` chỉ chứa CE và allocation KL |
| `core/training/engine.py` | DDP, scheduler concepts, checkpoint orchestration | viết một engine duy nhất cho hai stages |
| `core/training/trainer.py` | không giữ monkey-patch | xóa; CLI gọi trực tiếp engine mới |
| `core/training/online_kd.py` | không có phần dùng lại | xóa khỏi package AFMR |
| `core/data/dataset.py` | JSONL field mapping và cleaning đã kiểm chứng | tách schema/prepare/dataset/collate; bỏ online oracle/cache/unit path |
| `core/evaluation/generation.py` | greedy argmax loop và source KV reuse | bỏ nhánh gọi coverage/prompt-conditioned reroute sau `encode` |
| `core/evaluation/engine.py` | streaming output, metrics, OOM batch fallback | chuyển sang evaluator mới dùng `source_content_mask` và resume không hash |

Bug hiện hữu cần tránh bằng cấu trúc mới: depth residual đang nằm dưới
`encoder.*` nhưng vẫn trainable trong interface warm-up, trong khi parameter
mapper phân nó vào encoder group bị freeze. Toàn bộ depth/feature/focus module
mới phải nằm dưới namespace `bridge.*`; parameter manifest không dùng prefix
ngoại lệ.

CoVeR checkpoint và AFMR checkpoint không tương thích về graph. Strict load
phải fail rõ; không dùng partial shape matching cho `bridge.*`. Nếu cần tận
dụng weight cũ, chỉ allowlist pretrained encoder, decoder và copied
cross-attention với báo cáo đầy đủ missing/unexpected keys.

### Phase 0 — khóa reference và dọn rác an toàn

1. Ghi lại commit hiện tại và trạng thái untracked.
2. Không sửa `src/eviseq_v2`; nó là đường đánh giá checkpoint cũ.
3. Giữ `src/eviseq_new` hiện tại cho đến khi AFMR smoke pass.
4. Xóa `.pytest_cache`, `__pycache__`, `.DS_Store` khỏi working tree và thêm ignore rules.
5. Tạo namespace `eviseq_afmr`; không port compatibility aliases.

**Done khi:** tree mới import độc lập và chưa phụ thuộc `core` cũ.

### Phase 1 — config và data foundation

1. Viết strict schema validator.
2. Chỉ hỗ trợ một base + task override.
3. Viết canonical record and prepare pipeline.
4. Tạo top-3 positive metadata cho train/validation.
5. Viết collator offset mapping/truncation/allocation target.
6. Tests config/data pass.

**Done khi:** `prepare`, `validate-data` và collator chạy không load model.

### Phase 2 — encoder và AFMR

1. Định nghĩa `EncoderState`.
2. Port tối thiểu pretrained-native PPLX encoder wrapper.
3. Port Qwen backend riêng nếu cần ablation.
4. Implement controller.
5. Implement depth residual.
6. Implement feature residual.
7. Implement vectorized multi-scale focus.
8. Implement diagnostics detached.
9. Tests shape/identity/mask/gradient pass.

**Done khi:** bridge exact-anchor ở initialization, không mất token và không NaN BF16.

### Phase 3 — decoder integration

1. Port Qwen decoder wrapper tối thiểu.
2. Copy self-attention weights sang cross-attention.
3. Add bounded per-layer cross gate.
4. Pass source prior qua SDPA additive mask.
5. Implement modern Cache API và cross-KV reuse.
6. Tests cached/non-cached parity pass.

**Done khi:** greedy tiny decode đúng và bridge chỉ chạy một lần.

### Phase 4 — losses và training

1. Implement CE.
2. Implement allocation KL với valid-sample reduction.
3. Implement exact parameter manifest.
4. Implement warm-up/full freeze policies.
5. Implement optimizer-state carry giữa stages.
6. Implement zero-warmup branch.
7. Atomic epoch/last checkpoints.
8. Compact logs và teacher-forced validation.

**Done khi:** gradient contract pass và interrupted run resume đúng.

### Phase 5 — evaluation

1. Batched greedy generation.
2. Streaming JSONL write và resume by ID.
3. Progress, docs/s, ETA và VRAM.
4. Diagnostic metrics.
5. Isolated ROUGE-1.5.5 command.
6. Official last-checkpoint command.

**Done khi:** một interrupted smoke evaluation tiếp tục đúng record kế tiếp và output không đổi theo batch size.

### Phase 6 — server smoke và profiling

1. Run smoke on one B200.
2. Check BF16, checkpoint, generation/cache.
3. Profile AFMR overhead.
4. Sửa bottleneck trước pilot.

**Done khi:** toàn bộ gates ở Section 18.2 pass và overhead đạt target.

### Phase 7 — controlled experiments

1. Pilot A0–A7.
2. Full PubMed A0/A3/A4.
3. Official greedy test + Perl ROUGE-1.5.5.
4. Paired bootstrap.
5. Generalization runs.

**Done khi:** có bảng tách rõ lợi ích depth, feature, focus và allocation; không suy luận từ train diagnostics.

### Phase 8 — xóa implementation cũ

Chỉ sau khi smoke, checkpoint roundtrip và full import tests pass:

- Xóa `core/modeling/architecture.py` cũ.
- Xóa CoVeR/PCEB/dual-bridge/slot code.
- Xóa `training/objectives.py` cũ và `online_kd.py`.
- Xóa classification/QA/translation templates không thuộc summarization package.
- Xóa model/task configs `corrected`, `strong`, `continue`, `cover`.
- Xóa scripts GPU hard-coded khỏi package; giữ example commands trong README.
- Xóa cache artifacts.

Chạy static search và yêu cầu không còn các term:

```text
DualBridge
PCEB
CoVeR
coverage_slot
prompt_probe
memory_bank
source_swap
geometry_loss
online_kd
ranking_phase
fingerprint
hashlib
target_total_footprint_approx
```

**Done khi:** package mới không import file cũ, test pass từ clean environment và README chỉ mô tả AFMR.

---

## 21. CLI mục tiêu

Từ `src/eviseq_new`:

```bash
bash scripts/run.sh prepare pubmed --source-dir /absolute/path/to/pubmed

CUDA_VISIBLE_DEVICES=0 \
bash scripts/run.sh train configs/pubmed.yaml --overwrite-output-dir

CUDA_VISIBLE_DEVICES=0 \
bash scripts/run.sh train configs/pubmed.yaml \
  --resume-checkpoint runs/afmr/pubmed_pplx_qwen/epoch_004.pt

CUDA_VISIBLE_DEVICES=0 \
bash scripts/run.sh evaluate \
  configs/pubmed.yaml \
  runs/afmr/pubmed_pplx_qwen/last.pt \
  runs/afmr/pubmed_pplx_qwen/last_test_predictions.jsonl \
  --split test \
  --batch-size 96 \
  --resume

export PYROUGE_HOME_DIR=/absolute/path/to/ROUGE-1.5.5
bash scripts/run.sh rouge155 \
  runs/afmr/pubmed_pplx_qwen/last_test_predictions.jsonl \
  --details
```

Hai GPU dùng DDP data parallel, không phải `CUDA_VISIBLE_DEVICES=0,1` rồi
vẫn khởi tạo một process:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 run.py train \
  --config configs/pubmed.yaml \
  --overwrite-output-dir
```

`training.batch_size` là physical batch **mỗi rank**. Config/CLI phải in
`world_size` và effective batch trước khi load model.

Runtime override hợp lệ:

```bash
bash scripts/run.sh train configs/pubmed.yaml \
  --set training.batch_size=32 \
  --set training.gradient_accumulation_steps=3
```

CLI phải validate path và in resolved runtime values trước khi load model. Không cho sửa một file `resolved_config.yaml` trong run directory để thay batch hoặc model path.
`--resume-checkpoint` và `--overwrite-output-dir` loại trừ nhau.
`--init-checkpoint` nếu được giữ chỉ được phép allowlist weight
encoder/decoder/cross-attention cho một run mới; nó không được gọi là resume.

---

## 22. Rủi ro và biện pháp chặn

| Rủi ro | Dấu hiệu | Biện pháp |
|---|---|---|
| Focus prior học shortcut vị trí | scale nhỏ thắng mọi sample, prior giống lead bias | xem scale weights theo dataset/length; no-prompt/single-scale ablation |
| Allocation tốt nhưng ROUGE không tăng | alloc loss giảm, pos mass tăng, ROUGE đứng | giảm/tắt allocation; giữ architecture CE-only; không thêm loss khác |
| Bounded gates quá nhỏ | delta/cross ratios gần zero và A1–A4 ngang A0 | thử một sensitivity gate cap/init, không đổi nhiều biến cùng lúc |
| Gates quá mạnh phá pretrained path | loss spike, residual ratio lớn, ROUGE giảm | hạ gate max/init; xác nhận zero-init |
| Hidden taps tốn VRAM | peak tăng lớn so A0 | chỉ capture bốn final layers qua backend hooks; giải phóng taps sau bridge |
| SDPA rơi về math kernel | train/eval chậm bất thường | profile kernel; dùng supported additive mask shape/dtype; không materialize attention |
| Prompt không mang thông tin trong một task | no-prompt ngang main | trình bày prompt conditioning là generality mechanism, không claim PubMed gain |
| Budget cố định trong một task | no-budget ngang main | giữ cho multi-task/general setting hoặc bỏ claim adaptive budget |
| Evidence labels bị leakage | test output đổi khi label đổi | hard test; test dataloader không tạo allocation target |
| Length mismatch tạo ROUGE gap | too-short rate/length ratio xấu | sửa generation budget công bằng, không dùng beam hay target length oracle |

---

## 23. Quy tắc ra quyết định sau thí nghiệm

1. Không có kiến trúc nào được tuyên bố “chắc chắn vượt T5Gemma” trước full official evaluation.
2. Main target là vượt ít nhất một trong ROUGE-1/2/L với cùng split, preprocessing, max lengths, greedy decode và ROUGE-1.5.5.
3. Nếu AFMR tăng một metric nhưng giảm hai metric, báo đúng trade-off; không chọn checkpoint/test setting theo metric thuận lợi sau khi nhìn test.
4. Nếu allocation diagnostics tốt mà ROUGE không tăng, kết luận supervision không chuyển thành generation; không tự động tăng loss weight.
5. Nếu A3 tốt hơn A0 nhưng A4 không tốt hơn A3, giữ CE-only AFMR làm main architecture.
6. Nếu depth hoặc feature residual không giúp, bỏ component đó; novelty phải dựa vào phần có ablation hỗ trợ.
7. Nếu PubMed tốt nhưng CNNDM/WikiLingua không tốt, thu hẹp claim thành long/scientific summarization thay vì gọi general-purpose.
8. Mọi số paper phải đến từ `last.pt`, greedy predictions và Perl ROUGE-1.5.5; Python ROUGE chỉ để debug.

---

## 24. Definition of done

### Code

- [ ] Package `eviseq_afmr` độc lập, không import `core` cũ.
- [ ] Không có fingerprint/hash/reporting/footprint logic.
- [ ] Config schema tối giản và reject legacy keys.
- [ ] Main forward giữ full source memory và một source bias.
- [ ] CE và allocation loss có gradient đúng.
- [ ] Warm-up 0 hoặc 1 đều chạy.
- [ ] Per-epoch + `last.pt` save/resume đúng.
- [ ] Greedy cache và resume evaluation đúng.
- [ ] Unit/integration tests pass trong `bienkieu_env`.
- [ ] Không có model download trong local tests.

### Architecture

- [ ] Zero-init giữ đúng baseline path.
- [ ] Depth/channel/span gates đều bounded.
- [ ] Không target/reference leakage trong deployed graph.
- [ ] Teacher-forcing/generation parity pass.
- [ ] AFMR overhead dưới profiling budget.

### Experiment

- [ ] A0–A7 pilot hoàn tất.
- [ ] Full PubMed main và baseline cùng protocol.
- [ ] ROUGE-1.5.5 + paired bootstrap.
- [ ] Length/repetition/latency/VRAM được báo.
- [ ] Ít nhất một non-scientific dataset kiểm tra generality.

### Paper

- [ ] Claim về report dùng “motivated by/adapted”, không overclaim.
- [ ] Công thức paper khớp code và tensor shapes.
- [ ] Chỉ component có ablation dương mới được gọi là contribution hiệu quả.
- [ ] Không dùng auxiliary metric làm bằng chứng thay ROUGE.
- [ ] Parameter count được đo từ checkpoint/code, không lấy từ YAML.

---

## 25. Câu chuyện paper nếu AFMR được thực nghiệm xác nhận

1. Việc ghép hai LLM pretrained nhỏ đã gần chất lượng encoder–decoder jointly pretrained, nhưng interface đơn giản vẫn để lại một khoảng cách nhỏ.
2. Các interface evidence cũ thường chọn một representation depth, một granularity hoặc một training-only selector; auxiliary success không bảo đảm generation success.
3. AFMR coi summarization là adaptive information allocation theo depth, feature channel và source span.
4. Nó bảo toàn một full-memory main path, khởi tạo residual bằng zero, chặn gate và chặn độ mạnh của focus prior. Norm của depth/feature residual vẫn phải được theo dõi thực nghiệm.
5. Main học hoàn toàn qua CE; allocation supervision được đánh giá riêng để phân biệt đóng góp kiến trúc và nhãn phụ. Inference không cần label, model phụ, beam hay reranker.
6. Kết quả và ablation phải cho thấy phần nào thực sự thu hẹp khoảng cách T5Gemma; efficiency chứng minh lợi thế ghép model nhỏ.

Đây là câu chuyện thống nhất đủ mới để nghiên cứu, nhưng vẫn có thể bị falsify bằng ablation. Đó là tiêu chuẩn đúng cho một kiến trúc nghiêm túc.

## 26. Implementation audit — 2026-09-05

Không thêm thành phần kiến trúc hoặc objective ngoài thiết kế. Các sửa chữa đưa code về đúng công thức:

- Controller đúng source + prompt + ba budget features, với nhánh budget khởi tạo nhỏ.
- Depth là encoder-space low-rank residual; feature residual ánh xạ encoder-space sang decoder-space.
- Window index theo content của từng document, không theo chiều padded của batch. Pool/lift dùng prefix sums và scatter overlap-add tương đương masked region means, không tạo dense region-by-token matrix hay cache GPU không giới hạn.
- Region scorer có RMSNorm; scale không phân biệt được vị trí thì không đóng góp prior hoặc allocation loss.
- Reference-derived sentence indices chỉ tạo training/validation allocation targets. Model forward không nhận indices; greedy giữ full memory. CE-only hoạt động không có positive labels.
- Collator tokenize toàn source một lần với offset mapping. Prepare dùng Unicode, giữ list boundaries, tái tạo nhãn của chính AFMR và ghi atomically theo streaming.
- Encoder chỉ capture taps cần dùng và tắt KV cache; gradient checkpointing non-reentrant. Decoder greedy mask EOS trước argmax, có attention/position masks và cache parity.
- CE được tính theo token chunks với recomputation để giảm peak logits memory; objective không thay đổi. Accumulation weighted theo số token CE và số document allocation thực tế, kể cả partial window. Carry Adam moments qua hai stage; scheduler linear decay theo optimizer steps.
- Checkpoint deserialize ở CPU; runtime batch/paths lấy từ config đang truyền. Smoke dùng thư mục mới, không overwrite run thật. Evaluation kiểm tra ID/reference prefix và flush mỗi batch.

Finite sigmoid initialization yêu cầu gate init nằm strictly giữa 0 và cap; temperature init nằm strictly giữa min và max. Không âm thầm clamp giá trị ngoài/bằng biên. Zero-init nói đến output factors, không phải sigmoid gates.

Phân biệt đã verify và chưa verify: tests offline kiểm tra tensor/gradient, BF16 CPU SDPA backward, cache parity, label isolation, accumulation và warm-up resume. Cần chạy smoke/profiling trên PPLX/Qwen thực và CUDA/B200 trước khi full train. Runner hiện tại single-process; DDP và automatic OOM fallback chưa được triển khai. Không suy luận hiệu quả ROUGE từ các unit tests.

## 27. Token-wise depth + CE-only implementation — 2026-09-05

- Thay document-wide depth weights bằng `softmax(u_d^T Norm(H_t^j) + (W_depth c)_j)` trên từng token; giữ nguyên low-rank depth residual, feature residual, multi-scale prior và decoder.
- Shared token scorer thêm `d_e` tham số; tại hidden size 1024, bridge có 1,981,446 tham số. Không thêm layer, memory bank, decoder output gate hoặc objective mới.
- Main config và mọi task kế thừa dùng CE-only. Runtime bỏ allocation metadata khi weight bằng zero; raw text/summary được chấp nhận. Prepare có `--no-allocation-target` để chuẩn hóa dữ liệu mà không chạy miner.
- Allocation KL vẫn có thể bật riêng ở weight 0.10, tối đa ba positive, không thay graph inference.
- Graph checkpoint đổi sang `afmr_token_depth_lowrank_v3`. Checkpoint document-wise cũ bị từ chối; không force-load hoặc ghi đè checkpoint cũ. Khởi tạo từ pretrained backbones cho main run mới.
- Test xác nhận weights khác theo token khi candidate states khác, softmax theo depth, không ảnh hưởng token khác khi controller cố định, uniform/anchor ở initialization và CE-only gradient vào token scorer/router sau optimizer update.
- Không claim identity tạo sẵn alignment giữa hai LLM; không claim bounded gate tạo norm bound cho residual hoặc bảo đảm tăng ROUGE.


## 28. CE-only cleanup — 2026-09-05

- Giữ nguyên token-wise AFMR graph và toàn bộ tham số kiến trúc. Không thêm loss hoặc nhánh mới.
- Loại bỏ allocation loss/ramp, sentence positives, miner và tensor unit/allocation khỏi data/model/trainer. Raw `text/summary` đủ cho cả train và validation; nhãn ngoài không được sử dụng.
- Prepare chỉ chuẩn hóa `id/text/summary`, không có cờ test/mining. Config không còn section `objective`.
- Bỏ diagnostic reductions và anchor projection chỉ phục vụ logging khỏi forward. Không thay đổi memory hoặc source prior.
- Log train: stage, global epoch, optimizer step, CE weighted theo token, gradient norm, learning rates và step_s. Giữ checkpoint, validation CE, eval progress/ETA.
- Bộ test tiếp tục kiểm tra CE gradient, mask, cache parity, accumulation, resume và label isolation. Không dùng kết quả test chức năng để khẳng định vượt baseline.
