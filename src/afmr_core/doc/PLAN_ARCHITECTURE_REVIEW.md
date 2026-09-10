# Review lịch sử và phản biện plan/kiến trúc AFMR

Ngày: 2026-09-10.
Phạm vi: `README.md`, `DESIGN.md`, `IMPLEMENTATION_PLAN.md`, `ARCHITECTURE_AUDIT.md`, `DECISION_REPORT.md`, `EVALUATION_PLAN.md`, `RESEARCH.md`, đối chiếu code `eviseq_new` và `eviseq_update` tại commit được audit (`88486a25`). Tại thời điểm review chưa có implementation AFMR, checkpoint hay prediction để đo ROUGE; package hiện tại và bằng chứng triển khai nằm trong `IMPLEMENTATION_EVIDENCE.md` và `VERIFICATION_REPORT.md`.

Các verdict “chưa sẵn sàng code” bên dưới là trạng thái trước implementation và được giữ như lịch sử phản biện, không phải trạng thái hiện tại của package.

Review này không lặp lại quyết định đã chốt. Nó phản biện chỗ spec mâu thuẫn với code đang chạy, chỗ công thức không làm được điều nó tuyên bố, và chỗ protocol có thể biến một ablation thành claim giả.

## Kết luận trước

V5.1 là một chương trình thí nghiệm có kỷ luật hơn v4: giữ trunk đã kiểm chứng, bỏ hierarchy/planner, tách fusion khỏi decoding gauge, khóa source-empty, và từ chối trộn contrastive/rerank vào vòng CE. Những ràng buộc đó nên giữ.

Kiến trúc main **chưa sẵn sàng code**. Có bốn lỗ hổng phải sửa trên giấy trước khi viết `dual_readout.py`:

1. Công thức reader trong `DESIGN.md` không trùng graph v2 mà C2 bắt buộc tái lập.
2. `base_floor` không phải floor khi copy gate lớn; yêu cầu của `critic_eval_novelty` chưa được hiện thực.
3. Identity `Wo=0 ⇒ P = P_new` đúng đại số nhưng **sai bit** nếu implement đúng chữ `(1-g-α)P0 + αP0`.
4. Router không có tín hiệu khi `Wo=0`, trong khi mass semantic tại init chỉ khoảng `0.05`. Main `AFMR-safe` có thể không bao giờ đánh thức được nhánh mà nó muốn đo.

Giả thuyết trung tâm — nested gate `(1-g)β` làm mất R1/RL của `new` và cần mixture để giữ R2 — **chưa có ablation kiểm soát**. Chênh lệch đã báo (`R1 −0.138`, `R2 +0.052`, `RL −0.119`) nhỏ hơn nhiễu recipe đã biết (batch 96 vs 168, epoch, clip). Common-control A vs B phải là **cổng dừng**, không phải một hàng trong bảng.

Verdict: plan đủ tốt để làm thí nghiệm, chưa đủ chặt để implement. Sửa spec, siết kill-gate, rồi mới dựng package.

## 1. Những gì nên giữ

- Tách probability fusion khỏi hidden fusion, và bắt buộc control `hidden-interpolation` + `constant-alpha`. Đây là bài học đúng từ Mixture of Softmaxes, không phải phát minh.
- Giữ copy alignment/marginalization của `new`; không nhét semantic vào copy keys.
- `log(P)+Z0` khớp `mix_logits` hiện tại của `new` (`grounded_copy.py`, cộng `logsumexp(z0)` sau `logaddexp`). Probe `repetition_penalty=1.0` là cần, vì processor của `generate.py` phụ thuộc dấu logit.
- Mixture NLL bằng `logsumexp` ba nhánh, cấm `Σ π_k CE_k`. Code `new`/`v2` đã đi đúng đường `logaddexp(lm_target + log(1-g), copy_target + log g)`.
- Không đưa planner, 4×128, contrastive, BRIO, SDACL vào main.
- Không claim novelty cho copy mixture, MoS, zero-init, hay sigmoid gate.
- Artifact list (resolved config, Perl ROUGE raw, sample IDs, truncation stats) là điều kiện hợp lệ duy nhất cho bảng headline.

Những điểm trên không bị phản biện. Phần còn lại của tài liệu mới là vấn đề.

## 2. Finding phải sửa trước khi code

### F1 — Severity: blocker. Reader main không trùng v2

`ARCHITECTURE_AUDIT.md` ghi đúng code v2:

```text
q  = Wq RMS(h)
Ks = RMS(Wk RMS(H0))
Vs = Wv RMS(H0)
```

`DESIGN.md` §2 viết:

```text
Ks = Wk RMS(H0)
Vs = Wv RMS(H0)
```

Code thực (`eviseq_update/modeling/grounded_copy.py`, independent read):

```text
normalized_memory = RMS(H0)
semantic_keys     = RMS(Wk(normalized_memory))
semantic_values   = Wv(normalized_memory)
```

C2 đòi `legacy_semantic` trùng logits/loss/gradient v2 flat. Nếu implement theo `DESIGN.md`, C2 fail dù C1 pass. `IMPLEMENTATION_PLAN` nói “giữ RMS của v2” nhưng không chỉ RMS nào.

**Sửa:** chốt một công thức, copy nguyên từ v2, và ghi rõ phép RMS sau `Wk` là invariant của endpoint B. Mọi khác biệt reader phải là named variant, không phải silent drift trong main.

### F2 — Severity: blocker. `base_floor` không giữ floor

Công thức main:

```text
α      = min(α_raw, max(0, 1 − g − base_floor))
π_copy = g
π_sem  = α
π_base = 1 − g − α
```

Khi `g > 1 − base_floor`, `α = 0` và `π_base = 1 − g < base_floor`. Với `base_floor=0.05`:

| g | π_copy | π_sem max | π_base thực | floor có giữ? |
|---|---:|---:|---:|---|
| 0.10 | 0.10 | 0.20 | ≥ 0.70 | có |
| 0.80 | 0.80 | 0.15 | 0.05 | có, vừa khớp |
| 0.97 | 0.97 | 0 | 0.03 | **không** |
| 0.99 | 0.99 | 0 | 0.01 | **không** |

`critic_eval_novelty` yêu cầu floor cho base **và** copy. Independent simplex có floor base thật. Copy-preserving chỉ chừa phần dư sau copy. Tên `base_floor` đang che một ràng buộc khác: “không để semantic ăn phần residual của generate”.

Hệ quả: khi copy gate cao — đúng những token R2 phụ thuộc — base expert bị nén dưới 0.05, semantic bị tắt. Đó không phải “copy-preserving plus base safety”. Đó là copy thắng tuyệt đối, semantic không được vào.

**Sửa, chọn một và viết vào DESIGN:**

- A. Giữ công thức hiện tại, đổi tên thành `generate_reserve`, ghi rõ không bảo đảm `π_base ≥ 0.05`.
- B. Floor thật: `π_copy = min(g, 1 − base_floor)`, rồi cap α trên phần còn lại. Mất identity với `g` legacy khi g lớn.
- C. Softmax có mask + floor như independent, rồi mới nói “capped simplex”.

Không được để main và independent dùng hai nghĩa khác nhau của cùng một hằng số.

### F3 — Severity: blocker. Identity endpoint không ổn định số

Đại số: `Wo=0 ⇒ δ=0 ⇒ Ps=P0 ⇒ P=(1−g−α)P0 + αP0 + g Pcopy = (1−g)P0 + g Pcopy`. Đúng.

Implement theo chữ sẽ tính hai softmax rồi cộng. Với FP32:

```text
(1 − g − α) P0 + α P0    ≠    (1 − g) P0
```

C4 (`rtol=1e-5, atol=1e-6`) có thể fail trên hàng vocab ~1.5e5 của Qwen, đặc biệt token có xác suất nhỏ. `new` không đi đường này: nó dùng một `logaddexp` trên `P0` và `Pcopy` (`mix_logits`).

**Sửa:** khi `Ps` trùng `P0` (hoặc `α=0`), fallback đúng kernel `new`. Test C4 so với `mix_logits` của package `new` trên cùng weights/input, không so với một oracle tự viết lại bằng ba mix. Ghi rõ identity là “cùng đường generate+copy của new”, không phải “hai softmax cộng lại gần đúng”.

### F4 — Severity: blocker. Independent simplex chưa phải một mô hình

`DESIGN.md` §3:

```text
q = softmax(l_base, l_sem, l_copy)
π_base = base_floor + (1 − base_floor) q_base
π_sem  = (1 − base_floor) q_sem
π_copy = (1 − base_floor) q_copy
```

Không định nghĩa `l_*`. Không nói chúng có phải affine của `[q,u,g_logit,…]`, ba logit tự do, hay `l_copy` được neo vào copy gate. Không có init để `π_copy≈g`, `π_sem≈0` lúc bước 0. Không có mask expert khi copy/semantic source rỗng ngoài một câu “có thể mask”.

Không có identity với `new`. F không phải control của D: khác router, khác simplex, khác init, khác quan hệ với `g`. Nếu F thắng hoặc thua D, không biết biến nào gây ra.

**Sửa trước khi xếp F vào ma trận bắt buộc:**

- Công thức `l_*`.
- Stop-gradient hay không trên `g`.
- Init: ví dụ `l_sem → −∞`, `l_copy = logit(g)`, `l_base = 0`, rồi kiểm tra `π` khớp `(1−g, 0, g)` sau floor.
- Hard mask: copy rỗng ⇒ `l_copy=−∞`; semantic rỗng ⇒ `l_sem=−∞`; cả hai rỗng ⇒ `P=P0` không qua softmax ba nhánh.

## 3. Phản biện giả thuyết trung tâm

Plan xây cả AFMR trên câu chuyện:

> v2 sửa hidden rồi mới softmax, nên mất R1/RL của `new` trong khi giữ/nhỉnh R2; mixture với `P0` neo sẽ lấy lại R1/RL mà không mất R2.

Ba điểm yếu.

### 3.1. Hiệu ứng đã báo nhỏ hơn nhiễu recipe

| So sánh | ΔR1 | ΔR2 | ΔRL |
|---|---:|---:|---:|
| v2 − new (điểm đã báo) | −0.138 | +0.052 | −0.119 |
| T5Gemma − new | −0.046 | +0.089 | −0.432 |

`ARCHITECTURE_AUDIT` đã nói new dùng epoch 4, 1 GPU, 48×accum2 (global 96, clip 1.0); update v1 dùng epoch 5, 2 GPU, 84×accum1 (global 168, từng bỏ clip). Tỷ lệ optimizer update ~5/7. Điểm v2 `49.488/21.953/45.776` **không có** resolved config, prediction, hay seed. Không được dùng làm ground truth cho “semantic residual làm hỏng unigram”.

Nếu A và B dưới common recipe không tái lập trade-off R1↓ R2↑, chương trình mixture không còn đối tượng. EVALUATION_PLAN liệt kê A/B như baseline, không như kill-gate. Đó là sai thứ tự khoa học.

**Sửa protocol:** sau C1/C2, train A và B một seed common-control. Quyết định:

- Nếu `|ΔR1|, |ΔRL| < MDE` đã đăng ký: dừng mixture. Báo negative result. Không thêm α, K=M, null slot.
- Nếu trade-off tái lập: mới mở D/E/F.
- MDE phải viết trước. Với PubMed, ΔR2=0.05 trên một seed không phải tín hiệu.

### 3.2. Nested gate không phải cơ chế duy nhất đã đổi

v2 khác new ở tối thiểu bốn chỗ cùng lúc:

1. Independent semantic Q/K/V trên `H0` native, không pool theo copy token.
2. Residual `h+δ` trước `W_lm`.
3. Scalar semantic gate init 0.05.
4. RMS cap 0.10.

AFMR gán regression cho (2) bị nhân `(1-g)`, rồi thay bằng mixture. Nó đồng thời **bỏ (3)**. Đó là hai thay đổi. Nếu D thua B, không biết vì mixture hay vì mất residual gate.

`constant-alpha` và `hidden-interpolation` không hoàn lại gate v2. Control thiếu: **v2 graph nguyên**, chỉ đổi chỗ fusion thành mixture, giữ nguyên inner gate. Hoặc ngược lại: v5 reader không gate, nhưng fusion hidden như v2. Hiện H dùng `softmax(W(h+αδ))` với α là mixture mass — một hệ số xác suất bị nhét vào hidden. Đó không phải ablation vị trí fusion sạch.

### 3.3. Trần mass semantic có thể nhỏ hơn vấn đề cần sửa

v2 sau khi học: toàn bộ mass generate `(1−g)` đi qua `softmax(W(h+δ))`. `δ` bị cap RMS 0.10 và gate, nhưng **mọi token generate** thấy phân phối đã sửa.

AFMR sau khi học: semantic chỉ được `α ≤ 0.20` mass. `P0` luôn còn. Khoảng cách TV tới phân phối copy-only của `new` bị chặn:

```text
‖P − ((1−g)P0 + g Pcopy)‖₁ / 2  ≤  α  ≤  0.20
```

Mixture không thể sửa quá 20% mass generate. Plan không biện minh 0.20 từ dữ liệu `g` của `new`. Nếu `g` trung bình ~0.1 thì generate còn 0.9, và v2 được phép sửa cả 0.9; AFMR chỉ 0.20. Nếu giả thuyết “semantic hữu ích nhưng quá mạnh” đúng, 0.20 có lý. Nếu giả thuyết sai và v2 chỉ yếu vì recipe, 0.20 biến D thành `new` cộng nhiễu nhỏ — sẽ “parity” rồi tuyên bố mixture không giúp, trong khi chưa bao giờ cho semantic đủ mass.

Config cho phép `alpha_max ∈ {0.10, 0.25}` validation-only, nhưng ma trận bắt buộc không có sweep `alpha_max`. Đó là hyperparameter của chính phương pháp, không phải chi tiết runner.

**Sửa:** đo phân phối `g` trên validation của A (`new` common-control) trước khi khóa `alpha_max`. Thêm một control `alpha_max=0.50` hoặc `1−g−base_floor` không cắt 0.20, một seed, trước full matrix. Nếu 0.50 không hơn 0.20, lúc đó mới tin trần 0.20.

## 4. Phản biện tối ưu hóa: main đang là biến chậm nhất

### 4.1. Gradient tại `Wo=0`

Gọi `P = (1−g−α)P0 + α Ps + g Pcopy`.

Khi `Wo=0`, `δ=0`, `Ps=P0`, nên `∂P/∂α = 0`. Router `r_sem` không nhận tín hiệu từ NLL. `∂δ/∂u = Wo = 0`, nên Q/K/V cũng không nhận tín hiệu qua residual. `Wo` vẫn có

```text
∂L/∂Wo  ∝  α · (Ps − P0) …   nhưng Ps=P0 nên term mixture theo expert-diff = 0
```

Cẩn thận hơn: `Ps = softmax(W(h + Wo u))`, tại `Wo=0` vẫn còn đạo hàm theo `Wo`:

```text
∂zs/∂Wo = W_lm u
∂L/∂Wo  = α · (∂NLL/∂Ps) · (∂Ps/∂zs) · W_lm u
```

Hữu hạn, tỷ lệ với `α`. Q/K/V: chain qua `u` nhân `Wo=0` nên **đúng bằng không** trước bước optimizer đầu. Plan nói “sau update đầu, Q/K/V phải có gradient khác zero”. Đúng trên giấy, yếu trên thực tế: `Wo` sau một bước Adam với `α≈0.05` vẫn rất nhỏ, gradient Q/K/V bị hai hệ số nhỏ nhân nhau.

v2 không bị vậy theo cùng cách. Residual nằm trong generate path mass `(1−g)≈0.9`, và inner gate 0.05 nhân `δ` chứ không nhân `∂L/∂Ps`. `W_out` zero-init vẫn nhận gradient đầy đủ từ CE generate.

V5.1 **chuyển hệ số nhỏ từ quy mô residual sang quy mô mixture**. Init `α≈0.05` gần bằng `semantic_gate_init` của v2, nhưng ý nghĩa khác: 0.05 của v2 là scale của `δ`, 0.05 của AFMR là trọng số expert. Gradient semantic yếu hơn khoảng `(1−g)/α ≈ 10–20×` lúc đầu.

`AFMR-tiny` tồn tại đúng vì lý do này, nhưng main vẫn là `Wo=0`. Parity control không nên là ứng viên chất lượng. Đảo vai: `tiny` là train candidate; `zero` chỉ để C4 và một run ngắn chứng minh endpoint, không full-train ba seed trừ khi tiny đã thắng.

### 4.2. Router đọc copy features làm thủng “copy-preserving”

Router input: `[q, u, g_logit, copy_entropy, copy_support_mass]`.

`g_logit` vừa định nghĩa `π_copy=g`, vừa là feature của `α`. NLL semantic backprop vào copy gate. Copy không còn độc lập sau optimizer, ngay cả khi simplex giữ `π_copy=g` trong một forward.

Đây không phải lỗi lý thuyết nhỏ. Cả luận điểm “semantic không lấy mass copy” dựa trên `g` là copyleaving. Nếu semantic CE đẩy `g` xuống để lấy chỗ cho `α` (vì `α ≤ 1−g−floor`), copy-preserving chỉ đúng trong một bước, sai qua training.

**Sửa:** `stop_gradient` trên mọi copy feature vào router. Copy gate chỉ học từ copy mixture như `new`. Nếu muốn router thấy copy state, đó là named variant `router_sees_copy`, không phải main.

### 4.3. Bỏ inner gate mà không có off-switch lúc đầu

v2: `δ = σ(W_gate[q;u]) · Wo u`, rồi RMS cap. Gate có thể về 0 theo token.

AFMR: `δ = cap(Wo u)`. Khi `Wo` lớn, `δ` ngồi gần trần `0.10 RMS(h)` mọi token. Off-switch duy nhất là `α`. `α` không học khi `Wo=0` (F4.1). Khoảng thời gian đầu: residual bắt đầu lớn dần, router ngủ, copy gate vẫn sống. Shared trunk nhận gradient từ copy + từ một expert gần như `P0`. Đây đúng kịch bản “shared trunk làm yếu copy” mà DECISION_REPORT lo, nhưng plan không có freeze-copy hay stop-grad `h` cho copy head.

Control thiếu, một seed: freeze `grounded_copy` (hoặc stop-grad `h` trước copy query/gate), chỉ học semantic + router. Nếu freeze-copy thắng unfreeze, luận điểm copy-preserving ở **output simplex** là không đủ; cần bảo vệ ở **parameter**.

## 5. Phản biện đánh giá và điều kiện thắng

### 5.1. Thắng hệ thống bị trộn với thắng kiến trúc

EVALUATION_PLAN: candidate thắng khi một checkpoint `R1>49.626`, `R2>21.990`, `RL>45.895`, ba seed, bootstrap, không lấy R1/R2/RL từ checkpoint khác nhau.

Hai vấn đề.

`21.990` là R2 T5Gemma, backbone khác, recipe khác. Plan đúng khi không ép T5Gemma dùng optimizer EviSeq, rồi mâu thuẫn khi bắt D phải vượt T5Gemma mới được gọi là thắng. Thành công kiến trúc là **D hơn A và B cùng recipe**. Thành công hệ thống là best EviSeq vs T5Gemma, báo riêng.

Ngưỡng tuyệt đối `49.626` là điểm lịch sử chưa tái lập. Sau common-control, mốc phải là **chính A và B vừa train**, không phải hàng README. Nếu A common-control ra R1=49.40, đòi D > 49.626 là đòi D thắng cả kiến trúc lẫn một run cũ không tái lập.

### 5.2. Ba seed không cứu Δ = 0.05

Primary comparisons: D vs A (R1, RL), D vs B (R2), D vs T5Gemma (R2). Plan nói adjust cho ba so sánh. Thực tế còn F vs D, H vs D, I vs D, E vs D, và ba metric. Family-wise error lớn.

Bootstrap 10,000 trên document không thay variance giữa seed. PubMed abstractive với Qwen 0.6B + encoder 0.6B, ΔR2=0.05 thường nằm trong spread ba seed. Pre-register:

- Metric chính: R1, hoặc trung bình (R1+RL)/2, không phải “thắng đồng thời ba số”.
- MDE: ví dụ 0.20 R1 / 0.10 R2, hoặc khoảng tin cậy seed không chứa 0.
- Nếu không đủ GPU cho 3 seed × 4 candidate: 1 seed common-control A/B trước, 3 seed chỉ cho cặp vào được vòng 2.

### 5.3. Ngân sách compute không có

Ma trận: A–I cộng gauge-probe, tối thiểu A/B/D/F × 3 seed. PubMed 4–5 epoch, global batch 96, source 4096, **hai** vocab readout (~152k). Chưa kể C9 1/2 GPU, dense/chunked profile, C11 repetition probe.

Không có GPU-day, không có thứ tự kill, không có “pilot seed chỉ nhìn val CE + copy rate, không nhìn test”. Nguy cơ cổ điển: chạy hết variants rồi chọn D hoặc F theo test.

**Sửa thứ tự thật:**

1. C1–C15 trên tiny + 1 batch CUDA.
2. A và B, 1 seed, common recipe. Kill-gate trade-off.
3. D và H, 1 seed. Nếu H ≥ D: dừng claim mixture.
4. Chỉ lúc đó: E, F, I, G, rồi 3 seed cho cặp còn sống.
5. T5Gemma so sánh hệ thống, không phải cổng architecture.

### 5.4. Bug đánh giá cũ phải vào acceptance

Luna review v3: `evaluate()` có thể trả ROUGE từ file prediction cũ khi checkpoint thiếu/sai. Plan copy `evaluation/generate.py` và runner. C13 kiểm resume; không kiểm “refuse stale predictions”. Đây từng là finding nghiêm trọng về tính hợp lệ thực nghiệm, không phải nit.

Thêm C16: generate/eval từ chối prediction không khớp `(checkpoint, config fingerprint, split hash, decoder revision)`. Test regress với file JSONL cũ cố ý để sẵn.

## 6. Spec mơ hồ sẽ sinh implementation lệch

| Chỗ | Spec hiện tại | Rủi ro |
|---|---|---|
| `semantic_evidence` | 0 khi mask rỗng; null slot dùng `1−A_null` | Main có thể bị implement như mass liên tục từ RMS/attention. C6 chỉ test mask rỗng. Ghi `evidence ∈ {0,1}` cho main. |
| Source prior `b` | Cộng vào semantic score, copy score, và cross-attn | Một prior, ba lần. `AFMR-KM` mới giảm prior. Thiếu control `semantic_prior=0` trên main K=H0. |
| `hidden-interpolation` | `softmax(W(h+αδ))` | α là xác suất. Control sạch hơn: gate residual riêng, hoặc đúng graph v2. |
| `constant-alpha` | ~0.05 | Trùng init, không trùng `alpha_max`. Thêm constant 0.10 và 0.20, hoặc constant = `α_max/2`. |
| Dual readout cost | “gần gấp đôi output head”, chunk vs dense | Qwen3-0.6B vocab ~151936. Chunk bắt buộc trên PubMed B=48. Profile phải là acceptance, không phải nice-to-have: nếu D hơn A 0.05 R1 nhưng −35% throughput, đó là kết quả, không phải footnote. |
| Warmup vs full finetune | Common protocol ghi optimizer/stages | `new` 1 warmup + 3 full; v2 yaml 1+4. Chưa chốt epoch. Common-control phải khóa **số optimizer steps**, không chỉ “4 epoch”. |
| Gauge `C=(1−β)Z0+βZs` | “chỉ diagnostic” | Ai đó sẽ thử rồi giữ nếu ROUGE đẹp. Cấm ghi vào resolved config main. Chỉ log. |
| API `semantic_read` / `readout` | `hs, q, u, evidence` rồi `P, output_logits` | Generation hiện gọi `output_logits` một lần/`mix_logits` một lần. Hai `lm_head`. Cache không được giữ `z0/zs` theo vocab. Spec cache: source-side K/V only. |

## 7. Điểm plan tự mâu thuẫn nhẹ

- README: “semantic không còn bị nhân trực tiếp với `(1-g)β`”. Đúng. Nhưng α vẫn bị cap bởi `1−g`. Nested multiply thành nested cap. Cần nói thẳng: semantic **vẫn phụ thuộc g**, chỉ đổi từ nhân sang phần dư.
- DECISION_REPORT: “không để semantic router lấy mass trực tiếp từ copy”. Đúng trong một forward. Sai qua học nếu router thấy `g_logit` (mục 4.2).
- RESEARCH: đóng góp chỉ đứng nếu D hơn A/B/H/I. IMPLEMENTATION_PLAN/README vẫn viết mục tiêu như thể mixture sẽ thắng. Giữ câu RESEARCH ở đầu README, không chỉ ở cuối.
- “Copy đọc h gốc” được nhấn như bảo vệ. DESIGN §4 tự nói điều này chỉ là topology trước optimizer. Tên `copy-preserving` đang làm việc marketing mà phần rủi ro phải gỡ.

## 8. Việc plan làm tốt hơn các bản trước — không phủ nhận

So với v4/v3: không còn region mass `A[j]≤π[r]`, không usage penalty giả coverage, không tangent+restore-norm đổi thứ tự từ trong khi tuyên bố “giữ câu”. Audit v4 trong `ARCHITECTURE_AUDIT` là phần mạnh nhất của folder này; AFMR đi đúng hướng khi từ chối đưa planner trở lại main.

So với bản v5 nested-gate (trước phản biện chéo): bỏ `(1-g)β` là đúng hướng thí nghiệm. Ba critic không đồng ý K/V và simplex; cách tách `K=H0` main vs `K=M` ablation là hợp lý. Không chọn K/V bằng test.

Những điều đó giữ. Review này không đề xuất quay lại v4.

## 9. Việc phải làm trên giấy, theo thứ tự

1. **Sửa DESIGN.md cho khớp v2 reader** (F1). C2 trở thành so sánh với package `eviseq_update`, không với công thức đã rút gọn.
2. **Đổi tên hoặc đổi công thức `base_floor`** (F2). Viết bảng `g → π` như mục 2.
3. **Identity path = kernel `new`** (F3). C4 gọi `mix_logits` của `eviseq_new` làm oracle.
4. **Spec independent simplex đủ để một người implement ra cùng `π` tại step 0** (F4).
5. **`stop_gradient` copy features trong main router.** Variant thấy `g` phải có tên.
6. **Đảo vai trò zero/tiny:** zero = parity; tiny = train candidate.
7. **Kill-gate A vs B** trước D. MDE và số step bị khóa.
8. **Tách thắng kiến trúc / thắng hệ thống.** T5Gemma không phải cổng D.
9. **Thêm C16 stale-eval.** Thêm control freeze-copy một seed.
10. **Đo `g` trên A, rồi mới khóa `alpha_max`.** Đưa `alpha_max` vào ma trận, không chỉ config comment.

Nếu 1–4 không sửa, implementation sẽ fork thầm giữa DESIGN, AUDIT và code v2, rồi C1/C2/C4 tranh cãi tolerance thay vì sửa graph.

## 10. Câu hỏi plan chưa trả lời mà không nên trả lời bằng intuition

1. Phân phối thực của `g` trên PubMed validation của `new` là gì? Median, P90, tỷ lệ `g>0.8`?
2. Token gold copyable vs non-copyable chiếm bao nhiêu NLL của A và B?
3. Output length và copy rate A vs B dưới **cùng** recipe? Nếu length giải thích ΔR1, mixture là sai chỗ sửa.
4. `Wo=tiny` sau 50 step: `RMS(δ)/RMS(h)` và `α` trung bình là bao nhiêu? Nếu α kẹt 0.05 và δ kẹt 0, D là `new` đắt hơn.
5. Dense dual-softmax thêm bao nhiêu VRAM tại B=48, T_decode=512, V=151936, chunk 1024?

Không có năm số này, `alpha_max=0.20` và `base_floor=0.05` là hằng số thẩm mỹ.

## 11. Tóm tắt phản biện một câu mỗi lớp

| Lớp | Phản biện |
|---|---|
| Giả thuyết | Trade-off R1/R2 chưa được cô lập khỏi recipe; mixture có thể đang giải một hiện tượng không tồn tại. |
| Công thức | Copy-preserving không bảo vệ base khi `g` lớn; identity không bit-stable; independent simplex chưa spec. |
| Reader | DESIGN ≠ v2; C2 đang đo một model khác với B. |
| Tối ưu | Main `Wo=0` + `α≈0.05` làm đói gradient đúng nhánh cần học; router thấy `g` phá độc lập copy. |
| Đánh giá | Ngưỡng tuyệt đối + T5Gemma + 8 variant là điều kiện thắng vừa quá mạnh vừa dễ p-hack. |
| Novelty | RESEARCH đúng; README vẫn viết như mục tiêu kỹ thuật sẽ đạt. Giữ giọng RESEARCH. |

Không phản đối việc chạy thí nghiệm fusion ở xác suất. Phản đối việc coi spec hiện tại là đủ để code, và phản đối việc gọi D là main chất lượng trước khi A/B common-control và identity kernel được chốt.
