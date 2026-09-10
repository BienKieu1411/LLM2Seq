# Đối chiếu new, update_v2 và v4

Các invariant và quyết định sau audit được khóa trong
[ARCHITECTURE_TARGET_AUDIT.md](ARCHITECTURE_TARGET_AUDIT.md); file này giữ vai
trò snapshot đối chiếu code lịch sử.

## 1. Phạm vi chứng cứ

Snapshot code: commit `88486a25b4a68fdbbd9400dc65d15662f27af992`, đối chiếu ngày 2026-09-10. `Paper/README.md` và `Paper/main.pdf` đang có thay đổi ngoài phạm vi. Folder v5 chỉ chứa thiết kế.

Có ba tầng chứng cứ khác nhau: **code** xác định computation graph; **điểm tổng đã báo** xác định chênh lệch quan sát; **giả thuyết** giải thích vì sao một cơ chế có thể làm chất lượng thay đổi. Không có prediction, checkpoint và resolved config đầy đủ của các run để suy ra nguyên nhân nhân quả hoặc ý nghĩa thống kê.

`src/eviseq_update` hiện có cả code contrastive được thêm về sau. Cấu hình base đang tắt `evidence_contrastive`. Phân tích v2 ở đây là **independent source read + bounded residual + gold CE**, không gán các additions về sau cho run 49.488/21.953/45.776. Commit `381ebe7` là mốc lịch sử của thay đổi independent read/bound, không phải hash checkpoint trên server.

## 2. Phần chung đã đối chiếu

[AFMR của new](../eviseq_new/eviseq_afmr/modeling/afmr.py), [AFMR của v2](../eviseq_update/eviseq_update/modeling/afmr.py) và [AFMR của v4](../eviseq_update_v3/eviseq_update_v3/modeling/afmr.py) có cùng SHA-256:

```text
1b2ec11c756a6ab137b8061a5813296943410c14b4221142cee41083b3038e59
```

Ba file `modeling/encoder.py` cũng giống byte, SHA-256:

```text
0223b2924689cc867dd4c46989502bcbad532d441a10f564e68f6b8b8f3c2166
```

Điều này xác nhận implementation chung; config, tokenizer, pretrained checkpoint và trọng số đã train vẫn có thể khác.

Encoder chạy backend pretrained native. AFMR đọc final state và bốn depth taps; controller nhận pooled source, prompt embeddings và output budget. Nó tạo key memory `M` bằng depth residual rank 128 và feature residual rank 256, có gate và output projection khởi tạo zero. Value memory `H0` lấy final encoder state qua base projection. Khi hai hidden widths bằng nhau, projection này là identity.

**Value anchor không có nghĩa encoder đóng băng.** Encoder, projection khi có, memory norms và value projections ở decoder vẫn có thể học. Tách `M`/`H0` chỉ giới hạn đường mà AFMR depth/feature residual sửa values trực tiếp.

Source prior dùng cửa sổ 32/128/512, overlap 0.5, tạo bias trên toàn bộ token content hợp lệ. Prior được tính theo document/prompt một lần; query-key attention vẫn thay đổi theo prefix. Không đúng khi nói các bản này chỉ có attention tĩnh.

[Decoder new](../eviseq_new/eviseq_afmr/modeling/decoder.py), lớp `DecoderLayerWithCross`, giữ thứ tự:

```text
pretrained causal self-attention
→ residual + gated cross-attention(K từ M, V từ H0)
→ pretrained FFN và residual
→ ... → final decoder norm → h
```

Mọi decoder layer có cross-attention; q/k/v/o và norms được khởi tạo từ self-attention tương ứng. Gate cross scalar ban đầu 0.10. Đường đọc toàn nguồn vẫn bị giới hạn bởi source truncation 4096 trong recipe PubMed; “full memory” không có nghĩa nhìn thấy toàn bộ bài dài hơn giới hạn này.

## 3. new: đường sinh trực tiếp và exact copy

[Grounded copy new](../eviseq_new/eviseq_afmr/modeling/grounded_copy.py) dùng hai loại thông tin để tạo copy key: context encoder được pool theo character overlap và embedding token decoder. Query phụ thuộc `h`; attention cộng source prior. Các vị trí có cùng token ID được cộng xác suất, không lấy một occurrence duy nhất.

```text
P_new(v) = (1-g) softmax(W_lm h)[v] + g Σ_{j:id_j=v} A_copy[j]
```

Copy chỉ trỏ tới token decoder nằm trong nguồn hợp lệ đã align. Đây là subword copy trong vocabulary hiện có, không phải mở vocabulary OOV như pointer-generator word-level cổ điển. Copy giúp đặt thêm xác suất lên từ/số/thuật ngữ nguồn; tính đúng quan hệ giữa các token vẫn do model học.

**Điểm nên phát huy:** vocabulary head nhận trực tiếp hidden mà decoder đã xử lý qua nhiều FFN và final norm; không có output semantic perturbation. Cross-attention và copy vẫn cho nguồn tác động lên sinh từ. Chi phí và số đường điều khiển ở output head ít hơn các bản sau.

**Giới hạn:** copy chỉ cấp thêm xác suất cho token có trong source; không cung cấp một đường semantic read riêng tới vocabulary head. Điều này không có nghĩa LM không hiểu nguồn: cross-attention đã cung cấp thông tin đó. Không có cơ chế explicit bảo đảm bao phủ ý, thứ tự câu hay phrase-copy nguyên khối.

R1/RL của new cao nhất trong các bộ điểm EviSeq được báo. Chưa có ablation chứng minh chính việc bỏ semantic residual gây ra ưu thế này.

## 4. update_v2: semantic read độc lập

[Grounded copy v2](../eviseq_update/eviseq_update/modeling/grounded_copy.py), các hàm `prepare`, `read`, `loss_sum`, thực hiện:

```text
q_s = W_q RMS(h)
K_s = RMS(W_k RMS(H0)); V_s = W_v RMS(H0)
A_s = softmax(mask(q_s K_sᵀ / √128 + source_prior))
u_s = RMS(A_s V_s)
d_raw = sigmoid(W_gate [q_s;u_s] + b_gate) * W_out u_s
delta = smooth_relative_RMS_cap(d_raw, h, rho=0.10)
P_v2 = (1-g) softmax(W_lm(h+delta)) + g P_copy
```

Q/K semantic riêng, đọc native encoder positions. Semantic values không bị pool theo biên token decoder, và semantic retrieval không phải dùng lexical copy keys. Đây là điểm khác quan trọng so với shared read v1. Exact copy vẫn được tính từ `h`, trước semantic residual.

**Điểm nên phát huy:** source evidence có đường trực tiếp để sửa xác suất cả những từ không xuất hiện nguyên dạng trong source; semantic matching không bị ép trùng copy matching. Rank 128, một head, không phân cấp, không cắt vùng. Zero `W_out` giữ forward ban đầu trùng copy-only ở cùng base weights; gate dương cho output projection bắt đầu học.

**Giới hạn thực:** cap 0.10 chặn RMS của vector bổ sung, không chặn thay đổi log-odds. Với hai từ a,b:

```text
Δ(logit_a-logit_b) = (W_lm[a]-W_lm[b]) · delta
```

Một margin nhỏ có thể đổi dấu dù residual nhỏ. V2 không giữ một phân phối vocabulary gốc riêng khi semantic read đã học. RMS context làm mất nhiều thông tin về độ lớn context; gate có thể tắt read nhưng không có bằng chứng nó đã học lúc nào nên tắt. Không được mô tả v2 là “bắt buộc luôn dùng semantic” vì gate của nó vẫn có thể tiến gần zero.

V2 so với new: R1 −0.138, R2 +0.052, RL −0.119. Đây là tín hiệu đáng kiểm tra về sự đánh đổi, **không chứng minh bigram đúng hơn về ngữ nghĩa**. Output length, copy tỷ lệ, seed, checkpoint, recipe hoặc preprocessing cũng có thể ảnh hưởng bộ điểm.

## 5. v4: những ràng buộc còn lại sau review

Đọc [config base hiện tại](../eviseq_update_v3/configs/afmr_base.yaml), [grounded copy](../eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py) và [semantic planner](../eviseq_update_v3/eviseq_update_v3/modeling/semantic_plan.py):

| Thành phần | Giá trị/cơ chế hiện tại | Ý nghĩa |
|---|---|---|
| Semantic width | rank 512, 4 heads ×128 | Không còn matching 4×32 |
| `attention=hierarchical_coverage` | 64 content tokens/vùng; region softmax × within-region softmax | Region mass quyết định lượng chú ý tối đa từng token có thể nhận |
| Partition | `partition_heads=false` | Mọi head được đọc mọi vùng; không còn hard ownership mặc định |
| Planner | cumulative learned usage + recent region preference | Theo dõi mức đọc ước lượng; không theo dõi fact đã diễn đạt được xác nhận |
| Head gate | `post_norm`, gain `2*sigmoid` | Gate không còn bị global context norm triệt tiêu theo cách cũ; các head vẫn có thể trùng nhau |
| Fusion | tangent projection + RMS cap + restore norm | Chặn chuyển động theo hình học hidden; vẫn có thể đổi thứ tự từ |
| `query_cross_gate` | `false` | Không được quy lỗi run hiện tại cho module đang tắt nếu không có resolved override |

### 5.1. Region mass là ràng buộc thật

Với token j thuộc region r: `A[j]=pi[r]*A[j|r]`, nên `A[j]≤pi[r]`. Một con số quan trọng có token score cao vẫn bị region probability thấp giới hạn. Region key là pooled keys; nó có thể làm loãng chi tiết hiếm. Region chia theo số token cũng có thể cắt ngang fact/câu. Đây là rủi ro của graph, chưa phải bằng chứng nó xảy ra trong run giảm ROUGE.

### 5.2. Đọc lại không đồng nghĩa lặp ý

Planner cộng learned attention usage của observed summary prefix và phạt `log1p(coverage/8)`. Một đoạn Results có thể chứa nhiều kết quả; penalty có thể làm model chuyển vùng trước khi diễn đạt đủ chúng. Continuity có thể chống lại penalty nhưng cũng tăng xu hướng ở lại vùng đã chọn. CE phải cùng lúc học retrieval, usage và cường độ hai bias này, không có fact-completion labels.

Prefix scan causal, không đọc future target. Khác biệt teacher-forced prefix và generated prefix là exposure bias thông thường, không tự chứng minh leakage hay bug gradient. Bỏ planner trong v5 nhằm loại giả định không có chứng cứ thực nghiệm, không phải sửa một lỗi causal đã được tìm thấy.

### 5.3. Giữ norm không giữ câu

Lấy `h=(1,0)`, hai hàng LM weight `(1,0)` và `(0.999,1)`. Ban đầu logits là `(1,0.999)`. Sau một dịch chuyển tangent 0.01 rồi chuẩn hóa về norm 1, logits xấp xỉ `(0.999950,1.008950)`: từ thắng đổi dù norm giữ nguyên. Đồng thời tangent projection bỏ hướng radial mà model có thể cần để điều chỉnh độ sắc của phân phối.

### 5.4. Capacity và đường học

Semantic/planner params của v4 lớn hơn v2 và K/V semantic rộng gấp bốn. Không có bằng chứng head specialization, đủ coverage hoặc tăng R2 chỉ từ width này. Copy vẫn dùng original hidden, nhưng shared encoder/decoder/prior cùng nhận gradient; graph isolation không bảo toàn copy sau optimizer updates.

Review cũ ghi nhận 318 local tests và smoke pass. Chúng là bằng chứng kỹ thuật được báo trong [review revision](../../Technical_Report/EVISEQ_UPDATE_V3_REVIEW_REVISION.md), không phải một lần chạy test mới hay chứng minh chất lượng PubMed. Script hiện default 84/GPU ×2 ×accum1; dòng default 48/GPU trong report cũ đã lạc hậu.

## 6. Các biến ngoài kiến trúc cần khóa khi đối chứng

New được báo dùng epoch 4, 1 GPU, 48×accum2, clip 1.0. Update v1 dùng epoch 5, 2 GPU, 84×accum1; lịch sử code có giai đoạn bỏ clip. Với cùng N samples, số optimizer updates tương đối là `(5/168)/(4/96)=5/7`. Không được áp con số này cho v2/v4 nếu chưa biết recipe thực của chúng.

T5Gemma PubMed config hiện cũng clip 1.0; batch 4×accum8, LR 1e-5, cosine, 4 epochs. V2 config hiện đã có cosine/warmup; v4 CE engine dùng lịch riêng của nó. Khác optimizer schedule và effective batch có thể làm các run không tương đương. V5 phải có một common CE control harness, không dùng điểm lịch sử làm ablation kiến trúc.

## 7. Kết luận thiết kế từ audit

Giữ khả năng đọc toàn nguồn của trunk, copy alignment và đường từ vựng trực tiếp của new; giữ **flat independent semantic read** của v2. Thay cách semantic branch tác động lên output bằng mixture có anchor xác suất. Không cộng thêm region planner hoặc head diversity constraint vào v5 chính.

Điều được khắc phục về cấu trúc: v5 có phân phối base riêng, bỏ region mass bottleneck và learned usage penalty, giảm width semantic về 128. Điều chưa được khắc phục bằng chứng: R1/R2/RL thực tế, thiếu ý, câu gượng, redundancy và ảnh hưởng shared-weight adaptation. Các điểm này có điều kiện bác bỏ cụ thể trong [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

## Nguồn local

Ngoài các file code đã liên kết: [regression v1→v2](../../Technical_Report/EVISEQ_UPDATE_REGRESSION_2026_09_08.md), [thiết kế v3](../../Technical_Report/EVISEQ_UPDATE_V3_ARCHITECTURE.md), [coverage revision](../../Technical_Report/EVISEQ_UPDATE_V3_COVERAGE_REVISION.md), [Luna review](../../Technical_Report/EVISEQ_UPDATE_V3_LUNA_REVIEW.md). Điểm benchmark là thông tin kết quả đã báo trong dự án; thiếu raw outputs được giữ nguyên là giới hạn chứng cứ.
