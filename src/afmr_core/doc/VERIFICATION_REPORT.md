# Xác minh bản thiết kế AFMR

## Kết luận kiểm tra

**Kết luận:** AFMR đã tích hợp các đường mạnh cốt lõi của `eviseq_new` và
`update_v2`, đồng thời đặt cơ chế trực tiếp để xử lý hai đánh đổi chính của
v2: semantic residual có thể làm mất base vocabulary signal và semantic mass
bị nested gate giới hạn khi copy gate lớn. Package đã có implementation,
explicit v2 control và CPU/tiny acceptance tests; chưa có CUDA/DDP run,
checkpoint PubMed hoặc
prediction nên chưa thể xác nhận tác động lên ROUGE.

`PLAN_ARCHITECTURE_REVIEW.md` được giữ nguyên như review lịch sử. Các công thức/acceptance đã sửa trong [DESIGN.md](DESIGN.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [EVALUATION_PLAN.md](EVALUATION_PLAN.md) và report này.

## Ma trận giữ điểm mạnh

| Nguồn | Điểm mạnh đã xác nhận | Cách AFMR giữ | Trạng thái |
|---|---|---|---|
| `new` | Encoder native + AFMR pretrained path | Không đổi backbone; giữ source policy và cross-attention mọi decoder layer | Đã khóa trong spec |
| `new` | `H0` value anchor và `M` retrieval memory | `H0=bridge.value_memory`, `M=bridge.memory`; không trộn nhầm hai state | Đã khóa; cần C1 |
| `new` | Character-overlap copy alignment | Giữ candidate IDs, masks, source prior và duplicate-ID marginalization | Đã khóa; cần C3/C4 |
| `new` | Vocabulary read trực tiếp từ `h` | `P0=softmax(W_lm h)` luôn còn trong mixture | Đã khóa; cần C1/C4 |
| `new` | Copy gate `g` dễ bảo vệ thuật ngữ/số | `pi_copy=g` ở main; semantic không được đưa vào copy keys/values | Đã khóa; shared-trunk drift vẫn phải đo |
| `update_v2` | Semantic source read độc lập với lexical copy | Flat reader rank 128, native source positions, copy/semantic API tách riêng | Đã khóa; cần C2 |
| `update_v2` | Thứ tự RMS và source values đã có graph thực tế | `H0n=RMS(H0)`, `K=RMS(Wk(H0n))`, `V=Wv(H0n)`, context RMS | Đã khóa; reader test pass |
| `update_v2` | Residual được bound theo hidden RMS | Main giữ smooth relative RMS cap `rho=.10`; control legacy có cap v2 | Numerical test pass |
| `update_v2` | Semantic có thể tăng xác suất token không xuất hiện nguyên dạng | `Ps=softmax(W_lm(h+delta))` là branch vocabulary thứ hai | Đã khóa; tác động cần train |
| `new` + `v2` | Log-domain copy likelihood ổn định | Ba branch dùng `log_softmax/logsumexp`, target không copy được có `lc=-inf` | Oracle test pass |

Các điểm trên phù hợp với tiền lệ pointer-generator về việc tách copy khỏi vocabulary generation ([See et al.](https://aclanthology.org/P17-1099/)) và probability mixture ([Mixture of Softmaxes](https://arxiv.org/html/1711.03953v3)); vì vậy không được gọi từng thành phần riêng lẻ là novelty.

## Ma trận khắc phục điểm yếu

| Điểm yếu | Cơ chế AFMR | Mức khắc phục |
|---|---|---|
| `new` không có semantic vocabulary route riêng | Thêm `Ps` từ `h+delta` nhưng giữ `P0` | Khắc phục về cấu trúc; ROUGE chưa chứng minh |
| v2 để semantic hidden thay thế đường base | Ba-way probability mixture `P0/Ps/Pcopy` | Khắc phục trực tiếp |
| v2 semantic mass nằm trong `(1-g)` | `alpha` là mass riêng, chỉ cap khi `g` gần chiếm hết simplex | Khắc phục một phần có chủ ý; cap-hit phải báo |
| semantic residual nhỏ vẫn có thể đảo thứ tự logits | Giữ base anchor và giới hạn RMS residual | Giảm rủi ro, không bảo đảm thứ tự logits |
| `Wo=0` làm Q/K/V học chậm | `Wo=tiny_rms_1e-3` là candidate chính; zero chỉ parity | Khắc phục đường học; cần C5/C18 |
| v2 inner gate `.05` tạo gate kép/đường gradient khó đọc | `inner_gate=false` ở main; bật lại `.05` chỉ trong control tái lập v2 | Khắc phục main, bảo toàn C2 |
| Đổi `K=M` có thể vô tình đổi copy path | API có `copy_memory`, `semantic_key_memory`, `semantic_value_memory` riêng | Khắc phục contract |
| Independent simplex dùng `logit(g)` sai prior | Dùng `log(1-g)` và `log(g)` với residual router zero-init | Khắc phục công thức; đã có numeric smoke |
| Gradient copy bị diễn giải sai | Kiểm tra `dP/dg=Pcopy-P0`; log `dL/dg` riêng | Khắc phục acceptance |
| DDP 1/2 GPU có thể khác token weighting | Canonical global-batch manifest và `loss_r=world_size*L_r/N_global` | Khắc phục contract; cần C9/C19 |
| Scheduler stage-local làm thay đổi budget | Một global cosine schedule trên optimizer updates | Đã triển khai và có test |
| Đánh giá có thể dùng prediction cũ | Checkpoint/config/split/decoder fingerprint trước cache reuse | Đã triển khai và có test |

Ba ràng buộc `pi_copy=g`, `pi_base>=floor>0` và `pi_sem>0` không thể đồng thời đúng khi `g` gần một. Main ưu tiên giữ copy mass; independent simplex là control để đo lựa chọn ngược lại. Đây là trade-off được công khai, không phải lỗi chưa giải thích.

## Trạng thái implementation và blocker

Các blocker code đã được xử lý trong package AFMR: global cosine scheduler,
canonical manifest, named optimizer groups, strict checkpoint/config
fingerprints, stale-cache metadata check, `H0` value anchor, tách copy và
semantic reader, và dense/chunked mixture-NLL kernel. Bằng chứng cụ thể cùng
những gate chưa thể chạy trên máy CPU được ghi trong
[`IMPLEMENTATION_EVIDENCE.md`](IMPLEMENTATION_EVIDENCE.md).

## Công thức và đường gradient phải giữ nguyên khi code

```text
P0 = softmax(W_lm h)
Ps = softmax(W_lm (h + delta))
Pcopy = scatter_add(exp(log_attention), token_ids)

alpha_raw = alpha_max * sigmoid(r_sem) * evidence
g_route = stop_gradient(g)
alpha = min(alpha_raw, max(0, 1-g_route-generate_reserve))

pi_copy = g
pi_sem  = alpha
pi_base = 1-g-alpha
P       = pi_base*P0 + pi_sem*Ps + pi_copy*Pcopy
```

Tính loss trực tiếp trên `P` ở log-domain; không thay bằng tổng ba CE có trọng số. Với `Wo=0`, `Ps=P0` và forward phải đối chiếu kernel `_mix_logits` của `new`; raw logits v2 chỉ so khi dùng gauge `Zs`, còn main dùng `Z0`.

`semantic_evidence` main là nhị phân từ native `content_mask.any(dim=-1)`, shape `[B,T,1]`; không dùng copy mask `[B,W]`. Attention scale là `sqrt(r)`, source prior có `semantic_prior_scale` riêng. Cache source-side gồm `H0/M`, semantic K/V, copy state và cross K/V; `h/q/u/alpha/g/logits` tính lại theo prefix.

Ở endpoint `Wo=0`, `dL/dalpha=0` và Q/K/V không có gradient qua `u` ở backward đầu, nhưng base/copy/trunk vẫn có gradient legacy và `Wo` phải hữu hạn. Candidate `tiny` phải cho Q/K/V/router gradient khác zero ở các step đầu trên batch có signal. Đây là chain rule, không phải giả định tốc độ học.

## Target và phép so sánh

| Gate | Điều kiện |
|---|---|
| Architecture gate | Một checkpoint AFMR dưới common EviSeq recipe có R1 `>49.626`, R2 `>21.953`, RL `>45.895` |
| System stretch | Sau khi rerun T5Gemma với selection/evaluator minh bạch, một checkpoint duy nhất có R1 `>49.626`, R2 `>21.990`, RL `>45.895` |

Mốc lịch sử là `new=49.626/21.901/45.895`, `update_v2=49.488/21.953/45.776`, T5Gemma `49.580/21.990/45.463`. Các mốc này chưa có đủ resolved config/predictions để làm bằng chứng causal. Phải rerun A/B trước, dùng cùng processed manifests, tokenizer/alignment policy, visible source budget, optimizer update budget, validation-CE selection và Perl ROUGE-1.5.5.

Không được ghép R1/R2/RL từ các checkpoint khác nhau. Báo ít nhất ba seed, paired bootstrap trên cùng document IDs và spread giữa seed. ROUGE tăng không đủ để claim giảm hallucination; claim factuality cần metric/attribution riêng. Phân tích training dynamics cũng cho thấy copy propensity và hallucination có thể thay đổi ở các giai đoạn khác nhau, nên phải log branch diagnostics theo step ([Training Dynamics for Summarization](https://arxiv.org/abs/2110.08370)).

## Tham chiếu hiện đại và giới hạn chuyển giao

T5Gemma 2 giữ cross-attention ở mọi layer trong adaptation encoder–decoder và có UL2/pretraining riêng; v5 chỉ mượn quyết định source access trong trunk, không coi output mixture là bằng chứng sẽ thắng ([T5Gemma 2](https://arxiv.org/html/2512.14856v1), [UL2](https://arxiv.org/abs/2205.05131)). LongT5/CoLT5 chỉ đáng thử nếu audit chứng minh truncation/context là bottleneck; Mixture-of-Depths, DeepSeek-V3 và Attention Residuals khác scale/pretraining nên được hoãn ([LongT5](https://arxiv.org/abs/2112.07916), [CoLT5](https://arxiv.org/abs/2303.09752), [Mixture-of-Depths](https://arxiv.org/abs/2404.02258), [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals)).

Các nguyên tắc dùng để khóa plan là chọn likelihood theo phân phối, tính toán log-space, train-error-first, một thay đổi mỗi thí nghiệm và baseline/metric rõ ràng trong [Deep Learning](https://www.deeplearningbook.org/contents/optimization.html) và [Practical Methodology](https://www.deeplearningbook.org/contents/ml.html).

## Điều kiện kết luận

- **Đã xác nhận:** cấu trúc v5 giữ được các đường mạnh của `new/v2` và có cơ chế hợp lý xử lý semantic overwrite/nested gate.
- **Chưa xác nhận:** v5 chạy đúng gradient thực tế, parity số học, tốc độ, VRAM, ROUGE và hallucination.
- **Điều kiện dừng:** C1/C2/C3 hoặc fixed-small-set C18 fail thì không train PubMed; E không vượt common A/B sau ba seed thì không thêm planner/contrastive để che regression.
