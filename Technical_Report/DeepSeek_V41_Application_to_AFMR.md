# Ứng dụng DeepSeek-V4.1 vào AFMR để tăng ROUGE

## Phạm vi và kết luận quyết định

Tài liệu này đọc lại [bản trích xuất DeepSeek-V4.1-Flash](DeepSeek_V41_Tech_Report_Extraction.md), đối chiếu với package AFMR hiện tại trong `src/afmr_core`, và trả lời một câu hỏi hẹp: **ý tưởng nào từ report có khả năng giúp ROUGE-1/ROUGE-2/ROUGE-L trên PubMed, ý tưởng nào chỉ giúp serving, và nên thử theo thứ tự nào**.

Kết luận chính:

1. **Không nên đưa nguyên Causal Encoder-Decoder (CED), CSA2, sparse Top-K hay bounded replay vào vòng tối ưu ROUGE hiện tại.** Bằng chứng của DeepSeek cho các cơ chế đó chủ yếu là prefill, KV-cache và long-context serving. Report không có ablation CED-only và không chứng minh tăng ROUGE PubMed.
2. **Điểm đáng chuyển giao nhất là nguyên tắc anchor và training-aware state reuse**, nhưng AFMR đã có phần lớn nguyên tắc này: `H0` làm value anchor, `M` làm retrieval memory, cross-attention ở mọi decoder layer, semantic reader tách khỏi copy, và copy-mass-preserving output mixture. Chuyển CED trực tiếp sẽ trùng chức năng hoặc phá parity với checkpoint Qwen.
3. **Ưu tiên có xác suất tăng điểm cao hơn là training và ổn định hóa đường hiện có:** sửa blocker copy-alignment/protocol, chạy interface warmup rồi full fine-tune với cùng ngân sách update, và đánh giá EMA/SWA hoặc greedy checkpoint soup trên validation. Các bước này giữ đường `P0`/copy nên có cơ hội phục hồi ROUGE-1 và ROUGE-L mà v2 đã đánh đổi để tăng ROUGE-2.
4. **Ứng viên kiến trúc có thể nghiên cứu sau khi baseline ổn định** là một layer-adaptive anchor adapter rất nhỏ cho source K/V của cross-attention, khởi tạo identity và không đụng copy path. Đây là adaptation có kiểm soát lấy cảm hứng từ anchor của CED, không phải port nguyên CED.
5. **Engram-lite chỉ nên là pilot cuối**, vì có thể giúp thuật ngữ/n-gram y sinh và ROUGE-2 nhưng cũng có nguy cơ học thuộc, hash collision, tăng hallucination và tạo confound tokenizer. Không đưa vào main trước khi các control rẻ hơn thất bại.

Các mốc ROUGE dùng trong kế hoạch là số liệu lịch sử do dự án cung cấp, chưa được tái lập trong report này:

| Mốc | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| `eviseq_new` | 49.626 | 21.901 | 45.895 |
| `eviseq_update_v2` | 49.488 | 21.953 | 45.776 |
| Acceptance stretch cho AFMR | `>49.626` | `>21.953` | `>45.895` |

Không có kết luận nào dưới đây bảo đảm AFMR đạt các mốc này. Mục tiêu của report là chọn thí nghiệm có attribution rõ và loại sớm các hướng dễ làm tụt R1/RL.

## 1. Snapshot kiến trúc AFMR hiện tại

### Đường dữ liệu và gradient

AFMR hiện encode source một lần, tạo hai memory:

- `H0 = bridge.value_memory`: value anchor dùng cho grounded copy và semantic reader.
- `M = bridge.memory`: retrieval memory dùng làm key của cross-attention và có thể là semantic key ở variant được đặt tên riêng.
- `b = bridge.source_bias`: source prior đã mask.

`AFMRModel.encode_source()` truyền riêng `H0` cho copy/semantic value và `M` cho decoder memory (`src/afmr_core/afmr_core/modeling/model.py:44-75`). Đây là cấu trúc phù hợp với bài học “descriptor/retrieval” tách khỏi “content/value” trong report DeepSeek.

Decoder hiện bắt buộc cross-attention ở mọi layer (`decoder.py:311-333`). Mỗi layer sở hữu bản sao `q/k/v/o` của self-attention (`decoder.py:70-106`), do đó projection source đã là layer-specific. Main path hiện dùng `K` từ `M`, `V` từ `H0` (`decoder.py:96-106`, `model.py:110-120`). Copy head chỉ đọc `H0`; semantic reader main đọc `K=H0,V=H0`; dual readout trộn `P0`, `Ps` và `Pcopy` ở probability space.

Loss training là teacher-forced CE trên phân phối cuối, không sinh candidate và không dùng temperature/top-k/top-p (`decoder.py:342-484`, `configs/afmr_base.yaml:124-135`). DDP đã có token normalization theo logical window, explicit parameter groups, gradient clipping và BF16 autocast với parameter/optimizer FP32 (`training/engine.py:131-162`, `training/engine.py:263-328`). Vì vậy không được đưa DSpark, OPD, candidate mining hoặc self-improve vào thí nghiệm này.

### Các điểm mạnh cần giữ nguyên

| Thành phần | Vì sao có giá trị cho score | Điều kiện bảo toàn |
|---|---|---|
| `P0 = softmax(W_lm h)` | Giữ prior của decoder Qwen và giúp R1/RL không bị semantic branch thay đổi toàn bộ hidden | Không làm mất base route; fallback source rỗng trả `z0` |
| Grounded copy trên `H0` | Giữ thuật ngữ, số và cụm exact-match; có thể hỗ trợ R1/R2 | Không cho semantic residual sửa trực tiếp copy state |
| Semantic reader `K=H0,V=H0`, rank 256 | Bổ sung source-conditioned abstraction, có thể giúp n-gram dài | Residual cap và output init nhỏ; log `alpha`/delta |
| Capped simplex với `pi_copy=g` | Giữ trách nhiệm copy ở mixture interface, giảm nguy cơ mất R2 | Phân biệt rõ “mass preserving” với parameter isolation |
| Cross-attention mọi layer | Source access sâu, phù hợp ablation T5Gemma 2 | Không giảm còn global layers trong main |
| Greedy headline, sampling API riêng | ROUGE công bằng và tái lập | Temperature/top-k/top-p chỉ sau final mixture |

### Kiểm tra implementation hiện tại

Acceptance suite offline của package AFMR hiện chạy **33 passed, 2 warnings** trong môi trường `bienkieu_env`; test dùng tiny/local path và không tải pretrained model. Copy alignment đã tách candidate width `W` khỏi alignment-edge width `E` tại `data/copy_alignment.py:64-94`, và overlap nhiều span có test tại `tests/test_protocol.py:222-253`. Đây là điều kiện cần trước mọi so sánh score vì nếu làm mất một overlap edge thì R1/R2 có thể giảm do lỗi dữ liệu chứ không phải do semantic route.

Config main hiện đặt `key_dim=256` và `semantic_read.rank=256` (`configs/afmr_base.yaml:45-71`), trong khi một số spec cũ vẫn ghi rank 128. DeepSeek report cảnh báo không bê nguyên kích thước từ model lớn sang model nhỏ; vì vậy rank 128 và 256 phải được xem là **hai ablation**, không gọi 256 là mặc định tốt hơn khi chưa có validation evidence. Việc tăng rank làm tăng projection/activation cost và có thể giảm R1/RL vì overfit hoặc semantic route cạnh tranh với base.

Runner headline cũng phải dùng `ALLOW_CROSS_SPLIT_CONTENT=false`; bản bật duplicate chỉ được gắn nhãn diagnostic. Các con số test ở trên là kiểm tra implementation, không phải bằng chứng AFMR đã vượt các mốc ROUGE.

Có hai caveat cần giữ trong mọi ablation. Thứ nhất, `hidden_interpolation_state()` phải lấy semantic evidence từ semantic source mask; nếu lấy từ `copy_state.mask` thì một mẫu có semantic source nhưng không có copy candidate sẽ bị tắt semantic branch, khiến control không công bằng. Thứ hai, copy path chưa phải parameter-isolated: `source_bias` do `M` sinh ra đang được dùng đồng thời ở cross-attention, semantic reader và copy head. Semantic loss vẫn có thể cập nhật bridge rồi làm copy attention/gate đổi gián tiếp qua bias. `g_route` detach chỉ chặn đường gradient trực tiếp của router, không chặn shared-trunk drift. Hai điểm này cần được đo hoặc tách control trước khi diễn giải gain/loss của semantic branch.

## 2. DeepSeek report thực sự chứng minh điều gì

### CED/YOCO

DeepSeek chia stack causal thành causal encoder và decoder. Hidden giữa stack tạo global KV của các lớp trên bằng projection theo layer; local SWA vẫn lấy từ hidden hiện tại. Báo cáo cho prefill xấp xỉ `O(NL/2 + n_win L/2)`, nhưng thừa nhận bounded replay là xấp xỉ và không đưa ra CED-only quality ablation.^1 YOCO, nguồn cảm hứng được trích trong report, cũng mô tả lợi ích chính là giảm cache/prefill trong khi đạt chất lượng ngôn ngữ cạnh tranh, không phải tăng ROUGE summarization.^2

Điều này **không** chứng minh CED sẽ làm summary tốt hơn trên PubMed. Đặc biệt, CED được train từ đầu trên mô hình 40 layer, hidden 5120 và dữ liệu quy mô rất lớn; chèn nó sau một Qwen checkpoint sẽ thay đổi đường hidden và các giả định positional/cache.

### CSA2 và hierarchical sparse indexer

CSA2 chia Full/Reindex/Reuse và hierarchical indexer giới hạn candidate pool. Report yêu cầu cùng restriction trong training và inference để tránh mismatch. Đây là bài học tốt về protocol, nhưng selection có thể bỏ mất evidence; với PubMed 4096 token và copy head cần giữ số/symbol, hard Top-K có rủi ro trực tiếp làm tụt R1/RL. Không đưa sparse selection vào main để “mượn” điểm của DeepSeek.

### Engram

Engram tách static n-gram pattern storage khỏi dynamic Transformer computation, hash lookup `O(1)` và contextual gating. Paper gốc báo validation loss cải thiện khi tăng memory slots trong thiết lập iso-parameter/iso-FLOPs của các model lớn.^3 Đây là bằng chứng cho một hướng memory, không phải bằng chứng rằng bảng n-gram nhỏ sẽ tăng ROUGE PubMed. Tokenizer compression, leakage control và source/target asymmetry phải được thiết kế lại cho summarization.

### FP4, bounded replay, DSpark, RL và infrastructure

FP4 main KV, SWA replay, DSpark, asynchronous RL, effort control và DSec giải quyết memory/throughput/agent workload. Chúng không tạo thêm evidence semantic trong CE-only PubMed. FP4/quantized cache còn có rủi ro làm thay đổi logit và ROUGE nếu áp dụng trước khi có quality baseline.

### Optimizer và dữ liệu

DeepSeek dùng optimizer khác nhau cho matrix/non-matrix và đầu tư lớn vào filtering, deterministic packing, shared-state lifetime và post-training data. Bài học chuyển giao được là **tách biến thử nghiệm, giữ token accounting và lifetime của state đúng**, chứ không phải thay AdamW bằng Muon/Sinkhorn trong cùng một run architecture. AFMR đã có explicit optimizer groups và canonical sampler; cần kiểm tra blocker trước khi kết luận kiến trúc.

## 3. Ma trận chuyển giao

| Cơ chế từ report | Liên hệ với chất lượng PubMed | Rủi ro R1/RL | Quyết định |
|---|---|---|---|
| CED topology đầy đủ | Chủ yếu giảm prefill/cache; không có CED-only ROUGE ablation | Phá checkpoint/parity, bottleneck anchor | **Không port** |
| Projection theo layer từ anchor | Có thể cho mỗi decoder layer một source view chuyên biệt | Medium nếu gate làm thay source alignment | **Pilot riêng** |
| `K=M,V=H0` tách retrieval/content | AFMR đã dùng trong main | Double-count source prior nếu thêm route khác | **Giữ, đo control prior=0** |
| CSA2 Reindex/Reuse | Tiết kiệm attention ở context dài | Bỏ evidence/số; train-inference mismatch | **Không dùng trong score main** |
| Hierarchical Top-K | Có thể tập trung salience | Copy coverage giảm, boundary errors | **Không dùng** |
| Engram n-gram memory | Có thể giữ cụm thuật ngữ và token hiếm | Leakage, collisions, hallucination, tokenizer confound | **Pilot cuối** |
| FP4 KV/replay | Giảm memory | Quantization/reconstruction đổi logits | **Không dùng để tăng ROUGE** |
| Muon/Sinkhorn | Có thể thay optimization geometry | Confound mạnh, chưa có bằng chứng fine-tune 1.37B | **Ablation sau** |
| Data filtering/packing/state lifetime | Giảm noise và variance, bảo đảm parity | Nếu lọc sai có thể bỏ example hữu ích | **Ưu tiên audit** |
| Domain-adaptive pretraining (DAPT) | Tăng prior từ vựng/kiến thức PubMed trước SFT | Thêm compute; không tách khỏi data budget nếu so sánh không công bằng | **Data track riêng** |
| Model merging/SWA | Làm nghiệm cuối phẳng hơn, không đổi inference graph | Checkpoint khác basin, validation overfit | **Ưu tiên rẻ** |
| DSpark/RL/OPD | Không liên quan CE-only summary | Vi phạm yêu cầu không sinh candidate | **Bỏ** |

## 4. Các ứng viên nên thử theo thứ tự

### A. Chặn lỗi protocol trước khi tối ưu score

Đây là điều kiện cần, không phải một claim kiến trúc mới.

1. Sửa và test biểu diễn copy alignment với hai chiều độc lập: candidate width `W` cho `copy_token_ids/mask`, edge width `E` cho các overlap tensors. `align_copy_tokens()` hiện có thể tạo một ID cho mỗi decoder token nhưng nhiều alignment edge cho cùng token; `GroundedCopyHead.prepare()` lại yêu cầu các tensor cùng width. Trường hợp một token phủ ba span phải chạy được và giữ đủ trọng số.
2. Chạy preparation headline với `ALLOW_CROSS_SPLIT_CONTENT=false`, lưu `preparation_report.json`, báo duplicate count. Bản bật duplicate chỉ là diagnostic.
3. Đo boundary exclusion (`end >= visible_end`), copy coverage, tỷ lệ target token copyable và source-visible unigrams/bigrams. Không thay policy giữa `new`, v2 và AFMR.
4. Xác nhận ROUGE-1.5.5 cùng tokenizer/detokenization, split, checkpoint rule và greedy decoder. Không trộn kết quả Python ROUGE-1.0.0 vào headline.

**Lý do liên quan DeepSeek:** report coi candidate restriction và replay là “training-aware approximation”; nếu representation/truncation khác giữa train và eval thì attribution của mọi gain bị hỏng. Đây là chuyển giao chắc chắn hơn việc lấy kích thước CED.

Nếu audit cho thấy dữ liệu PubMed có nhiều malformed row, trùng gần nhau, article bị cắt hoặc tokenizer chưa có prior domain, mở một **data track** riêng: strict dedup/normalization, source-visibility statistics và DAPT trên raw PubMed text trước SFT. DAPT có tiền lệ trên pretrained language models và có thể tăng domain prior, nhưng đây là thêm training budget và không được gộp gain vào claim kiến trúc.^6 So sánh công bằng phải giữ tổng token/compute cố định hoặc báo rõ phần budget tăng thêm; không dùng target/reference để lọc train.

### B. Giữ tổng ngân sách 5 epoch nhưng bật interface warmup

`AFMRTrainer` đã hỗ trợ hai stage nhưng `afmr_base.yaml` đang đặt `interface_warmup_epochs: 0`. Chạy control `0+5` và candidate `1+4` với cùng logical batch/update budget:

- epoch warmup: chỉ `bridge`, cross-attention, grounded copy, semantic reader và dual readout train; encoder/backbone pretrained giữ nguyên;
- bốn epoch sau: full fine-tune với scheduler global không reset;
- seed, manifest, batch 84/GPU, accumulation 1 và validation checkpoint rule giữ nguyên.

Giả thuyết: route mới học cách đọc source trước khi được phép làm encoder/decoder drift, nhờ đó giữ `P0` và ROUGE-L tốt hơn. Đây là một adaptation training có thể kiểm chứng, không phải claim rằng DeepSeek warmup đã được chứng minh cho PubMed. Nếu `1+4` giảm validation CE/R1/RL hoặc route gate không học, bỏ.

### C. EMA/SWA hoặc greedy checkpoint soup trên validation

DeepSeek report nhắc model merging trong post-training nhưng không cung cấp công thức score attribution. Model Soups cho thấy averaging các model fine-tuned có thể cải thiện generalization mà không tăng inference cost; SWA adaptation cho PLM báo cải thiện trên cả generation tasks mà không cần teacher hay double forward/backward.^4,^5

Áp dụng an toàn cho AFMR:

1. Giữ một run architecture cố định và lưu checkpoint mỗi epoch/update window.
2. Chọn candidates bằng validation CE hoặc registered validation ROUGE, tuyệt đối không dùng test.
3. Thử uniform average của hai hoặc ba checkpoint liền kề trong cùng basin; sau đó thử greedy soup, chỉ thêm checkpoint nếu validation cải thiện.
4. Đánh giá một model soup duy nhất trên test với cùng output processor. Lưu hash từng checkpoint, trọng số soup, resolved config và prediction fingerprint.

SWA/soup không làm AFMR “mới” hơn, nhưng có expected cost thấp và có thể khôi phục R1/RL nếu v2/AFMR overfit ở epoch cuối. Nếu các checkpoint có semantic router khác hẳn hoặc validation không đơn điệu, không average mù.

### D. Tách source prior theo chức năng — control ưu tiên trước khi thêm module

Đây là cách chuyển giao trực tiếp nhất từ nguyên tắc DeepSeek phân biệt indexer/retrieval state với main KV/content state. Giữ `M` làm retrieval key và `H0` làm value/copy anchor, nhưng tạo bias riêng:

```text
b_cross    = source_bias(M)
b_semantic = source_bias_semantic(H0 hoặc M)
b_copy     = source_bias_copy(H0)
```

Ở control rẻ nhất, chỉ detach `b_copy` khỏi nhánh semantic/bridge hoặc đặt `b_semantic=0` để xem semantic route có đang làm copy drift không. Không thay copy mass, không thêm loss và không giảm source tokens. Log chênh lệch `Pcopy`, copy precision, `g`, source-visible coverage và gradient của bridge giữa bản dùng chung bias và bản tách bias.

Nếu tách bias giữ được R1/RL trong khi semantic CE/R2 không giảm, đây là bằng chứng implementation đang double-count hoặc chia sẻ tín hiệu không mong muốn. Nếu không có thay đổi, không nên thêm một source-prior module phức tạp hơn. Đây là **decomposition/control**, không được claim là CED mới.

### E. Layer-Adaptive Anchor Source Projection (LAASP) — candidate kiến trúc

Đây là adaptation nhỏ nhất có ý nghĩa từ CED:

```text
K_l = W_l^K( norm(M + beta_l * (H0 - M)) )
V_l = W_l^V( norm(H0 + gamma_l * (M - H0)) )
```

Trong đó `beta_l` và `gamma_l` là scalar hoặc vector rank thấp, bị chặn bằng `tanh`, khởi tạo bằng 0 để đúng main `K=M,V=H0`. Copy vẫn đọc `H0` trước semantic residual; LAASP chỉ nằm trong cross-attention. Gradient phải đi từ cross-attention về `beta/gamma`, `M/H0` và projection `W_l`, không detach anchor.

Tại sao có thể giúp score: CED cho mỗi decoder layer projection riêng từ một anchor ổn định; LAASP cho layer tự điều chỉnh mức retrieval-vs-content mà không thay backbone topology. Tại sao có thể làm tụt: `M-H0` có thể là noise; source prior đang tác động ở nhiều route; layer gate có thể làm cross-attention lệch khỏi pretrained copy alignment.

Ma trận tối thiểu:

| Variant | `K` | `V` | Gate | Mục đích |
|---|---|---|---|---|
| A0 | `M` | `H0` | none | AFMR main hiện tại |
| A1 | `M + beta(H0-M)` | `H0` | `beta=0` init | Chỉ kiểm tra retrieval key |
| A2 | `M` | `H0 + gamma(M-H0)` | `gamma=0` init | Chỉ kiểm tra value adaptation |
| A3 | cả hai | cả hai | bounded | Candidate đầy đủ |

Promotion chỉ khi A1/A2/A3 giữ hoặc tăng R1/RL, R2 không thấp hơn update_v2, validation CE giảm, copy precision không giảm và gradient anchor hữu hạn. Nếu chỉ latency tốt hơn, ghi là efficiency result.

### F. Engram-lite cho thuật ngữ — pilot cuối

Nếu diagnostics chứng minh lỗi chính là rare biomedical phrase/number recall chứ không phải source alignment, thử một memory nhỏ:

- hash source-side token n-grams bậc 2–4 vào tối đa 16k bucket/order, embedding 128–256; tổng thêm khoảng vài đến vài chục triệu parameter, giữ dưới 1.5B;
- fuse vào semantic context hoặc một branch vocabulary riêng, gate khởi tạo 0 và cap nhỏ;
- không cho memory thay `H0` copy state; chỉ activation khi n-gram nằm trong source-visible span;
- hash/tokenizer version nằm trong checkpoint fingerprint; bảng được reset nếu tokenizer đổi;
- train/eval cùng canonicalization, không xây memory từ test/reference.

Đây là **Engram-inspired source phrase memory**, không phải implementation nguyên Engram của DeepSeek. Cần ablation `memory off/on`, bucket size, orders và source-only vs target-prefix. Nếu R2 tăng nhưng R1/RL giảm hoặc hallucination tăng, loại.

### G. DAPT/curation — hướng score có thể cao nhưng tách khỏi architecture

DeepSeek report dành phần lớn pretraining/post-training discussion cho filtering, deterministic packing, domain mixture và verified data, trong khi CED-only attribution không có. Với PubMed, một DAPT ngắn trên **raw source text không có summary/reference** có thể giúp Qwen/PPLX có prior thuật ngữ, viết tắt và cấu trúc câu y sinh trước khi học mapping source→summary. Tuy nhiên DAPT làm thay đổi checkpoint khởi tạo; nó phải được báo như `DAPT+SFT`, không so trực tiếp với `SFT` rồi gọi là kiến trúc tốt hơn.

Thiết kế tối thiểu:

1. `D0`: SFT hiện tại, cùng tokenizer/split/seed.
2. `D1`: DAPT trên train-source-only rồi SFT, số token tổng được đăng ký trước.
3. Không dùng validation/test article trong DAPT; lưu hash raw corpus và manifest.
4. Đánh giá trước/sau DAPT bằng validation CE, source-visible entity/number recall, R1/R2/RL và length distribution.

Nếu D1 tăng score nhưng LAASP/mixture không tăng, kết luận là domain adaptation. Nếu D1 chỉ tăng R2 nhưng làm giảm R1/RL, giảm DAPT budget hoặc giữ checkpoint soup của SFT control.

## 5. Những hướng không nên dùng để cứu điểm

- **CED full rewrite:** không có quality evidence cho PubMed và phá warm-start.
- **Hard Top-K/CSA2:** dễ bỏ evidence cần cho exact overlap; không có lỗi latency cần giải quyết ở source length 4096.
- **SWA bounded replay/FP4:** xấp xỉ và quantization thay logits; chỉ dùng khi mục tiêu là memory/serving.
- **DSpark, RL, OPD, self-improve, candidate mining, reranking:** vi phạm CE-only và làm mất attribution.
- **Muon/Sinkhorn trong cùng run với LAASP/Engram:** có thể hữu ích ở pretraining lớn nhưng không tách được nguyên nhân trên fine-tuning 1.37B. Nếu thử, chạy riêng sau architecture gate.
- **Tăng thêm head/rank không chẩn đoán:** semantic rank 256 hiện đã được nâng; nhiều head có thể redundancy và làm semantic branch tranh mass với `P0`.
- **Tối ưu theo test ROUGE hoặc dùng test để chọn soup:** tạo leakage và làm claim không công bằng.

## 6. Plan thí nghiệm có attribution

### 6.1. Giai đoạn smoke và protocol

Chạy trước trên fixed-small-set, FP32 và BF16 nếu có CUDA:

- copy alignment overlap 1-edge/3-edge;
- source mask rỗng và source có punctuation/number;
- dense vs chunked mixture NLL;
- `P0` parity khi `alpha=0`, `Wo=0` chỉ là endpoint;
- finite gradient cho encoder, bridge, cross, copy, semantic và router;
- cache/reorder/compaction không đổi output;
- `W/E` alignment không làm crash batch.

### 6.2. A/B chính

Giữ cùng dataset manifest, target-token accounting, world size, dtype, decoder settings, seed và checkpoint rule:

| ID | Mô hình | Training |
|---|---|---|
| B0 | `eviseq_new` control | recipe lịch sử được rerun |
| B1 | `eviseq_update_v2` control | recipe lịch sử được rerun |
| F0 | AFMR hiện tại | `0+5`, AdamW |
| F1 | AFMR warmup | `1+4`, AdamW |
| F2 | F1 + checkpoint soup/SWA | validation-only postprocess |
| G1 | LAASP key-only | same as F1 |
| G2 | LAASP value-only | same as F1 |
| G3 | LAASP both | same as F1 |
| H1 | Engram-lite | chỉ sau G/F gate |
| D1 | DAPT + SFT | data track riêng, cùng token budget |

Mỗi candidate pilot một seed chỉ để loại failure; candidate được promotion phải chạy ít nhất ba seed định trước. Báo mean/spread theo seed và paired bootstrap theo document; không lấy R1 từ checkpoint này và R2 từ checkpoint khác.

### 6.3. Diagnostics cần log

- validation CE và ROUGE-1/2/L;
- output/reference length, EOS, repeated 3-grams;
- copy rate, copy precision, source-visible unigram/bigram coverage;
- `pi_base`, `pi_sem`, `pi_copy`, `g`, `alpha`, cap-hit rate;
- `RMS(delta)/RMS(h)`, logit-margin changes và semantic delta sau BF16 cast;
- cross-attention entropy và source-prior scale;
- gradient norm pre/post clip, clip coefficient, từng optimizer group;
- prefill latency/VRAM chỉ là secondary metric.

### 6.4. Promotion rule

Một candidate được gọi là thắng architecture chỉ khi:

1. `R1 > 49.626`, `R2 > 21.953`, `RL > 45.895` trên cùng protocol;
2. cải thiện lặp lại qua seeds với khoảng bất định đã đăng ký;
3. validation CE không xấu đi đáng kể;
4. không có giảm copy precision/source-visible coverage giải thích bằng leakage;
5. ablation control cho thấy gain đến từ candidate, không chỉ từ decoder setting, checkpoint selection hoặc soup.

Nếu F2 tăng hơn F1 nhưng G1/G2 không tăng, kết luận là training/postprocess tốt hơn chứ không phải CED-derived architecture. Nếu G chỉ tăng R2 và làm tụt R1/RL, giữ F2 hoặc F1 và không thêm planner/contrastive loss để che regression.

## 7. Ranh giới novelty và claim

Không được gọi các cơ chế sau là novelty riêng: CED, copy mixture, pointer-generator, probability mixture, semantic attention, sigmoid gate, SWA/model soup hay n-gram memory.

Nếu LAASP vượt control, claim an toàn hơn là:

> Một adapter source-conditioned theo layer, khởi tạo identity, cho phép điều chỉnh retrieval/content anchor trong cross-attention của một summarizer ghép checkpoint, trong khi giữ đường copy và probability base tách biệt.

Nếu F1/F2 thắng mà LAASP không thắng, claim là **training stabilization/checkpoint averaging cho AFMR**, không phải DeepSeek-style CED quality gain. Nếu chỉ latency giảm, đó là system contribution. Hallucination cần metric factuality độc lập; ROUGE không đủ.

## 8. Kết luận theo mức độ tin cậy

### Source fact or data

- DeepSeek CED tạo global KV decoder từ hidden anchor giữa stack; SWA vẫn layer-local.
- CSA2 có Full/Reindex/Reuse và hierarchical candidate restriction.
- Engram dùng hashed n-gram conditional memory và contextual gating.
- YOCO/DeepSeek báo lợi ích chính ở cache, prefill, context length và throughput.
- AFMR hiện đã có `H0/M`, layer-specific cross projections, copy tách riêng và probability mixture.

### Reasoned inference

- CED full không có lý do trực tiếp để tăng ROUGE PubMed và có rủi ro phá warm-start.
- Anchor adapter nhỏ có thể kiểm tra source alignment theo layer mà vẫn giữ copy path.
- Warmup + EMA/SWA/soup có expected risk thấp hơn thay topology.
- Engram-lite có thể hỗ trợ lexical recall nhưng là pilot rủi ro cao hơn.
- DAPT có thể tăng domain prior nhưng là data/training contribution, không phải evidence CED tăng chất lượng.

### Unverified

- Bất kỳ candidate nào có thực sự vượt `new` ở R1/RL và `update_v2` ở R2.
- LAASP có tăng score hay chỉ tăng capacity/variance.
- Engram-lite có giúp thuật ngữ y sinh mà không hallucinate.
- DeepSeek CED có lợi cho một model 1.37B fine-tune trên PubMed.

## Nguồn

[^1]: DeepSeek-V4.1-Flash, *trích xuất và phân tích kỹ thuật* trong repository này, đặc biệt PDF tr. 9–12 và tr. 20; [bản Markdown](/Users/kieugiangbien/Downloads/Project/LLM2Seq/Technical_Report/DeepSeek_V41_Tech_Report_Extraction.md). Report không cung cấp CED-only ROUGE ablation.
[^2]: Yutao Sun et al., *You Only Cache Once: Decoder-Decoder Architectures for Language Models*, arXiv:2405.05254. [Bài gốc](https://arxiv.org/abs/2405.05254).
[^3]: DeepSeek-AI, *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*, arXiv:2601.07372. [Bài gốc](https://arxiv.org/abs/2601.07372).
[^4]: Wortsman et al., *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time*, arXiv:2203.05482 / ICLR 2022. [Bài gốc](https://arxiv.org/abs/2203.05482).
[^5]: Peng Lu et al., *Improving Generalization of Pre-trained Language Models via Stochastic Weight Averaging*, arXiv:2212.05956. [Bài gốc](https://arxiv.org/abs/2212.05956).
[^6]: Gururangan et al., *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks*, ACL 2020. [Bài gốc](https://aclanthology.org/2020.acl-main.740/).

Các nguồn trên xác nhận cơ chế và tiền lệ. Kỳ vọng tác động lên ROUGE, thứ tự ưu tiên và công thức LAASP là phân tích thiết kế cho AFMR; chưa phải kết quả thực nghiệm.
