# Thiết kế contrastive training cho EviSeq từ SDACL

Ngày: 2026-09-08. Trạng thái: **người dùng đã chốt hướng; chưa tích hợp objective vào training**.

Kế hoạch triển khai theo module và tiêu chí kiểm tra nằm trong
[implementation plan](EVISEQ_EVIDENCE_CONTRASTIVE_IMPLEMENTATION_PLAN.md).

Nguồn xuất phát là [report SDACL](SDACL_METHOD_EXTRACTION.md) và PDF người dùng cung cấp:
[SDACL.pdf](/Users/kieugiangbien/Downloads/Paper/VAI/Summarization/SDACL.pdf),
đặc biệt §3.2-3.3, Eq. (6)-(15), Table 4-5.
Đã đọc lại code hiện tại trong src/eviseq_update, không suy ra graph chỉ từ README.

NEFTune đã được gỡ khỏi model, config và script. Baseline thực thi: gold-reference CE,
1 epoch interface warmup + 3 full, global batch 96, clip 1.0, LR warmup + cosine.
Sampling temperature/top-p vẫn là API riêng, không được gọi trong training.

## 1. Quyết định nghiên cứu

**Ứng viên nên nghiên cứu: dùng contrastive supervision để phân biệt các vị trí
nguồn có cùng từ nhưng khác ngữ cảnh, trên cả copy attention và semantic-read
attention đang có.**

Tên mô tả tạm thời: *Evidence disambiguation for decoupled copy and semantic reading*.
Chưa đặt tên như một thuật toán đã chứng minh novelty.

Lý do chọn: EviSeq có hai đường đọc nguồn với hai không gian token khác nhau, và
phép cộng xác suất copy theo token ID làm mất thông tin về vị trí. Đây là một vấn
đề cụ thể để đặt giả thuyết, thay vì gắn một projection head rồi thêm InfoNCE chung.

Áp dụng phương pháp có trích dẫn không phải đạo văn. Để có đóng góp phương pháp,
cần chứng minh vấn đề, thiết kế và thực nghiệm vượt quá việc thay backbone hoặc
đổi hệ số. Không thể bảo đảm “chưa ai làm” chỉ vì tên mới hoặc vài truy vấn tìm kiếm.

## 2. Bài học thực sự từ SDACL

| Thành phần | SDACL gốc | Điều rút ra cho thiết kế |
|---|---|---|
| Cấp so sánh | Câu source với gold summary/câu summary | Chọn evidence ở cấp nhỏ hơn document có thể hữu ích |
| Representation | Hidden [CLS] của cùng encoder | Không bắt buộc giữ nguyên nếu mục tiêu là sửa đường đọc |
| Gold representation | Encoder riêng cho target, Eq. (6) | Không được gọi reproduction là một lượt source-encoder |
| Summary nhiều câu | Max cosine với từng câu gold, Eq. (8) | Không nên gộp cả abstract thành một vector duy nhất |
| Positive/negative | Top/bottom similarity trong document | Không cần sinh candidates; negative có thể lấy từ source có sẵn |
| Hardness | Khoảng cách semantic và chất lượng positive | Phân biệt “khó cho model” với “nhãn không đáng tin” |
| Nơi loss tác động | Chủ yếu encoder representation | EviSeq cần kiểm tra cả nơi decoder chọn evidence |
| PubMed | PEGASUS 45.09/19.56/40.42 → 47.89/21.05/42.96 | Là bằng chứng để thử, không phải dự đoán mức tăng của EviSeq |

Nguồn: PDF SDACL trang 4-8. Backend paper là rouge-score 0.1.2, input PubMed 1024
tokens, target 256. Các điểm đó không so trực tiếp với Perl155 và input 4096 của
EviSeq. Mục 11 của report trích xuất trước đây là đề xuất biến thể, không phải
phương pháp gốc và chưa đủ để trở thành contribution riêng.

Eq. (13) in weight âm trong khi Eq. (14) nhân weight vào negative denominator.
Không kế thừa công thức bất nhất này. Cũng không kế thừa tuyên bố giải quyết
exposure bias/metric mismatch chỉ vì thêm contrastive: phương pháp vẫn teacher forcing
và không tối ưu trực tiếp ROUGE.

## 3. Graph hiện tại và nơi objective có thể tác động

Đặt H0 = base_projection(encoder.final), M = AFMR memory và b = source_bias.

~~~text
encoder final ──base_projection── H0 ──> native semantic keys/values
      │                              └─> aligned copy context keys
      └─depth/feature adaptation── M ──> per-layer cross-attention keys
                                      values lấy từ H0
M + prompt + output budget ──> focus prior b

decoder hidden h_t
  ├─copy query + contextual/lexical keys + b──> a_copy(position)
  │                   └─sum by token ID──────> P_copy(v)
  └─semantic query + native keys + b─────────> a_sem(position)
                      └─read H0 values───────> bounded residual ──> P_LM(v)

P(v) = g_copy * P_copy(v) + (1-g_copy) * P_LM(v)
~~~

Các điểm xác minh từ code:

- encode_source trong modeling/model.py truyền value_memory vào grounded_copy.prepare
  khi dùng afmr_value_anchor. Copy head và native semantic head đọc H0, không phải
  chỉ M. Một loss pool M chung không trực tiếp giám sát các lựa chọn trong hai head.
- GroundedCopyHead._attention tạo copy logits theo decoder-source tokenization.
- GroundedCopyHead.read có semantic_query/semantic_key riêng và đọc native encoder
  positions; residual RMS được chặn ở 0.10 lần RMS hidden trong config PubMed.
- GroundedCopyHead.loss lấy logsumexp trên **mọi vị trí source có token ID bằng
  target**. _mix_logits dùng scatter_add theo token ID với cùng ý nghĩa.
- AFMRTrainer._LossOnlyModel lấy loss_ce. Chỉ thêm output.loss mới sẽ không tự khiến
  trainer tối ưu auxiliary loss.
- Decoder dùng hidden[:, :-1] để dự đoán labels[:, 1:]. Mọi nhãn phụ phải theo cùng
  phép dịch này.

## 4. Vấn đề: đúng từ chưa phân biệt được đúng lần xuất hiện

Ví dụ minh họa do ta tạo, không phải sample kết quả:

~~~text
Câu A: Mortality decreased in the treatment group.
Câu B: Mortality increased in the control group.
Gold : Mortality decreased in treated patients.
~~~

Ở bước dự đoán “Mortality”, cả A và B đều cung cấp cùng token. Nếu tổng xác suất
copy của hai vị trí không đổi, nhánh copy cho cùng P_copy("Mortality"), dù phần lớn
attention nằm ở A hay B.

Với x_i là token nguồn theo tokenizer decoder:

\[
C_t = \{i:x_i=y_t\},\qquad
P_{\mathrm{copy}}(y_t)=\sum_{i\in C_t}a^c_{t,i}.
\]

Khi giữ P_LM và g_copy cố định, mọi tái phân bổ a_copy bên trong C_t có cùng tổng sẽ
cho cùng token CE. Đây là **bất định vị trí ở phép marginalization copy**, không
phải tuyên bố toàn bộ mạng bất biến: gate hiện tại phụ thuộc copy context, và h_t,
P_LM cũng phụ thuộc source.

Hệ quả cần kiểm chứng: model có thể phát âm đúng từ nhưng đọc context không phù hợp
cho cụm từ, quan hệ hoặc phần diễn đạt quanh nó. CE của các token kế tiếp có thể
khắc phục phần nào, nên đây không phải bằng chứng model đang lỗi hoặc chắc chắn cần
loss mới.

Tính thử với P_LM(y)=0.2, g_copy=0.6, tổng copy đúng từ=0.5:

| a_copy ở evidence đúng / occurrence khác | CE của token | -log P(evidence đúng \| cùng token) |
|---|---:|---:|
| 0.40 / 0.10 | 0.967584 | 0.223144 |
| 0.10 / 0.40 | 0.967584 | 1.609438 |

**Chỉ sửa copy occurrence chưa đủ để hứa tăng ROUGE.** Phân bố từ ở bước này vẫn
bằng nhau; semantic-read objective phải làm việc chọn context có ích cho generation.
Đó là lý do giữ cả hai head trong giả thuyết chính.

## 5. Tạo supervision từ source/gold sẵn có

Không thêm teacher model, không chạy source encoder lần hai, không sinh câu.
Các cặp bên dưới là nhãn alignment yếu suy ra từ gold đã có; không gọi chúng là
nhãn factuality được con người xác minh.

### 5.1. Đơn vị target và xử lý tokenizer

- Tách source và target gold thành câu/cụm ngắn với character offsets ổn định.
- Dùng từ đầy đủ hoặc cụm có nội dung làm anchor; không dùng mọi BPE fragment,
  dấu câu hoặc stopword như một evidence unit độc lập.
- Map anchor về target token positions bằng tokenizer decoder.
- Map occurrences source về copy-token positions và native encoder positions bằng
  character offsets. Tận dụng alignment hiện có, nhưng cần giữ thêm offsets/IDs.
- Chỉ dùng source thực sự còn nhìn thấy sau truncation; phần reference không tìm
  được evidence trong vùng đó vẫn giữ CE và không nhận auxiliary loss.

Không giả định source token ID của PPLX bằng token ID của Qwen.

### 5.2. Các đoạn dễ nhầm phải chứa cùng anchor

Với một target anchor a, tập source candidate gồm những occurrence cùng từ/cụm.
Trong loss copy ở mỗi subtoken t, tiếp tục giới hạn candidate theo x_i=y_t.

Positive là occurrence nằm trong vùng có ngữ cảnh khớp phần target chứa a.
Negative là occurrence cùng anchor trong vùng có ngữ cảnh ít khớp hơn rõ ràng.
**Câu không được chọn chưa chắc là nhiễu cho cả abstract:** nó có thể hỗ trợ một
câu target khác. Nhãn phụ phụ thuộc target unit hiện tại.

Đây là thay đổi chính so với SDACL: bottom cosine thường tạo easy negatives;
ở đây negative bị ràng buộc chứa cùng từ/cụm mà copy có thể phát ra.

### 5.3. Tránh nhãn chỉ phản ánh trùng từ

Một mining recipe khởi đầu có thể triển khai:

1. Chuẩn hóa để alignment so khớp được spacing, giữ mapping về chuỗi gốc. Không sửa
   prediction/reference metric files.
2. Với từng occurrence, lấy câu hoặc local clause chứa nó; chỉ dùng sentence-level
   trước, clause-level là ablation riêng.
3. Khi chấm độ khớp với target sentence, bỏ chính anchor khỏi cả hai phía.
   Dùng context-word overlap có trọng số IDF và phrase/LCS overlap của phần còn lại.
   IDF nếu dùng chỉ fit trên train.
4. Yêu cầu bằng chứng ngoài anchor, ví dụ nhiều content words/context phrase trùng,
   và khoảng cách đủ rõ giữa nhóm được nhận/nhóm bị loại. Ngưỡng là hyperparameter
   của mining, phải log và chọn trên train/dev, không phải giá trị từ SDACL.
5. Nhiều occurrence có evidence tương đương đều được làm positive. Ties hoặc
   paraphrase khiến không xác định được thì bỏ supervision cho unit đó.
6. Không tự xem khác con số/negation là chứng minh contradiction. Dùng ngữ cảnh
   phân biệt; nếu không đủ, bỏ nhãn. Nhãn yếu không thay một entailment judge.

Bỏ anchor lúc mining giúp ví dụ không tự thắng chỉ vì có đúng từ được copy.
Điều này làm supervision khắt khe hơn nhưng giảm coverage; chưa biết đủ pairs trên
PubMed hay không vì local package chỉ có fixtures, chưa có dữ liệu train thực.

Phải audit ít nhất một tập train mẫu về nhãn đúng, coverage và bias theo section,
độ dài, câu có nhiều evidence, số/đơn vị và negation trước full training.

### 5.4. Confidence khác hardness

SDACL tăng weight khi positive-negative khó phân biệt. Với nhãn yếu, score gần nhau
cũng có thể do không biết evidence nào đúng. Vì thế thiết kế này:

- giảm/bỏ weight khi **alignment không chắc**;
- giữ hard negatives có cùng anchor khi **alignment đủ chắc**;
- confidence c_a nằm [0,1], không trainable, không phụ thuộc attention đang tối ưu.

Đây là quy tắc kiểm soát lỗi nhãn, không phải một novelty độc lập đã được xác minh.

## 6. Objective: contrast trong tập occurrence có thể bị nhầm

Giữ hai attention độc lập. Không ép a_copy bằng a_sem theo từng token.

### 6.1. Copy: phân biệt occurrence sau khi đã cố định từ

Gọi P_t^c là tập positive occurrences, N_t^c là negative occurrences chắc chắn;
C_t^c=P_t^c∪N_t^c chỉ gồm vị trí có cùng target token ID. Occurrence không rõ nhãn
được loại khỏi cả hai tập. Dùng logits thật u^c của copy head:

\[
\ell^c_t =
\operatorname{LSE}_{i\in C_t^c} u^c_{t,i}
-\operatorname{LSE}_{i\in P_t^c} u^c_{t,i}
= -\log \frac{\sum_{i\in P_t^c} a^c_{t,i}}
                     {\sum_{i\in C_t^c} a^c_{t,i}} .
\]

Đây là set-based conditional likelihood / một dạng supervised contrastive alignment.
Công thức logsumexp không phải công thức mới. Đóng góp dự kiến nằm ở tập cạnh tranh
và chỗ supervision được gắn vào kiến trúc.

- Không ép tất cả probability chuyển sang copy hoặc g_copy=1.
- Không phạt token không có occurrence nguồn.
- Không chọn ngẫu nhiên một “đáp án duy nhất” nếu nhiều source spans cùng hỗ trợ.
- Gradient ngoài candidate set bằng zero khi xét các logits là biến độc lập.
  Vì logits dùng chung weights, tham số thay đổi vẫn có thể ảnh hưởng vị trí khác.

### 6.2. Semantic read: phân biệt vùng ngữ cảnh của cùng anchor

Dùng chính semantic logits u^s trong GroundedCopyHead.read. Positive P_t^s gồm native
encoder tokens của các vùng context nhận làm positive; negative N_t^s là vùng context
của occurrences cùng anchor bị loại.

\[
\ell^s_t =
\operatorname{LSE}_{k\in P_t^s\cup N_t^s}u^s_{t,k}
-\operatorname{LSE}_{k\in P_t^s}u^s_{t,k}.
\]

Cả câu/context được dùng thay vì ép semantic attention vào riêng chữ anchor.
Semantic head vẫn có thể chọn các token mô tả quan hệ, kết quả hoặc diễn đạt quanh
anchor. Hai head cùng nhận supervision về evidence nhưng thực hiện hai chức năng.

Các vùng positive/negative phải rời nhau; nhiều occurrence trái nhãn trong cùng
câu không thể dùng nguyên câu cho cả hai. Bỏ unit đó hoặc tách clause bằng một
phiên bản mining được ghi rõ, không tạo mask chồng lấn.

Không uniform hóa attention trên vùng positive. Cần kiểm tra sentence-length bias
vì logsumexp tích lũy mass theo số token. Loss length-normalized là ablation khác,
không còn đúng xác suất attention mass của công thức trên.

### 6.3. Tổng loss và nhiệt độ

Với unit a chứa các token hợp lệ T_a, lấy trung bình theo unit để cụm nhiều subtokens
không tự có weight lớn hơn:

\[
\ell_a=\frac{1}{|T_a|}\sum_{t\in T_a}
             \frac{\ell^c_t+\ell^s_t}{2},\quad
L_{\mathrm{evidence}}=\frac{1}{M}\sum_{a=1}^M c_a\ell_a .
\]

\[
L = L_{\mathrm{CE}}+\lambda L_{\mathrm{evidence}}.
\]

M đếm unit hợp lệ trước confidence weighting; không chia cho tổng confidence vì
điều đó có thể triệt tiêu tác dụng hạ weight khi mọi nhãn đều kém tin cậy.

Khởi đầu lambda=0.05; đối chứng lambda=0 và một giá trị nhỏ/lớn hơn nếu dev có tín
hiệu. Đây là lựa chọn engineering, chưa phải tối ưu đã được paper chứng minh.
Dùng logits native không chia thêm temperature (tương đương tau_aux=1). Scale
attention 1/sqrt(rank) vẫn giữ. Nếu đổi tau_aux, phải ghi là surrogate khác và không
còn đúng conditional probability của actual attention.

Không dùng soft weight âm của SDACL, không nhân loss với gate copy trainable và
không thêm KL buộc hai attention giống nhau.

## 7. Gradient và lộ trình training

| Đường | Gradient trực tiếp từ auxiliary | Những gì không được bảo đảm |
|---|---|---|
| Copy | query, context_key, lexical_key, source_bias và upstream hidden/H0 | Không có loss bắt gate copy tăng |
| Semantic | semantic_query/key, source_bias và upstream hidden/H0 | semantic_value/output/gate không nhận gradient trực tiếp từ attention-only loss |
| Shared source | base_projection, encoder nếu đang mở, AFMR focus; M ảnh hưởng qua h_t/prior | Không ép anchored values phải giống retrieval keys |
| Decoder | hidden trước target → các layers/cross-attention đang trainable | Attention đúng không chứng minh LM sử dụng evidence |

Vẫn cần CE để học semantic values, residual output/gate và từ ngữ tự nhiên.
Trong interface warmup, encoder/base decoder frozen theo code; cross/bridge trainable.
Kế hoạch khởi đầu giữ epoch interface CE-only để không bắt các head mới học đồng
thời nhãn yếu. Full stage tăng lambda tuyến tính từ 0 đến đích trong khoảng 10%
updates đầu, sau đó cố định. Ramp này là hyperparameter, phải ablate nếu kết luận
tác dụng của curriculum.

Không detach logits/hidden dùng tính loss. Chỉ mining/index/confidence không có grad.
Không cập nhật nhãn bằng model outputs hoặc một vòng self-improve.

Dùng hidden trước khi thấy target token: index j của hidden dự đoán labels[j+1].
Gold toàn câu được dùng để xây loss target, không nối vào source, prompt hay attention
mask đầu vào. Đây là supervised target như CE; inference không cần gold metadata.

## 8. Chi phí và thay đổi code nếu triển khai

Một lượt source encoder và một lượt teacher-forced decoder. Không có generation,
reranking, queue negatives cross-GPU hay target encoder pass. Không cần thêm
trainable projection head cho auxiliary, vì dùng query/key của actual read paths.

“One forward” không có nghĩa không tăng chi phí: scatter/segment reductions,
logsumexp, gradient và mining vẫn có overhead; activation checkpointing vẫn
recompute trong backward. Cần benchmark CUDA, không báo tỷ lệ overhead từ CPU toy.

| File/module | Thay đổi cần làm (chưa làm) |
|---|---|
| data/copy_alignment.py, data/collate.py | Giữ offsets, thêm sparse evidence mapping và shifted target unit IDs |
| modeling/grounded_copy.py | Tính numerator/denominator trong chunk_loss, tái dùng copy/semantic scores |
| modeling/decoder.py | Trả CE sum và evidence sum/count; không giữ full attention tensor để tính loss sau |
| modeling/outputs.py, modeling/model.py | Output tách loss_ce, evidence_sum, evidence_count và total loss contract |
| training/engine.py | Thay _LossOnlyModel vốn chỉ lấy loss_ce; chuẩn hóa hai loss theo denominator riêng |
| config.py, training/checkpoint.py | Cấu hình mining/lambda/ramp, version/hash alignment; guard resume objective |
| scripts/run_pubmed_pair.sh | Recipe mới tách directory, CE baseline vẫn chạy được |

Với batch84, target512, source4096, một tensor attention FP32 có khoảng 0.656 GiB;
hai full tensors cộng backward sẽ đáng kể. Tính auxiliary ngay trong chunks hiện
tại, chỉ trả scalar sums/counts. Không xin attention weights từ mọi decoder layer
vì sẽ ảnh hưởng đường SDPA và memory.

### Chuẩn hóa trên 1/2 GPU và accumulation

CE hiện chia theo tổng gold tokens toàn accumulation window và mọi rank.
Evidence loss theo tổng unit hợp lệ, không chia nhờ CE token count.

Cho rank r, một microbatch có CE_sum_r và evidence_weighted_sum_r; N, M là tổng
denominator của **cả window, mọi rank**, W=world_size:

\[
L_r^{backward}=W\,\frac{CE\_sum_r}{\max(1,N)}
+\lambda W\,\frac{evidence\_weighted\_sum_r}{\max(1,M)} .
\]

DDP average và cộng các microbatch cho đúng global objective. CE và evidence logs
cũng dùng denominator riêng. Rank không có positive hợp lệ vẫn tham gia collective;
zero auxiliary phải graph-connected khi cần tránh unused-parameter/reducer lỗi.
M và confidence xác định từ metadata trước forward nên có thể all-reduce count
trước backward.

Chỉ đặt total_loss rồi để trainer nhân theo CE tokens sẽ tạo weight phụ thuộc
microbatch/world size và không đúng objective đã nêu.

## 9. Vì sao có thể giúp ROUGE, và khi nào sẽ thất bại

Giả thuyết: same-word contrast huấn luyện lựa chọn context, semantic read sử dụng
context đó để hỗ trợ những từ liên kết/quan hệ xung quanh, CE giữ áp lực diễn đạt
đúng reference. Có thể có ích cho ROUGE-2/L và nội dung quan trọng bị lẫn giữa các
section. **Đó là suy luận, chưa có kết quả EviSeq cho thấy quan hệ nhân quả này.**

Các khả năng thất bại phải đo:

- Không đủ repeated-anchor pairs tin cậy: objective quá thưa, không giải quyết
  salience chung. Không nới label bừa để tăng coverage.
- Gold là paraphrase: overlap mining bỏ qua nhiều nội dung, bias về extractive spans.
- Nhiều evidence hợp lệ bị gán negative: loss gây giảm chất lượng.
- Semantic residual/gate bị model giảm tác động: attention alignment cải thiện nhưng
  generation không đổi. Theo dõi RMS(delta)/RMS(h), gate và sensitivity của logits.
- Copy occurrence được sửa nhưng P_copy giữ nguyên: ROUGE có thể không tăng.
- Global source_prior b giải phần lớn loss thay cho query-dependent read: ablation
  bỏ gradient trực tiếp từ auxiliary tới prior để kiểm tra cơ chế.
- Model có thể tập trung vào một phần trong positive union và bỏ phần còn lại.
  Không claim coverage toàn summary từ set likelihood.
- Training CE tăng không đồng nghĩa lỗi: so clean validation CE, generated ROUGE,
  length/extractiveness. Không dùng total CE+aux để so với CE-only.

Không gọi “attention đúng” là metric hallucination hay explanation đáng tin cậy.
Muốn kiểm tra read có tác dụng, có thể làm can thiệp ở **evaluation**: che/đổi values
của evidence và so thay đổi gold log-prob với che cùng số tokens ở vùng đối chứng.
Giữ có/không renormalization và các can thiệp là protocol riêng; kiểm tra này cũng
không tự chứng minh factuality. Nó không thêm generated text vào training.

## 10. Novelty: phần có thể bảo vệ và các tiền lệ

Đã đối chiếu những nguồn gần nhất sau; đây là targeted prior-art search, không phải
xác nhận toàn bộ literature hoặc quyền ưu tiên.

| Tiền lệ | Điều đã có | Khoảng khác dự kiến phải kiểm chứng |
|---|---|---|
| SDACL 2024, PDF người dùng | Within-document sentence salience, semantic contrast, distance-aware weights | Chọn competitors theo same emitted token, giám sát actual copy/read thay vì CLS similarity |
| [Pointer-generator, ACL 2017](https://aclanthology.org/P17-1099/) | Copy distribution cộng các occurrences theo cùng word, trộn với vocabulary distribution | Bất định occurrence không phải phát hiện “copy mới”; cần đánh giá cách khai thác nó để training |
| [Bottom-Up, EMNLP 2018](https://aclanthology.org/D18-1443.pdf), §4.1-4.3 | Content selection, copy masking; có cả baseline sửa gold copy attention khi train | Không được claim “lần đầu giám sát vị trí copy”; phải so baseline alignment thông thường |
| [Inconsistency Loss, ACL 2018](https://aclanthology.org/P18-1013/) | Ghép sentence/word attention và khuyến khích nhất quán | “Cùng evidence nhiều cấp” cũng có tiền lệ; mục tiêu riêng ở đây là conditional same-token competition trong decoupled heads |
| [Contrastive Attention, EMNLP 2019](https://aclanthology.org/D19-1301/) | Relevant/opponent attention với softmax/softmin | Contrast trên actual attention không mới tự thân; đề xuất không xây opponent generation branch |
| [Span-level supervised attention, NAACL 2021](https://aclanthology.org/2021.naacl-main.225.pdf), §2, App. C.2 | Học attention từ span alignment, có biến thể không uniform hóa positive span | Không claim span-mass supervision hoặc freedom inside span là mới; khác ở conditional competing occurrences và cơ chế copy/read |
| [PROM, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1148/), §4 | Phrase overlap labeling, copy supervision và LM loss | Shared phrase labels không mới; cần contrast among occurrences, phụ thuộc target context, không chỉ dự đoán copyability |

**Phần mới có thể đề xuất ở mức hệ thống:** kết nối một bất định cụ thể của copy
marginalization với contrastive training có điều kiện theo token identity, đồng
thời giám sát semantic context ở native positions trong graph có query/key riêng
và anchored values. Không cần gold encoder hoặc generated negatives.

Đây là một cấu hình phương pháp đủ cụ thể để nghiên cứu. Nếu chỉ cùng hiệu quả với
ordinary span supervision trên cùng labels, đóng góp có thể chỉ là engineering
integration, chưa phải novelty phương pháp mạnh. Không che giấu tiền lệ bằng cách
đổi tên loss, thêm acronym hay gọi công thức logsumexp là một lý thuyết mới.

Có thể viết contribution sau khi có bằng chứng:

> We study source-occurrence ambiguity in a summarizer with decoupled copy and
> semantic reading, and train both read paths with reference-derived, token-identity-
> conditioned evidence supervision without generating training candidates.

Không viết “first”, “state of the art”, “reduces hallucination” hoặc “surpasses
T5Gemma” trước khi có kiểm chứng và prior-art search rộng hơn.

## 11. Ablation để có câu chuyện paper

Giữ cùng initialization/backbones, global batch/update budget, scheduler và decode.
NEFTune không có trong mọi run. Các ablation không phải tất cả cần train ngay.

| So sánh | Câu hỏi trả lời |
|---|---|
| A: CE-only vs B: EviSeq + SSCL/SDACL-style | Thêm contrastive generic có đủ không? |
| A vs C: same-token loss trên copy | Sửa riêng occurrence có ảnh hưởng generation không? |
| A vs D: same-context loss trên semantic read | Lợi ích có chỉ đến từ read supervision không? |
| C/D vs E: objective hai đường đề xuất | Có lợi ích khi hai đường cùng học evidence? |
| E vs F: ordinary span attention supervision với cùng labels/compute | Conditional competition có lợi hơn giám sát trực tiếp thông thường? |
| E vs G: random/disjoint-word negatives cùng số lượng | Hard negatives theo token identity có cần thiết? |

SSCL/SDACL exact cần target encoder pass; nếu dùng decoder hidden để tiết kiệm thì
ghi “SDACL-style adaptation”, không gọi reproduction. Dấu Eq. (13) phải công khai
cách xử lý. Cho generic baseline một budget tuning tương xứng.

Trước mắt chỉ chạy CE và E sau khi audit pairs, rồi C/D/F nếu dev có lợi ích.
B là đối chứng cần cho claim khác SDACL trong bài hoàn chỉnh, không bỏ vì bất tiện.

Ngoài ROUGE155, đo trên cùng tập:
- attention mass của evidence đúng trong nhóm cùng token;
- breakdown theo repeated-anchor density và mapping coverage;
- ROUGE theo source length, summary length, mức extractiveness;
- hội tụ, optimizer updates, throughput, GPU-hours, VRAM;
- clean CE và mức sử dụng semantic residual.

Dự đoán có thể bác bỏ: lợi ích lớn hơn ở tài liệu nhiều same-token competing
contexts. Nếu không thấy, phải sửa giải thích cơ chế thay vì chỉ báo điểm tổng.

Chọn hyperparameter/checkpoint trên validation; test cho kết quả cuối, paired
bootstrap theo document và nhiều seed khi làm claim. Các test scores cũ đã được
dùng định hướng phát triển thì cần minh bạch hoặc có confirmation split chưa dùng.
So với T5Gemma/decoder-only fine-tuned phải match dữ liệu, source visibility,
generation budget, metric và validation tuning budget.

## 12. Trạng thái thực hiện và điều kiện trước full training

Đã thực hiện:
- Bỏ implementation/config/script NEFTune, giữ CE + cosine mặc định.
- Giữ sampling temperature/top-p ở API riêng.
- Chặn config đã bỏ và resume checkpoint có noise khác 0 để không âm thầm đổi recipe.
- 235 tests qua trên CPU, PyTorch 2.12.1 / Transformers 5.16.1; gồm hai-process Gloo,
  parameter updates, scheduler, sampling và shell config generation.
- Kiểm tra số học của conditional loss: cùng copy marginal có cùng CE nhưng khác
  occurrence loss; gradients tăng positive và giảm competitor trong toy logits.

Chưa thực hiện:
- Chưa implement mining/contrastive objective vào EviSeq.
- Chưa có nhãn PubMed để đánh giá coverage/precision.
- Chưa có training CUDA, overhead measurement hoặc ROUGE mới.

Các bước triển khai theo thứ tự:
1. Audit nhãn trên source/gold train sau truncation và kiểm tra có đủ signal.
2. Implement scalar loss trong chunks, lambda=0 giữ CE/logits/gradient đúng baseline.
3. Tests valid/empty/disjoint masks, repeated IDs, truncation, subtoken boundaries,
   dense/chunked loss và gradient, và DDP với số evidence units không đều.
4. Train thử ngắn trên một train subset cố định và validation cố định.
5. Chạy full và ablations nếu có signal; không biến một heuristic chưa đo thành default.

**Kết luận thiết kế:** SDACL cho cơ sở để dùng contrastive trên source evidence.
Đóng góp EviSeq nên được đặt ở cách phân biệt evidence mà cơ chế copy làm nhập nhằng
và cách semantic read sử dụng nó. Điều đó cụ thể hơn việc cộng SDACL trực tiếp,
nhưng độ mới và khả năng tăng ROUGE vẫn phải được thử nghiệm.
