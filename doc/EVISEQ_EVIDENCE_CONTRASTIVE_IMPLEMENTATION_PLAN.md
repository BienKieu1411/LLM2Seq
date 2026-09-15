# EviSeq: kế hoạch code tích hợp evidence contrastive training

Ngày: 2026-09-08. **Hướng nghiên cứu đã được người dùng chốt và đã tích hợp vào `src/eviseq_update`.**

Trạng thái hiện tại: đã có cache mining tokenizer-only, sparse collation, set-based
multi-positive contrastive loss trên actual copy/semantic reads, chuẩn hóa theo
toàn bộ window/DDP, lambda ramp, checkpoint/config guard và recipe shell. Đã qua
test CPU một process, DDP Gloo có placeholder rank, cache CLI và regression suite.
Chưa chạy full PubMed/GPU, nên chưa có claim ROUGE hoặc factuality mới.

Cơ sở:
- [Thiết kế đã chốt](EVISEQ_EVIDENCE_CONTRASTIVE_DESIGN.md).
- [Report SDACL và đính chính](SDACL_METHOD_EXTRACTION.md).
- Code hiện tại: src/eviseq_update.

## 0. Objective đã chốt là supervised contrastive learning

Chuẩn bị alignment là bước dựng positive/negative pairs. Phần model học là
**query–source contrastive loss trên representations của các read heads hiện tại**.
Không thêm một classifier dự đoán nhãn salience rồi coi classifier đó là toàn bộ
phương pháp. Tên kỹ thuật của loss là set-based multi-positive contrastive likelihood;
với một positive, nó có dạng InfoNCE trên candidate set đã chọn.

Với mỗi nhánh b thuộc {copy, semantic}:

~~~text
anchor query q_t^b = projection_b(normalized decoder hidden trước target token)
positive keys     = source keys thuộc evidence phù hợp với target context
negative keys     = source keys thuộc các occurrence/context dễ nhầm
score(t,i)        = q_t^b · k_i^b / sqrt(rank_b) + source_bias_i^b

L_contrastive(t,b) = -log(
    sum_{i in positives} exp(score(t,i))
    / sum_{i in positives union negatives} exp(score(t,i))
)
~~~

Gradient đi qua q và k của **actual read**, không detach chúng. Copy có contextual
và lexical keys đã ghép; semantic có native keys độc lập. Với nhiều positives,
loss tăng tổng positive mass, không đảm bảo mọi positive đều được kéo gần như nhau.

Khác với SDACL: dùng decoder query phụ thuộc bước sinh làm anchor; chọn các negative
có cùng khả năng phát ra từ cần dự đoán; contrast ngay trên hai read paths. Phần
kế thừa SDACL là lấy positive/negative trong source để học phân biệt evidence.
Không cần một lượt encoder cho gold summary hay generated candidates.

Các công thức conditional attention mass trong plan chính là cách triển khai loss
contrastive này bằng logsumexp ổn định. Ordinary span-supervision chỉ xuất hiện như
**ablation để đối chứng novelty**, không thay thế objective chính.

## 1. Phạm vi phiên bản đầu

Tích hợp ngay trong package eviseq_update bằng một recipe riêng, giữ CE-only để đối chứng.

Mục tiêu training:

\[
L = L_{\mathrm{CE}}+\lambda L_{\mathrm{evidence}},
\qquad
L_{\mathrm{evidence}}=\frac{1}{M}\sum_a
  \frac{c_a}{|T_a|}\sum_{t\in T_a}
  \frac{\ell^c_t+\ell^s_t}{2}.
\]

- Copy học phân biệt các occurrence cùng token ID.
- Semantic read học vùng context tương ứng trên native encoder positions.
- Dùng source/reference sẵn có, một lượt source encoder và một lượt decoder teacher forcing.
- Giữ query/key độc lập và bounded residual hiện tại.
- Không NEFTune, R-Drop, teacher model, target encoder pass, generated negative hay self-improve.
- Không thêm trainable projection head; sử dụng logits của actual copy/read.
- Giữ API temperature/top-p riêng cho inference.

Ba chế độ objective: copy, semantic, both. Bản chính là both; hai chế độ còn lại
phục vụ ablation cùng dữ liệu alignment, không thay graph.

Phiên bản đầu chỉ chấp nhận kiến trúc afmr_value_anchor, grounded_copy enabled,
semantic_read enabled với independent_source. CE-only vẫn chạy được các graph cũ.
Mở rộng sang shared_v1 hoặc LM-only là nghiên cứu khác, không âm thầm fallback.

## 2. Các quyết định cần khóa trước khi code

### 2.1. Mining offline, không chạy lại theo weights

Tạo evidence cache **một lần** trên source/gold bằng tokenizers và quy tắc text.
Không load encoder/decoder weights. Training chỉ đọc cache và dựng tensor metadata.

Lý do: mining trong collator mỗi epoch có thể làm CPU thành bottleneck trên 2 GPU;
nhãn phụ thuộc weights còn làm khó resume và kiểm soát thực nghiệm.

Cache riêng cho từng cặp tokenizer và source/target limits. PPLX và Qwen-Embedding
không dùng chung token-position cache chỉ vì cùng dataset.

Data gốc vẫn là đầu vào chuẩn; cache là sidecar. Cache mất/hết hạn làm recipe mới
báo lỗi rõ ràng, không lặng lẽ chuyển về CE-only.

### 2.2. Đơn vị evidence

- Anchor là một từ đầy đủ có nội dung, hoặc số kèm đơn vị được match nguyên cụm.
  V1 không cố phát hiện mọi biomedical entity và không dùng NER model.
- Không dùng stopword, dấu câu hay BPE fragment đơn lẻ làm anchor.
- Một occurrence của anchor trong target là một unit a; cùng từ ở hai câu gold có
  thể tạo hai unit với evidence khác nhau.
- Các unit target không chồng nhau; ưu tiên số+đơn vị rồi từ đầy đủ.
- Cùng anchor phải xuất hiện trong ít nhất hai câu source còn nhìn thấy.
- Khi có mâu thuẫn nhãn trong cùng source sentence, bỏ unit ở v1.
- V1 dùng source sentence làm context; clause-level/phrase-level mining tổng quát
  để dành ablation sau, tránh đổi quá nhiều biến.

### 2.3. Recipe mining khởi đầu có thể triển khai được

Tất cả là lựa chọn engineering để audit, **không phải hyperparameter từ SDACL**.

1. Dùng đúng normalized strings sau data.normalization.detokenize như dataset.
2. Tách câu có character offsets bằng quy tắc xác định, xử lý decimal, viết tắt và
   xuống dòng; không tải model sentence splitter. Bộ fixtures phải có e.g., et al.,
   số thập phân, phần trăm, đơn vị và dấu câu biomedical.
3. Tokenize bằng chính đường code dùng trong collator; chỉ giữ anchor và context
   hợp lệ trong source visibility thực tế. V1 bỏ source/target sentence bị cắt cụt
   nếu cần phần bị cắt để quyết định nhãn.
4. Trong từng câu target chứa anchor, tìm các câu source chứa cùng anchor.
5. Loại chính anchor khỏi context của hai phía trước khi so khớp.
6. Đặt q(s,r|a) trong [0,1] là trung bình:
   - F1 của content-word multiset overlap;
   - LCS F1 trên chuỗi content words còn lại.
   Không stem, không nhập hai con số/đơn vị/negation khác nhau thành cùng token.
   V1 dùng trọng số từ đồng đều; chưa thêm IDF/BM25 để không có bước corpus fitting.
7. Positive: q >= max(q)-positive_band, q >= min_positive_score và có ít nhất
   min_context_matches content words khớp ngoài anchor.
8. Negative: q <= max_negative_score và q <= max(q)-min_context_gap.
   Các câu ở giữa không tham gia loss.
9. Giữ nhiều positive nếu tương đương. Không chọn một positive tùy tiện khi có ties.
   Không đủ positive/negative, context rỗng hoặc mọi câu đều tương đương: bỏ unit.
10. Confidence c_a = min_positive_q * (min_positive_q - max_negative_q), chặn [0,1].
    Không trainable, không tính từ model logits.
11. Map source/target offsets sang token positions rồi kiểm tra lần cuối:
    mỗi timestep tham gia copy phải có positive và negative có ID bằng gold label.
    Lấy T_a là các timesteps còn hợp lệ, giữ một unit khi T_a không rỗng.
12. Cap units/record và source sentences/unit để chặn cache size/compute.
    Chọn theo confidence, ties theo vị trí để deterministic; thống kê phần bị cap.

Defaults đề xuất cho vòng audit đầu: min_positive_score=0.30, max_negative_score=0.15,
positive_band=0.05, min_context_gap=0.20, min_context_matches=2,
max_units_per_example=32, max_positive_sentences=4, max_negative_sentences=4.
Với các threshold này, confidence có thể nhỏ; phải log distribution và gradient
trước khi chọn lambda. Không gọi 0.05 là “hệ số paper” hay cam kết scale tối ưu.

Nếu coverage/precision không đạt trên dữ liệu thật, sửa mining rồi tăng version;
không tự giảm ngưỡng lúc train và không đổi sang sinh pseudo-reference.

## 3. Data/cache contract

### 3.1. Những file mới

~~~text
eviseq_update/data/evidence.py
    evidence dataclasses, invariants, sparse tensor keys
eviseq_update/data/evidence_mining.py
    sentence/word spans, context scoring, deterministic positive/negative mining
eviseq_update/data/evidence_cache.py
    sidecar writer/reader, manifest fingerprint, per-process file handles
scripts/prepare_evidence.py
    cache preparation + audit commands, tokenizer-only
~~~

Đường dẫn trên tương đối với src/eviseq_update. Chỉ là file dự kiến.

### 3.2. Sidecar

Một JSONL entry cho mỗi record theo cùng thứ tự dataset, kể cả entry không có unit:

~~~text
row_index, record_id
normalized_source_sha256, normalized_target_sha256
source_input_ids_sha256, target_input_ids_sha256
units:
  confidence
  target_token_positions
  copy_positive_positions_per_target
  copy_negative_positions_per_target
  semantic_positive_spans
  semantic_negative_spans
skip_reason_counts
~~~

Token positions target trong cache tính trong target-only tokenization. Collator
cộng prompt length để chuyển sang decoder input positions, sau đó dịch -1 để
đánh chỉ số hidden dùng dự đoán. EOS và prompt không là evidence tokens.

Semantic spans là intervals native source token đã lọc content mask; nếu tokenizer
tạo lỗ, biểu diễn nhiều interval. Copy positions thuộc source sau tokenize bằng
decoder tokenizer, đã qua align_copy_tokens filtering. Deduplicate mọi vị trí.

manifest.json ghi:
- schema/mining version và toàn bộ mining config;
- hash dataset, normalization settings/version;
- tokenizer artifacts/revisions, special IDs;
- encoder prefix, decoder prompt/template, max lengths;
- hash sidecar, record count, coverage và lý do loại;
- runtime/library versions dùng lúc chuẩn bị.

Không hash chỉ model path: đổi file tokenizer trong cùng path phải invalid cache.
Ghi file tạm rồi rename nguyên tử, không để training đọc cache đang xây.
Reader index offset bằng row_index và verify hashes; không phụ thuộc ID là unique.
DataLoader spawn mở handle riêng theo PID, tương tự dataset hiện tại.

### 3.3. Những file hiện tại cần sửa

- data/schema.py: thêm optional evidence annotation vào CanonicalRecord, default None;
  as_dict() vẫn serialize source/reference như trước.
- data/dataset.py: đọc annotation từ reader sau khi normalize record; cache path chỉ
  được truyền cho train. Tránh làm normalization lần hai trên đã-normalized offsets.
- data/copy_alignment.py: tách helper alignment để mining và training dùng **cùng**
  kết quả token/offset/filtering. Mặc định COPY_INPUT_KEYS cũ giữ nguyên.
- data/collate.py: dựng EvidenceBatch, kiểm tra shifted labels, token IDs và masks.
  include_targets=False thì không đọc/tính/đưa evidence vào batch.
- data/sampling.py: DistributedCollator xóa evidence validity cho placeholder rows
  cùng lúc mask labels=-100, tránh học lại example đầu ở rank rỗng.
- runtime.py: resolve cache path, bật reader chỉ cho evidence training; validation
  CE và generation không yêu cầu cache.

### 3.4. Sparse batch tensors

Dùng flat top-level tensors để _move() hiện tại chuyển device được; không nhét
nested Python objects chứa GPU tensors vào batch mà _move không xử lý.

~~~text
evidence_unit_batch_index   [U]     int64
evidence_unit_confidence    [U]     float32
evidence_unit_valid         [U]     bool
evidence_target_unit        [V]     int64
evidence_target_hidden_pos  [V]     int64
evidence_target_weight      [V]     float32  # 1/|T_a|, tính trên cả unit
copy pair arrays           [Ec]    owner V, source position, positive flag
semantic span arrays       [Es]    owner U, start, end, positive flag
~~~

Mỗi timestep được nối với unit của nó; semantic regions chia sẻ trong một unit.
Không lưu permanent masks [batch,target,source]. Có thể dựng masks theo chunk.

Counts lấy sau placeholder masking và kiểm tra token mapping. CPU xác định toàn bộ
valid units trước forward. GPU không được âm thầm bỏ thêm unit mà trainer vẫn dùng
denominator cũ; invariant sai phải báo lỗi.

## 4. Contract loss và refactor model

### 4.1. Scalar statistics

Thêm kiểu LossStatistics trong modeling/outputs.py:

~~~text
ce_sum           differentiable scalar
gold_token_count integer scalar
copy_sum         differentiable scalar = sum_a c_a * mean_t ell_copy
semantic_sum     differentiable scalar = sum_a c_a * mean_t ell_sem
evidence_count   integer scalar = number of valid units, not sum(confidence)
~~~

loss_ce vẫn là ce_sum/max(1,gold_token_count) để giữ interface cũ.
Không đưa lambda hoặc global DDP normalization vào GroundedCopyHead.

Giữ public QwenCrossDecoder.forward trả tuple (logits,past,loss_ce) cho generation
và các caller cũ. Tách private computation dùng chung trả DecoderResult;
training entry point lấy LossStatistics từ computation này. Không gọi backbone
hai lần chỉ để lấy statistics.

EviSeqAFMR.forward thêm optional evidence inputs tường minh; không đẩy chúng vào
**copy_inputs của encode_source. AFMROutput bổ sung optional loss_stats ở cuối,
giữ legacy loss_ce và loss khi objective tắt.

### 4.2. Grounded copy / semantic read

Trong modeling/grounded_copy.py:

1. Tách calculation nội bộ để trả log-attention của copy và semantic khi cần.
   read/output_logits/distribution đang dùng cho inference giữ interface cũ.
2. Chỉ lấy các attention đã được tính cho CE ở cùng chunk. Không gọi lại read
   hoặc semantic attention chỉ để tính auxiliary.
3. Tính positive và candidate logsumexp ở FP32:
   ell = LSE(positive+negative) - LSE(positive).
   Dùng log-attention cũng tương đương logits vì cùng normalizer triệt tiêu.
4. Trong loss chunk, trả bộ ce_sum/copy_sum/semantic_sum.
   Gradient checkpointing replay cùng pure tensor computation, không cập nhật
   counters/Python lists/hook state bên trong hàm replay.
5. Target weight 1/|T_a| tính toàn sequence và lưu metadata. Nếu một unit trải qua
   nhiều chunks, chia theo toàn unit, không chia lại theo số tokens của chunk.
6. Nối zero với graph khi không có valid units; không tính logsumexp trên empty set
   rồi nhân zero, vì -inf - -inf có thể tạo NaN.
7. Không detach logits/hidden trong loss; mining/confidence không có grad.
8. Nhánh copy, semantic bật/tắt bằng scalar coefficients; không thêm loss ép gate.

Với nhánh disabled hoặc lambda=0, bỏ hẳn auxiliary computations. Cùng seed/input,
CE, logits và gradients phải khớp baseline trong tolerance đã chọn.

### 4.3. Dense/chunked reference

Production training vẫn return_logits=False và CE chunking.
Trong tests dùng dense reference tính cùng formula để xác minh losses/gradients,
không giữ dense full-attention cho PubMed.

Phiên bản đầu không lấy attention từ mọi decoder layer và không thay đường SDPA.
Memory phụ scale theo chunk và sparse metadata, không theo toàn B*T*S được lưu lâu.

## 5. Trainer/DDP: chỗ thay đổi quan trọng nhất

### 5.1. Đổi _LossOnlyModel thành wrapper tạo backward root đúng

Hiện wrapper chỉ trả .loss_ce. Wrapper mới nhận normalization coefficients đã biết
cho microbatch, gọi model một lần và trả:

~~~text
backward_loss                # duy nhất graph-bearing output
detached scalar statistics   # logs, counts
~~~

Không trả bridge/hidden hoặc unused differentiable loss roots ra ngoài DDP wrapper.
find_unused_parameters=True tiếp tục phân tích graph từ backward_loss.

### 5.2. Chuẩn hóa từng objective

Trước vòng microbatch, thống kê cả accumulation window:
- N: tổng gold target tokens thực trên mọi rank;
- M: tổng valid evidence units thực trên mọi rank;
- example_count thực.

All-reduce integer counts, không all-gather candidates hoặc embeddings.
Với W=world size:

~~~text
backward_loss =
    W * ce_sum_local / max(1,N)
  + lambda_step * W * (
        beta_copy * copy_sum_local + beta_semantic * semantic_sum_local
    ) / max(1,M)
~~~

both: beta_copy=beta_semantic=0.5.
copy-only: beta_copy=1, beta_semantic=0.
semantic-only: beta_copy=0, beta_semantic=1.

Ablation dùng cùng valid units M trong cả ba chế độ; không thay sample eligibility
theo objective rồi gọi là đối chứng cùng dữ liệu.

no_sync giữ nguyên cho các microbatch trước cuối window. Sau accumulation mới
clip, optimizer.step, scheduler.step. Rank không có data/evidence vẫn forward/backward
và collective đúng lượt.

Không lấy local mean rồi trung bình giữa ranks. Không scale evidence bằng N,
vì số evidence units không tỷ lệ cố định với số gold tokens.

### 5.3. Stage/ramp

- Interface epoch: CE-only.
- Full stage: lambda từ 0 lên lambda_max trong 10% optimizer updates đầu.
- Với full stage có S updates, R=max(1,ceil(0.1*S)):
  lambda(s)=lambda_max*min(s/R,1), s là số updates full đã hoàn thành trước step.
- Schedule/R/s được lưu để resume đúng, không restart ramp từ đầu.
- Tắt evidence do stage/lambda không đổi cách tính validation CE.
- NEFTune vẫn không tồn tại trong recipe mới.

### 5.4. Logs và checkpoint selection

Log CE, copy loss, semantic loss, lambda, N/M, evidence coverage/confidence,
candidate sizes, seconds, peak VRAM và total objective. Coverage không có evidence
không được in thành auxiliary loss “cải thiện”.

loss_ce sạch vẫn là criterion mặc định cho best.pt trong trainer; total objective
có ramp không dùng để chọn checkpoint. Cấu hình cuối được chốt bằng generated
validation ROUGE155; không đổi protocol decode đồng thời với đổi objective.

Logging gradient từng loss nếu debug có thể cần autograd.grad riêng: không bật
mặc định hoặc mô tả là miễn phí.

## 6. Config, checkpoint và shell

Contract cấu hình đã chạy:

~~~yaml
training:
  evidence_contrastive:
    enabled: true
    mode: both
    max_weight: 0.05
    ramp_ratio: 0.10
    cache_path: /path/to/pplx/evidence.jsonl
    mining:
      min_positive_score: 0.30
      max_negative_score: 0.15
      positive_band: 0.05
      min_context_gap: 0.20
      min_context_matches: 2
      max_units_per_example: 32
      max_positive_sentences: 4
      max_negative_sentences: 4

~~~

Không có auxiliary sampling temperature; dùng attention native. Input không có
block này tương đương enabled=false. Validate coefficient finite, thresholds
có miền hợp lệ, copy/semantic prerequisites, cache manifest match.

- training/checkpoint.py: lưu resolved objective spec, mining version, manifest
  content hash, weighting mode, ramp settings/progress trong training metadata.
- Không thêm các training-only settings vào architecture_spec vì state dict và
  inference graph không thay.
- Resume strict: chặn đổi objective, coefficient, mining/cache hay ramp. Chuyển
  CE checkpoint sang objective mới là warm-start run khác, không giả vờ exact resume.
  V1 chạy so sánh từ cùng pretrained initialization; chưa cần thêm warm-start API.
- Evaluation weights của recipe mới không cần cache/gold alignment. Checkpoint
  loading kiểm tra graph; training recipe guards chỉ nằm ở resume training.
- Cache manifest và cấu hình được lưu/copy vào output để lần sau tái lập.

Shell dự kiến:
- Thêm recipe evidence vào run_pubmed_pair.sh, một output directory riêng.
- Dùng cùng default batch48/GPU*2*accum1=96; một GPU dùng48*accum2.
- Cache path theo encoder; prepare-evidence chạy trước torchrun nếu thiếu cache,
  hoặc xác minh manifest nếu có. Cache khác hash không bị ghi đè tự động.
- Smoke có thêm recipe evidence với fixture thực sự có repeated-anchor contrasts.
- Thiết lập full recipe mới làm mặc định sau khi integration checks qua; vẫn cho
  AFMR_TRAINING_RECIPE=cosine chạy CE-only.

Commands đã có:

~~~bash
python scripts/prepare_evidence.py --config configs/afmr_pubmed.yaml \
  --split train --output-dir /path/to/evidence/pplx --audit-size 200

AFMR_TRAINING_RECIPE=evidence RUN_ENCODERS=pplx \
  bash scripts/run_pubmed_pair.sh
~~~

Script cần đặt model/tokenizer paths thật và prepare đúng effective config do
queue sinh ra. Không chuẩn bị theo placeholder path ở YAML rồi train backbone khác.

## 7. Thứ tự triển khai và tiêu chí hoàn thành

| Mốc | Công việc | Đủ điều kiện chuyển mốc khi |
|---|---|---|
| 1 | Types/config, shared offset helpers, mining/cache CLI | Đã xong: cache deterministic, hash/checksum/fingerprint và CLI |
| 2 | Collator, dataset, runtime và DDP placeholder metadata | Đã xong: sparse tensors, shifted-position checks, fake rank vô hiệu cả CE/evidence |
| 3 | Loss trong grounded_copy, decoder/model statistics | Đã xong: FP32 grouped logsumexp trên actual read keys, CE interface giữ nguyên |
| 4 | Wrapper/trainer/DDP/ramp/checkpoint/logging | Đã xong: global N/M, lambda ramp, recipe guard và Gloo placeholder test |
| 5 | Shell recipe, smoke, regression và docs | Đã xong ở fixtures/regression; full PubMed/GPU chưa chạy |
| 6 | Audit/minirun trên PubMed thật | Đo coverage/precision/throughput; có validation evidence để quyết định full run |
| 7 | Full run và ablations | Report ROUGE155 cùng protocol, đủ để đánh giá hypothesis |

Mốc 1-5 có thể làm offline với fixtures. Mốc 6-7 cần dữ liệu/model/GPU thực; không
gọi tiny smoke là bằng chứng ROUGE tăng.

Audit cache xuất:
- summary.json về coverage, score/confidence histogram, units bị lọc/cap;
- examples.jsonl chứa source context/gold/unit/nhãn/lý do, phục vụ kiểm tra;
- manifest với config/hash.

Nếu audit cho thấy nhiều nhãn sai hoặc coverage thấp, quay về mining trước full
run. Không yêu cầu người dùng phê duyệt từng bước implementation; đây là điều kiện
chất lượng của thí nghiệm.

## 8. Bộ kiểm tra cần viết

### Data/alignment

- Hai câu có cùng anchor nhưng khác context; thay câu gold phải đổi positive đúng.
- Bỏ anchor khỏi scoring; overlap chỉ có anchor không được tạo confident label.
- Nhiều positive tương đương, ties, zero context, no negative.
- Số thập phân, phần trăm, drug-name punctuation, casing và whitespace token boundary.
- Encoder/decoder tokenizer khác nhau; không dùng ID equality giữa hai tokenizer.
- Unit bị cắt ở source/target; prompt/EOS/padding không có evidence.
- Sidecar content/tokenizer/prefix/length mismatch fail rõ.
- prepare và collator tạo giống token IDs; duplicate record IDs không ghép nhầm sidecar.
- Với distributed placeholder: CE count=0 và evidence_count=0.

### Loss/gradient

- Fixture copy marginal bằng nhau nhưng occurrence loss khác, giống phân tích.
- Positive gradient tăng positive logit, giảm competing negative; unrelated logit
  độc lập không có gradient từ conditional loss.
- Multiple positives, masks rời nhau, candidate permutation invariant.
- Multi-subtoken unit qua nhiều chunk giữ nguyên normalization.
- Dense và chunked loss/gradients khớp FP32/BF16.
- Không có eligible evidence: finite graph-connected zero, CE không đổi.
- lambda0/mode off có logits, CE, gradient parity với baseline.
- Một source-backbone forward và một decoder-backbone forward mỗi microbatch;
  activation checkpoint replay khi backward được đếm riêng.
- Auxiliary trực tiếp tác động query/key, CE vẫn học semantic values/output/gate;
  không giả định attention-only loss phải có gradient trực tiếp tới mọi module.
- Mock generation APIs để train fail ngay nếu lỡ gọi generate.

### DDP/resume

- Hai CPU/Gloo processes thật so gradient và weight update với serial global batch.
- Target-token counts khác evidence-unit counts; counts không đều giữa ranks.
- Một rank không có evidence, rank placeholder, cả window không có evidence.
- Accumulation dư, unit qua chunk boundary, both/copy/semantic mode.
- Checkpoint giữa ramp và sau ramp; reload chính xác lambda/scheduler/RNG/optimizer.
- Resume objective/cache đổi bị chặn; inference load không yêu cầu cache.

### End-to-end

- Fixture train phải có ambiguous occurrences thật, không chỉ từ alpha/beta không
  tạo contrast như fixture CE trước đó.
- Save/load, greedy eval và sampled API không phụ thuộc evidence metadata/reference.
- Shell generator chọn đúng encoder-specific cache, global batch96, recipe/output.
- All existing regression tests, bash syntax và git diff --check.
- CUDA benchmark sau đó kiểm tra VRAM tăng theo chunks và real training throughput.

## 9. Những điều giữ lại để bảo vệ contribution

Implementation phải cho phép các ablation đã nêu trong design:
- cùng labels: generic span supervision vs conditional same-token loss;
- copy-only / semantic-only / both;
- same-token competitors vs random hoặc different-token competitors cùng budget;
- CE baseline và SDACL-style baseline được mô tả đúng, không gọi decoder-pooling
  adaptation là SDACL exact.

Bản đầu tích hợp objective chính và copy/semantic/both switch. Các đối chứng khác
được thêm sau khi pipeline chính đúng; không mở rộng thành nhiều loss mặc định.

Đầu ra đầu tiên cần đạt là: **một recipe chạy được đúng trên 1/2 GPU, không sinh
training text, có provenance và ablation controls**, rồi mới đo khả năng tăng ROUGE.
