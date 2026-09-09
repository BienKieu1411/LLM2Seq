# Luna review độc lập: EviSeq update v3 cuối

Ngày review: 2026-09-09.

> Review này giữ nguyên snapshot trước sửa. Phần triển khai đáp lại các finding,
> config mặc định mới và kiểm chứng nằm trong
> [EVISEQ_UPDATE_V3_REVIEW_REVISION.md](EVISEQ_UPDATE_V3_REVIEW_REVISION.md).

Snapshot: `HEAD=595f511c0f45e6d406949e1a3f419298d845b142`.

Phạm vi code: `src/eviseq_update_v3` và phần tương ứng của `src/eviseq_update`.

Phạm vi thay đổi: chỉ tạo báo cáo này; không sửa runtime, config hoặc test.

Luna thực hiện review độc lập; agent chính đối chiếu kết luận, bổ sung các
probe và hiệu đính độ chính xác của bản báo cáo cuối.

## Kết luận trước

V3 hiện là một graph có ý đồ rõ và đường tính toán chính nhất quán: AFMR tạo
memory nguồn đầy đủ, Qwen causal decoder đọc memory ở mỗi layer, grounded copy
trộn xác suất copy đã cộng gộp theo token ID với phân phối LM, còn nhánh
semantic hierarchical đọc source bằng bốn head có tổng rank 512. Planner dùng
chỉ prefix summary đã quan sát; nó không nhận label tương lai. Fusion
norm-preserving giữ norm của hidden đi vào vocabulary head. Với cùng hidden và
copy state trong một forward, P_copy vẫn được tính trước fusion; sau khi các
tham số shared được fine-tune thì không có bảo đảm P_copy hay R2 giữ nguyên.

Trong các probe đã thực hiện, chưa thấy bug làm sai gradient trên đường train
chuẩn, label leakage, sai parity giữa dense và KV-cache, hoặc sai reset khi
compact batch. Root đã chạy toàn bộ suite hiện tại với `286 passed` trong
`57.99s`; probe bổ sung trên graph tiny cũng cho cached-vs-dense
`max|delta logit|=4.768e-7`, prefix không phụ thuộc token tương lai, và
history coverage reset đúng. Những kết quả đó chứng minh các invariant được
kiểm tra, không chứng minh ROUGE tăng.

Có ba finding cần tách khỏi kết luận kiến trúc:

1. **Bug đánh giá có thể làm sai claim ROUGE — mức nghiêm trọng cao về tính
   hợp lệ thực nghiệm.** `evaluate()` có thể dùng nguyên file predictions cũ
   và trả metrics trước khi đọc hoặc kiểm tra checkpoint. Root đã tái hiện với
   checkpoint không tồn tại: hàm vẫn thành công và giữ hai prediction cũ.
2. **Bug API cross-cache — mức thấp trong runner chuẩn, nhưng tái hiện được.**
   Decoder eval dùng `_cache` đã chuẩn bị dù caller truyền memory mới và không
   yêu cầu dùng cache. Đây là hành vi kế thừa từ v2. `_generate()` chuẩn có
   `prepare` và `finally clear`, nên finding này không giải thích điểm ROUGE
   của run chuẩn.
3. **Bug decoding API với non-chat prompt không đều — mức thấp.** Khi
   `pad_token_id == eos_token_id`, non-chat path đưa cả left-padding vào
   repetition/no-repeat history. Điều này có thể cấm EOS hoặc phạt token chỉ
   vì padding; prompt PubMed hiện được cố định nên không tác động recipe hiện
   tại.

Về thiết kế, hard partition có giới hạn định lượng: khi số vùng thấp hơn số
head, một số head không có vùng bị zero; khi số vùng không vượt quá số head,
bias coverage/recent không thể đổi lựa chọn region của một head có đúng một
region. Root chỉ đo trực tiếp trường hợp source128 tokens/2 regions; phần còn
lại là hệ quả của mask singleton. Planner cũng chỉ là learned usage tracker,
chưa theo dõi fact, câu hoặc thứ tự discourse một cách tường minh. Vì vậy v3
là một bước kiến trúc hợp lý để thử nghiệm, nhưng chưa đủ bằng chứng để claim
“vượt v2”, “vượt T5Gemma”, tăng ROUGE ổn định, hoặc giảm hallucination.

## 1. Cách đọc graph hiện tại

### 1.1. Data flow end-to-end

Đường train và inference có dạng:

```text
encoder final/taps
        │
        ├─ AFMR refined memory H_key + source prior b
        │              │
        │              └─ Qwen copied cross-attention ở mọi decoder layer
        │                              │
        │                              └─ causal hidden h_t
        │
        └─ final-state value anchor H_value
                       │
                       ├─ grounded copy keys + values + token alignment
                       ├─ semantic source K/V, rồi region cache
                       └─ prefix planner (train scan hoặc inference history)

h_t ── copy query ── P_copy
 │
 ├── semantic query → region/head read → tangent bounded fusion → h'_t → P_LM
 │
 └── copy gate

P_mix = g_copy P_copy + (1-g_copy) P_LM
loss = gold-token CE(P_mix)
```

`EviSeqAFMR.encode_source()` ở
[`model.py:34-71`](../src/eviseq_update_v3/eviseq_update_v3/modeling/model.py:34)
chạy encoder, lấy prompt embedding, chạy bridge, ép memory/value memory về
dtype của decoder, rồi tạo `CopyState`. Khi `architecture.name` là
`afmr_value_anchor`, `bridge.value_memory` là `base_projection(final)` ở
[`afmr.py:239-243`](../src/eviseq_update_v3/eviseq_update_v3/modeling/afmr.py:239).
Memory refined dùng cho key và source prior; value anchor dùng state cuối chưa
qua các depth/feature residual. Copy semantic values cũng lấy value anchor qua
[`model.py:60-61`](../src/eviseq_update_v3/eviseq_update_v3/modeling/model.py:60).

### 1.2. AFMR/value-anchor và cross decoder

`AdaptiveFullMemoryResidualBridge` ở
[`afmr.py:19-244`](../src/eviseq_update_v3/eviseq_update_v3/modeling/afmr.py:19)
không rút source xuống một vector. Nó giữ mọi source position, trộn depth taps
bằng router, thêm low-rank feature residual có gate giới hạn, tạo focus prior
đa scale, rồi trả:

- `memory`: key-side refined representation;
- `value_memory`: final-state anchor nếu bật;
- `source_bias`: focus prior trên content token;
- `content_mask` và `attention_mask`.

Trong cross attention, v3 sao chép `q_proj/k_proj/v_proj/o_proj`, q/k norm và
input norm của self-attention Qwen tại
[`decoder.py:58-104`](../src/eviseq_update_v3/eviseq_update_v3/modeling/decoder.py:58).
Mỗi `DecoderLayerWithCross` chạy self-attention trước, cộng cross residual
qua scalar gate rồi chạy post-attention norm và MLP ở
[`decoder.py:217-241`](../src/eviseq_update_v3/eviseq_update_v3/modeling/decoder.py:217).
`cross_attention_every=1` bị bắt buộc; `query_cross_gate=false` trong recipe
cuối nên gate theo query/head của v3 ban đầu không nằm trong run chính.

Điểm quan trọng cho diễn giải: value anchor không làm cross attention chỉ đọc
final state. Cross key vẫn lấy từ refined `memory`, còn cross value lấy từ
`value_memory`; source prior đi vào mask bias. Nhánh copy/semantic đọc anchor,
nhưng hidden Qwen vẫn bị ảnh hưởng bởi AFMR refined key, source bias và cross
residual. Đây là thiết kế giữ nhiều tín hiệu, không phải một “copy path bất
biến” sau khi toàn bộ model được fine-tune.

### 1.3. Grounded copy và marginalized CE

`CopyState` và `PlannedCopyState` ở
[`grounded_copy.py:14-41`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:14)
chứa source keys, token IDs, mask, bias, semantic cache và planner region
cache. `prepare()` ở
[`grounded_copy.py:122-215`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:122)
thực hiện các bước:

1. RMS-normalize memory và tạo contextual copy key.
2. Pool encoder positions về decoder-token alignment bằng overlap weights.
3. Cộng lexical key từ embedding của token ID.
4. Giữ token IDs trùng nhau trong cùng vocabulary; mask các alignment không có
   content hoặc không có tổng weight.
5. Nếu semantic read bật, tạo semantic value trên native source positions và
   semantic key riêng; với hierarchical mode, mean-pool key thành region.

Alignment ở
[`data/copy_alignment.py:16-61`](../src/eviseq_update_v3/eviseq_update_v3/data/copy_alignment.py:16)
chỉ dùng source text, encoder offsets, decoder tokenizer và overlap ký tự. Nó
không đọc target/reference. Token cắt ở biên truncation bị bỏ nếu không được
phủ trọn phần non-space; đây là lựa chọn an toàn cho copy, nhưng có thể làm
giảm số cơ hội copy ở biên source.

`_attention()` ở
[`grounded_copy.py:217-229`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:217)
tạo `log_attention`, `log_copy`, `log_generate`. `_mix_logits()` ở
[`grounded_copy.py:378-387`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:378)
scatter-add attention mass về vocabulary token ID rồi tạo:

```text
P_mix(v) = g_copy P_copy(v) + (1 - g_copy) softmax(LM_logits)(v)
```

Hàm trả `log P_mix + logsumexp(LM_logits)` để softmax của output vẫn là
`P_mix`; đây là cách biểu diễn logits hợp lệ. Trong chunked loss, target copy
được tính bằng `logsumexp` trên mọi source position có cùng target token ID ở
[`grounded_copy.py:389-400`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:389).
Vì vậy duplicate source occurrences được marginalize, thay vì chọn một
alignment tùy ý. Target không copy được tự động rơi về nhánh LM thông qua
`logaddexp`.

Đây là điểm mạnh giữ lại từ v2. Tuy nhiên “giữ lợi thế R2” chỉ đúng có điều
kiện: semantic branch đọc hidden trước fusion để lấy copy distribution, nhưng
encoder, decoder, bridge, source prior và copy gate đều nhận gradient từ CE.
Sau training, hidden và `P_copy` có thể đã đổi. Test
`test_semantic_changes_leave_copy_distribution_exact_for_fixed_base_hidden`
chỉ chứng minh khi giữ cố định hidden/state và sửa riêng semantic parameters;
nó không chứng minh P_copy của hai checkpoint sau fine-tune giống nhau.

### 1.4. 4×128 semantic attention và hard partition

Recipe cuối đặt `rank=512`, `num_heads=4`, nên mỗi head có width 128. Các
projection Q/K/V đều rộng 512 rồi reshape; key được RMS-normalize theo từng
head ở [`grounded_copy.py:166-178`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:166).
Đây là **4×128**, không phải 4×32. So với v2 1×128, mỗi head có cùng chiều
matching 128, còn tổng capacity và cache tăng.

Hierarchical mode tạo region 64 content positions. Region key là mean token
key rồi per-head RMSNorm. `_hierarchical_context()` ở
[`grounded_copy.py:260-300`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:260)
chọn region trước, rồi normalize token attention trong từng region. Khi
`partition_heads=true`, owner là:

```text
owner(r) = r mod 4
```

Mỗi region vẫn có một head sở hữu; các head không đọc vùng của nhau. Đây là
phân vùng theo vị trí, không phải phân đoạn discourse hay clustering semantic.
Một câu có thể bị cắt giữa hai vùng; hai vùng có thể chứa cùng fact; head khác
nhau vẫn có thể học hành vi gần giống nhau trên các vùng được giao. Learned
head gate ở [`grounded_copy.py:343-350`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:343)
có thể suppress head, nhưng không tạo diversity bắt buộc.

### 1.5. Prefix planner, coverage và continuity

`CoveragePlanner.forward()` ở
[`semantic_plan.py:66-90`](../src/eviseq_update_v3/eviseq_update_v3/modeling/semantic_plan.py:66)
nhận hidden decoder và `summary_input_mask`. Nó tính semantic region
probability, usage mass, cumulative coverage và region gần nhất của summary
prefix. Bias ở [`semantic_plan.py:92-98`](../src/eviseq_update_v3/eviseq_update_v3/modeling/semantic_plan.py:92)
giảm score vùng đã có coverage và tăng score vùng recent khi các cờ được bật.

Trong train, `QwenCrossDecoder.forward()` tạo
`summary_mask=current_valid & positions.ge(prompt_lengths)` ở
[`decoder.py:354-375`](../src/eviseq_update_v3/eviseq_update_v3/modeling/decoder.py:354).
Planner chạy trên toàn sequence một lần, dùng cumulative sum; chunk CE chỉ lấy
slice của `ReadPlan` ở [`grounded_copy.py:389-415`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:389).
Trong generation, history được giữ riêng và được select cùng self/cross cache
khi compact batch.

Điều tracker thực sự đo là **mô hình usage của attention trên các region**.
Nó không biết target token kế tiếp, không xác minh một fact đã được diễn đạt
đúng, không biết ranh giới câu, và không lập danh sách objective/method/result/
conclusion. Tên “coverage” nên được dùng theo nghĩa learned attention usage,
không nên viết thành factual coverage trong paper nếu chưa có metric kiểm chứng.

### 1.6. Tangent/norm-preserving fusion

`_fuse()` ở
[`grounded_copy.py:302-320`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:302)
đầu tiên bỏ thành phần radial của delta, giới hạn RMS tương đối `rho=0.10`,
rồi rescale `h+delta` về norm của `h`. Với `h != 0` và số học chính xác,
norm output bằng norm input; với zero hidden, code fallback về base.

Điều được giữ là magnitude của hidden đi vào LM head. Điều thay đổi là hướng
hidden, vì vậy logits và xác suất vẫn có thể đổi. Root probe cũng cho thấy
giảm đều các gain head từ 1 xuống 0.1 không gây suppression đồng loạt sau
fusion: tỉ lệ norm correction quan sát được vẫn xấp xỉ `0.998516` do RMSNorm
context sau head gate. Đây là hạn chế của vị trí head gate, không phải lỗi của
cap: cap vẫn giới hạn delta, còn head gate chủ yếu thay đổi đóng góp tương đối
giữa các head. Phần LUNA-DESIGN-005 phân tích riêng điểm này.

## 2. Findings phân loại riêng

### 2.1. Bug tái hiện được

#### LUNA-EVAL-001 — evaluation có thể trả ROUGE từ prediction cũ

**Mức độ:** ảnh hưởng trực tiếp tính hợp lệ thực nghiệm khi reuse file predictions
khác checkpoint/config. Đây là lỗi eval; cần xác minh provenance trước khi dùng
score cho claim, không phải bằng chứng mọi kết quả đã báo đều sai.

**Vị trí:**

- [`runtime.py:320-339`](../src/eviseq_update_v3/eviseq_update_v3/runtime.py:320)
  đọc output JSONL và chỉ kiểm ID/reference prefix.
- [`runtime.py:349-364`](../src/eviseq_update_v3/eviseq_update_v3/runtime.py:349)
  trả metrics ngay nếu file đã đủ dòng.
- Việc load checkpoint chỉ xảy ra sau đó ở
  [`runtime.py:365-372`](../src/eviseq_update_v3/eviseq_update_v3/runtime.py:365).

**Trigger tối thiểu:**

1. Tạo predictions JSONL đủ toàn bộ example của split, với ID/reference đúng
   nhưng prediction tùy ý.
2. Gọi `evaluate(config, checkpoint_path_mới, output_path_cũ)`, trong đó
   `checkpoint_path_mới` không tồn tại hoặc là checkpoint epoch khác.
3. Vì `resumed_count == total`, hàm tính metrics trên rows cũ rồi return; model
   không được khởi tạo và checkpoint không được đọc.

Root đã chạy đúng probe này trong `TemporaryDirectory`: split test có 2 dòng,
`prediction='deliberately old summary'`, checkpoint path không tồn tại. Kết quả
là `requested_checkpoint_exists=false`, `evaluation_returned_success=true`,
`reused_predictions=2`, `output_unchanged=true`.

Với file partial, code cũng có thể nối prediction từ checkpoint mới vào prefix
cũ. Guard ID/reference không phát hiện khác checkpoint, generation flags,
tokenizer, preprocessing hay config. Do đó một lệnh eval epoch 3 sau khi
epoch 4 đã sinh cùng output path có thể báo điểm epoch 4, hoặc một score hỗn
hợp.

**Tác động:** Có thể làm sai hoàn toàn so sánh v2/v3/T5Gemma và chọn nhầm
checkpoint. Đây là rủi ro lớn hơn mọi chênh lệch R2 vài phần trăm điểm.

**Cách sửa ưu tiên:** Trước khi đọc/reuse output, tạo manifest immutable chứa
ít nhất:

- hash hoặc fingerprint checkpoint, epoch/step và `architecture_spec`;
- hash `resolved_config.yaml` sau khi resolve path;
- fingerprint train/eval JSONL, split, tokenizer name/version/vocab/special IDs;
- generation config, max/min tokens, repetition/no-repeat, detokenization và
  ROUGE backend.

So sánh manifest rồi mới cho phép reuse. File complete vẫn phải xác minh
manifest; file partial phải reject khi mismatch. Nếu chưa thêm manifest, dùng
output path riêng cho từng checkpoint và xóa output cũ trước eval. Chỉ kiểm
`Path(checkpoint).exists()` là chưa đủ vì checkpoint khác có thể cùng tồn tại.

#### LUNA-MODEL-001 — cross-cache dùng stale source khi caller đổi memory

**Mức độ:** thấp trong runner chuẩn, trung bình đối với API public/inference
service; finding tái hiện được và kế thừa v2.

**Vị trí:**

- `_cache` là plain attribute tại
  [`decoder.py:80-92`](../src/eviseq_update_v3/eviseq_update_v3/modeling/decoder.py:80).
- `forward()` chọn `_cache` chỉ dựa trên cache khác `None` và `not self.training`
  tại [`decoder.py:114-120`](../src/eviseq_update_v3/eviseq_update_v3/modeling/decoder.py:114).
- API chuẩn chuẩn bị cache ở [`decoder.py:430-435`](../src/eviseq_update_v3/eviseq_update_v3/modeling/decoder.py:430)
  và xóa ở [`decoder.py:437-441`](../src/eviseq_update_v3/eviseq_update_v3/modeling/decoder.py:437).

**Trigger tối thiểu:**

```text
decoder.eval()
prepare_cross_cache(memory_A)
forward(query, memory_B, ..., use_cache=False)
```

`CopiedCrossAttention.forward()` vẫn dùng K/V đã tạo từ `memory_A`. Probe tiny
với cùng query cho thấy output sau bước trên khớp output fresh của A, còn khác
output fresh của B với `max|delta|=0.0322478637`.

**Tác động:** Caller có thể đọc nhầm source mà không có exception. Đây không
phải bug của `generate._generate()` chuẩn: hàm này gọi prepare một lần tại
[`generate.py:140-143`](../src/eviseq_update_v3/eviseq_update_v3/evaluation/generate.py:140)
và luôn clear tại [`generate.py:200-201`](../src/eviseq_update_v3/eviseq_update_v3/evaluation/generate.py:200).
Root đã kiểm cached-vs-dense bằng API decoder tiny theo vòng đời cache chuẩn
trên CPU và đạt sai số `4.768e-7`;
không có cơ sở gán regression ROUGE cho finding này.

**Cách sửa:** Tách rõ cờ `use_cross_cache` khỏi `use_cache` của self-attention,
hoặc chỉ dùng `_cache` khi caller truyền cờ explicit. Lựa chọn an toàn hơn là
reject nếu cache đã tồn tại nhưng memory/value-memory mới không có fingerprint
giống cache; `prepare_cross_cache()` phải thay cache atomically. Thêm regression
test A→B như probe trên và test clear/reprepare.

#### LUNA-GEN-001 — non-chat no-repeat/repetition tính cả left padding

**Mức độ:** thấp; ảnh hưởng API generation có prompt độ dài khác nhau, không
ảnh hưởng PubMed recipe hiện tại vì prompt collator cố định.

**Vị trí:**

- Prompt được left-pack và mask được giữ ở
  [`generate.py:127-132`](../src/eviseq_update_v3/eviseq_update_v3/evaluation/generate.py:127).
- Non-chat path đặt `constraint_history=history` tại
  [`generate.py:158-161`](../src/eviseq_update_v3/eviseq_update_v3/evaluation/generate.py:158).
  Chỉ chat path mới bỏ trọn `width` prompt.
- Các helper phạt mọi ID trong history ở
  [`generate.py:15-32`](../src/eviseq_update_v3/eviseq_update_v3/evaluation/generate.py:15).

**Trigger tối thiểu:** decoder có `pad_token_id=eos_token_id=2`, prompt rows
không đều, non-chat. Sau reorder, row ngắn có ID `2` ở vị trí padding. Probe
helper với `history=[[2,2,5]]` cho no-repeat ngram size 1 danh sách banned
`[2,5]`; history hợp lệ chỉ `[[5]]` banned `[5]`. EOS bị cấm do padding.

**Tác động:** Có thể phạt EOS, thay đổi token greedy hoặc sampling, và làm
no-repeat ngram phụ thuộc vào độ dài prompt của row khác trong batch. Với
PubMed, `_prompt_ids` lấy cùng instruction cho mọi record nên không tạo padding
prompt; không dùng finding này để giải thích score hiện tại.

**Cách sửa:** Xây constraint history từ token có `decode_mask=True`; hoặc giữ
riêng `prompt_ids` hợp lệ và generated IDs thay vì truyền toàn `history`. Thêm
test non-chat unequal prompts với pad=eos, kiểm EOS không bị ban khi chỉ xuất
hiện ở masked position.

### 2.2. Hạn chế thiết kế đã quan sát

#### LUNA-DESIGN-001 — hard partition làm mất head ở tài liệu ngắn

**Vị trí:** [`grounded_copy.py:260-300`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:260),
đặc biệt [`grounded_copy.py:272-280`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:272).

Region owner là `region % heads`; nếu source có `R < H`, một số head không có
region hợp lệ nên masked softmax trả context zero cho head đó. Root probe với
source 128 tokens, 2 regions và 4 heads xác nhận chỉ head 0/1 có một region;
head 2/3 inactive. Đây không phải numerical bug vì output đúng theo mask.

Root thay coverage/recent rất lớn và quan sát `context maxdiff=0.0` trong
probe 128 content tokens, tức R=2. Đây là measurement trực tiếp;
với R=3 hoặc R=4, việc planner không đổi được region probability của các head
đang active là suy ra từ mask singleton, chưa phải một phép đo riêng cho từng
R. Với PubMed4096 và region64 thường có nhiều vùng hơn, nhưng phân phối source
sau truncation phải đo thật.

**Tác động:** Capacity 4×128 không được sử dụng đồng đều trên source ngắn;
continuity/coverage có thể không có effect. Claim “bốn head phủ tài liệu” chỉ
đúng theo hợp region, không đúng theo hiệu quả mỗi head.

**Ablation/sửa:** So sánh `PARTITION_HEADS=false`; tự động tắt partition khi
`R <= H` để mọi head còn có lựa chọn region (khi `R=1` thì không có lựa chọn
vùng nào để planner thay đổi, đó là hệ quả tự nhiên). Đo số active head theo
source-length bucket. Giữ region64/32/128 và báo coverage effect, R1/R2/L từng
bucket. Không đổi code trong turn review này.

#### LUNA-DESIGN-002 — planner không phải factual/discourse coverage

**Vị trí:** [`semantic_plan.py:66-98`](../src/eviseq_update_v3/eviseq_update_v3/modeling/semantic_plan.py:66).

Planner ước lượng usage từ `h_t` và region key. `coverage` là tổng mass
attention, còn `recent` là probability của region summary gần nhất. Không có
supervision cho “fact đã nói”, không có sentence boundary, không có relation
giữa claim và evidence. Coverage penalty có thể phạt một vùng vì model đọc nó
nhiều dù chưa phát biểu fact; continuity bonus có thể giữ vùng recent dù cần
nhảy tới result ở xa.

**Tác động:** Cơ chế có thể giảm lặp attention, nhưng chưa bảo đảm giảm lặp
ý, đủ objective/method/result/conclusion hay giảm hallucination. Cần gọi đúng
là prefix-conditioned learned usage trong report/paper.

**Ablation:** Báo các cặp `USE_COVERAGE`/`USE_CONTINUITY`, planner off,
partition off, và metric factuality/novelty riêng. Không suy ra factual
coverage từ giá trị `coverage` trong checkpoint.

#### LUNA-DESIGN-003 — 4 head không tự tạo semantic diversity

**Vị trí:** semantic projection/cache ở
[`grounded_copy.py:166-180`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:166),
head gate ở [`grounded_copy.py:343-350`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:343).

4 head có support khác nhau chỉ vì partition vị trí; chúng không bị buộc đọc
fact khác nhau hoặc có orthogonal query/key. Region mean có thể làm mất tín
hiệu token hiếm; source có fact lặp lại ở nhiều region vẫn cho duplicate
semantic content. Gate có thể suppress head nhưng không phải diversity loss.

**Tác động:** Tăng rank từ128 lên512 tăng capacity và compute, nhưng chưa có
bằng chứng phần tăng đó là bốn góc nhìn bổ sung. Có thể tăng overfit hoặc
chi phí mà R1/L không tăng.

**Ablation:** 1×128, 4×32, 4×128 free-head (`partition=false`), 4×128
partition, và head-gate off; đo attention overlap, active-head rate, và
ROUGE theo source length. Đây là kiểm tra capacity, không nên gọi là proof
novelty.

#### LUNA-DESIGN-004 — norm-preserving không bảo đảm copy/R2 sau fine-tune

**Vị trí:** [`grounded_copy.py:302-320`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:302),
semantic read và LM ở [`grounded_copy.py:322-370`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:322).

Fusion loại radial delta và giữ norm, nhưng vocabulary head phụ thuộc hướng
hidden. Copy distribution được tính từ `hidden` trước fusion trong một forward,
đây là isolation đúng cho fixed hidden; song toàn bộ shared graph vẫn cập nhật
qua CE. AFMR refined memory, source prior, Qwen cross output, copy query/key và
copy gate đều có thể thay đổi giữa checkpoint v2 và v3.

**Tác động:** Không thể viết “semantic branch không ảnh hưởng P_copy” hoặc
“R2 được bảo toàn” cho hai run đã fine-tune. Chỉ có thể viết: ở cùng hidden và
copy state, semantic parameters riêng không đổi P_copy; fusion giới hạn độ lớn
thay đổi hidden theo norm. Đây là giới hạn của bảo đảm sau training, không phải
khẳng định fusion trực tiếp sửa copy distribution trong cùng forward.

**Ablation:** Ghi `P_copy` trước/sau training; chạy semantic-only với copy/AFMR/
cross frozen như một chẩn đoán; so sánh fusion residual và norm-preserving; báo
R2 và copyable token recall. Freeze làm thay đổi protocol và chỉ dùng để tách
đường tác động, không phải điều kiện bắt buộc để train v3 hay để tuyên bố giữ
R2.

#### LUNA-DESIGN-005 — head gain bị RMSNorm triệt tiêu khi scale đồng loạt

**Vị trí:** head gain được áp dụng trước context RMSNorm tại
[`grounded_copy.py:343-359`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:343).

**Trigger/đo lường:** Root giữ global `semantic_gate` cố định, lần lượt scale
đồng đều bốn head gain từ 1 xuống 0.1. Context sau head gate được normalize
chung tại dòng 353, vì vậy scale chung bị triệt tiêu. Probe quan sát tỉ lệ norm
correction sau fusion giữa hai mức scale bằng khoảng `0.998516`. Kết quả này là
measurement trên cấu hình probe. RMSNorm gần bất biến với scale chung khi RMS
context lớn so với epsilon của normalization; tính bất biến không tuyệt đối
khi context gần zero hoặc do làm tròn floating point.

**Tác động:** `semantic_head_gate` chủ yếu điều chỉnh hướng/tỉ lệ tương đối
giữa các head; nó không cung cấp núm điều chỉnh biên độ semantic toàn cục.
Global `semantic_gate` mới điều chỉnh amplitude sau normalized context. Đây là
hạn chế của vị trí gate trong graph, không phải lỗi của cap `rho` hay định
nghĩa bounded fusion. Nếu head bị coi là “suppression” để bảo vệ R2, diễn giải
đó có thể sai khi tất cả gain cùng giảm.

**Ablation:** So sánh gate trước context normalization (hiện tại) với gate sau
normalization hoặc normalize từng head trước khi ghép, trong khi vẫn giữ
global `semantic_gate`. Ghi norm/context angle, global gate và R1/R2/L để biết
head gate có thật sự giảm semantic amplitude hay chỉ đổi hướng. Không coi
ablation này là thay đổi bắt buộc cho v3 trước khi có số liệu.

### 2.3. Rủi ro chưa kiểm chứng đầy đủ

#### LUNA-RISK-001 — resume không kiểm tra training/data provenance

Checkpoint lưu `training_spec` tại
[`training/checkpoint.py:123-147`](../src/eviseq_update_v3/eviseq_update_v3/training/checkpoint.py:123),
nhưng `load_checkpoint()` chỉ so `architecture_spec` ở
[`training/checkpoint.py:159-177`](../src/eviseq_update_v3/eviseq_update_v3/training/checkpoint.py:159).
`AFMRTrainer.fit()` gọi load ở
[`training/engine.py:291-335`](../src/eviseq_update_v3/eviseq_update_v3/training/engine.py:291)
nhưng không reject khi khác world size, batch, accumulation, effective batch,
LR/scheduler, precision, dataset hoặc tokenizer.

Nếu người dùng cố ý đổi protocol thì đây là một lựa chọn resume, không phải
runtime crash. Nhưng nếu tưởng là tiếp tục đúng quỹ đạo, optimizer state/RNG
được nạp trong khi số step mỗi epoch, sampler grouping và gradient weighting
đã đổi. Với DDP, checkpoint ghi world size nhưng không dùng field đó để chặn.

**Khuyến nghị:** validate exact training/data manifest khi resume; nếu muốn
đổi protocol, bắt buộc cờ explicit và ghi run lineage. Bổ sung attention
dropout, model/tokenizer identity và các giá trị initialization vào provenance
manifest dù weights đã lưu, để biết config thực sự của phần tiếp theo.

#### LUNA-RISK-002 — cache plain attribute và device ownership

`CopiedCrossAttention._cache` ở
[`decoder.py:80-92`](../src/eviseq_update_v3/eviseq_update_v3/modeling/decoder.py:80)
không phải parameter/buffer. Nếu caller prepare cache rồi mới `.to(device)` hoặc
đổi device/dtype, K/V cache không tự được chuyển. Root chưa coi đây là bug của
runner chuẩn vì `_generate()` prepare sau model đã ở device; đây là API misuse
chưa có guard.

**Khuyến nghị:** clear cache trong device transition, lưu cache owner/device
metadata, hoặc thiết kế context manager để cache không sống qua `.to()`.
Thêm probe prepare trên CPU rồi chuyển model nếu API tiếp tục public.

#### LUNA-RISK-003 — CUDA/real Qwen/peak memory chưa được benchmark trong review

Suite dùng `__tiny__`, CPU và CPU BF16 autocast cho nhiều invariant. Root có
probe native backbone BF16, bridge/planner FP32 không autocast trên CPU; parity
đạt. Chưa có peak VRAM/throughput với Qwen3-0.6B thật, encoder PPLX thật,
`sdpa`, NCCL và source distribution PubMed. Vì vậy mọi khẳng định tốc độ hoặc
“chạy chắc batch64 eval” còn là giả thuyết.

#### LUNA-RISK-004 — thiếu kiểm chứng sampling trên CUDA thật

`generate_sampled()` truyền `generator` trực tiếp vào
`torch.multinomial()` ở [`evaluation/generate.py:75-90`](../src/eviseq_update_v3/eviseq_update_v3/evaluation/generate.py:75).
README đã hướng dẫn tạo `torch.Generator(device=next(model.parameters()).device)`.
Truyền generator sai device gây lỗi ở primitive sampling, không phải một đường
âm thầm sinh sai phân phối đã được xác nhận. Điểm còn thiếu là kiểm chứng API
được hướng dẫn trên CUDA thật; test CPU hiện tại không thay thế được phép này.
Sampling vẫn chỉ nằm ngoài training và benchmark greedy.

#### LUNA-RISK-005 — checkpoint spec chưa bao phủ mọi yếu tố hành vi

`architecture_spec()` ghi graph version, rank/head/planner/fusion ở
[`training/checkpoint.py:18-81`](../src/eviseq_update_v3/eviseq_update_v3/training/checkpoint.py:18),
đủ để từ chối v2 copy-only hoặc rank khác trong test. Tuy nhiên model path,
tokenizer, attention dropout, semantic/cross gate initialization và data
preprocessing không nằm trong spec. Shape mismatch thường khiến model khác bị
reject khi load weights, nhưng hai model cùng shape vẫn có thể bị trộn provenance.
Đây là lý do cần manifest riêng, không coi `architecture_spec` là toàn bộ
checkpoint identity.

## 3. Các invariant đã kiểm chứng và kết quả thực sự có ý nghĩa

### 3.1. Causality và không leakage label/reference

- Encoder copy alignment chỉ nhận source text/offset/tokenizer tại
  [`data/copy_alignment.py:16-61`](../src/eviseq_update_v3/eviseq_update_v3/data/copy_alignment.py:16).
- `model.forward()` truyền decoder input/attention mask và labels riêng tại
  [`model.py:73-113`](../src/eviseq_update_v3/eviseq_update_v3/modeling/model.py:73).
- Planner tạo mask từ current decoder positions và prompt length; không dùng
  `labels` ở [`decoder.py:354-373`](../src/eviseq_update_v3/eviseq_update_v3/modeling/decoder.py:354).
- Qwen causal self-attention nhận hidden prefix; label chỉ vào CE sau hidden.

Root thay hai future input tokens trong probe inference và thấy logits prefix
không đổi (`maxdiff=0`). Prefix scan và incremental history cũng khớp trong
`test_prefix_scan_matches_incremental_history_and_excludes_prompt_padding_future`.
Đây là bằng chứng tốt rằng planner không nhìn tương lai trong đường đã kiểm
tra. Nó không chứng minh mọi custom remote Qwen implementation ngoài
`__tiny__` giữ đúng mask; cần smoke với model thật.

### 3.2. Dense/train và KV-cache inference parity

V3 giữ static source K/V cache tách khỏi self-attention past. Generation:

1. encode source một lần;
2. `prepare_cross_cache()` một lần;
3. prefill prompt;
4. mỗi bước dùng self past và semantic prefix history;
5. compact các row đã EOS, select past/cross/history/source/copy state cùng
   indices;
6. `finally clear_cross_cache()`.

Root supplemental inference pass trên source640, semantic rank512, region64,
prompt rows không đều và left-padding, backbone BF16/bridge planner FP32,
không autocast, đạt cached-vs-dense `max|delta logit|=4.768e-7`. History
coverage totals `[2,2]` tương ứng bốn summary input positions với usage0.5;
`clear_cross_cache()` đưa history về `None`. Đây là bằng chứng đường runner
hiện tại chính xác trong probe đã chọn.

`test_model_cached_prefix_and_compaction_match_dense_with_live_planner` và
`test_decoder_cached_matches_uncached_with_left_padding` củng cố kết quả.
Finding stale cache ở trên chỉ xảy ra khi caller phá contract prepare/clear;
không được dùng nó để phủ định parity chuẩn.

### 3.3. Copy distribution và chunked CE

`GroundedCopyHead.loss()` chia sequence thành stride
`max(1, chunk_size // batch)` tại
[`grounded_copy.py:402-415`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:402).
Mỗi closure chỉ nhận hidden/target slice và coverage/recent slice tương ứng.
Planner không cập nhật mutable state trong closure; do đó gradient checkpoint
không reset hoặc double-count coverage giữa chunks.

Các test dense-vs-chunked cho chunk size 1/7/1024, FP32/BF16, có planner, so
sánh loss và toàn bộ gradient oracle. Test cũng xác nhận masked source và
masked token không nhận gradient sai. Đây là kiểm chứng toán học hữu ích cho
loss implementation; root đã chạy cả suite, không có failure.

Một chi tiết cần giữ khi phân tích log: training trả `loss_ce` đã normalize
theo số target hợp lệ trong mỗi batch; trainer sau đó scale theo token count
window/global rank. Không lấy trung bình đơn giản trên microbatch để so với
run có target length khác.

### 3.4. Zero-init semantic branch và optimizer stage

Semantic output và semantic gate weight zero-init tại
[`grounded_copy.py:99-115`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:99),
head gate cũng zero-init. Ở bước đầu, chỉ đường mở output có gradient trực
tiếp; các projection phía trước có thể gradient zero cho tới khi output branch
mở. Đây là khởi tạo an toàn để checkpoint semantic v3 bắt đầu gần copy-only,
không phải lỗi gradient.

Root probe sau hai updates với full width1024/rank512, source640 cho cả 13
semantic/planner parameter tensors gradient finite và nonzero. Diễn giải đúng
là “có thể học sau khi output mở”, không phải “mọi tham số học ngay step đầu”.

`set_stage_trainability()` và `build_optimizer()` ở
[`training/optimizer.py:17-51`](../src/eviseq_update_v3/eviseq_update_v3/training/optimizer.py:17)
đưa grounded copy/semantic/planner vào nhóm cross_attention trong warmup, rồi
mở toàn bộ model ở full stage. DDP reducer được rebuild mỗi stage tại
[`training/engine.py:101-112`](../src/eviseq_update_v3/eviseq_update_v3/training/engine.py:101).
Với semantic output zero-init, `find_unused_parameters=True` và warmup stage
đã được kiểm tra trên CPU Gloo; không thấy parameter bị bỏ khỏi optimizer.

### 3.5. DDP gradient weighting

Trainer đếm local valid target tokens, all-reduce `window_tokens`, rồi scale
loss mỗi microbatch bằng `world_size * local_tokens/window_tokens` ở
[`training/engine.py:152-190`](../src/eviseq_update_v3/eviseq_update_v3/training/engine.py:152).
Vì DDP averages rank gradients, hệ số world size khôi phục mean trên toàn bộ
target tokens thực. `no_sync()` chỉ bỏ all-reduce các microstep chưa cuối
window; accumulation remainder vẫn có zero-label placeholder ở rank thiếu
sample.

Test hai process Gloo so sánh serial với DDP, target length không đều, odd
dataset size và accumulation remainder; gradients/weights/metrics khớp trong
tolerance. Đây là bằng chứng logic DDP tốt. Chưa nên gọi đó là benchmark NCCL
2 GPU thật; kernel, memory pressure và thứ tự floating point có thể khác.

### 3.6. FP32/BF16

Train ép model parameters về FP32 tại
[`runtime.py:242-249`](../src/eviseq_update_v3/eviseq_update_v3/runtime.py:242),
`AFMRTrainer` cũng `.to(..., dtype=torch.float32)` ở
[`training/engine.py:81-89`](../src/eviseq_update_v3/eviseq_update_v3/training/engine.py:81).
Trên CUDA, autocast BF16 bao quanh forward/loss; optimizer state và update là
FP32. Trên CPU, path dùng FP32 dù test có CPU BF16 autocast để kiểm numerical
tolerance. `torch.float16` không phải compute dtype được validator chấp nhận.

Eval CUDA nạp encoder và decoder backbone bằng compute dtype BF16 tại
[`runtime.py:365-370`](../src/eviseq_update_v3/eviseq_update_v3/runtime.py:365);
bridge và grounded-copy/planner mới tạo vẫn FP32, và `evaluate()` không bật
autocast. Root đã kiểm tra cách dựng model này trên CPU: semantic K và V đều
FP32. Đây là khác biệt có chủ ý so với train FP32 parameters. Cần ghi rõ khi so
logits/ROUGE và khi reproducibility-sensitive. Chưa có đo sai số/throughput
Qwen3 thật dưới BF16 SDPA.

### 3.7. Nhật ký kiểm chứng của review

Không chạy lại full suite theo yêu cầu của parent; dùng kết quả root đã chạy
`PYTHONPATH=. /Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q`:
`286 passed` trong `57.99s`. Các probe root cung cấp gồm full-width1024/
rank512 gradient sau hai updates, parity cached/dense trên native BF16 tiny
graph, future-token causality, left-padding, unequal prompt, compaction,
history reset và DDP/Gloo token weighting.

Trong turn review này, tôi chạy probe helper thuần CPU cho
`_apply_no_repeat_ngram()` với `pad_token_id=eos_token_id=2`. Kết quả:

```text
history=[2,2,5], ngram=1  -> banned=[2,5]
history=[5],   ngram=1    -> banned=[5]
```

Probe này không tạo artifact trong repository. Stale cross-cache là probe của
Luna; evaluation provenance là probe của root trong `TemporaryDirectory`, đã
được tự động dọn. Root bổ sung kiểm tra dtype semantic K/V khi dựng native
backbone BF16, bridge/head FP32 trên CPU và xác nhận cả K/V đều FP32. Các con
số/claim trong báo cáo được đánh dấu là probe
logic/theoretical khi chưa có CUDA hoặc PubMed run thật.

## 4. Chi phí và cấu hình PubMed thật

### 4.1. Số tham số và memory đã tính được

Root đếm trên decoder width1024:

| Graph | Grounded head | Semantic + planner |
|---|---:|---:|
| v2 1×128 | 918,018 | 524,545 |
| v3 revised 4×128 | 3,021,065 | 2,627,592 |

Con số này không gồm toàn encoder/decoder. Semantic K/V rank512 tăng 4 lần so
v2 rank128 cho phần cache tương ứng. Với eval batch64, source4096, semantic
K+V FP32 riêng phần đó là:

```text
64 × 4096 × 512 × 2 × 4 bytes = 1,073,741,824 bytes ≈ 1 GiB
```

Đây là dung lượng K/V theo dtype của code eval hiện tại: backbone BF16 nhưng
semantic head FP32, không autocast, nên semantic K và V đều FP32. Root đã xác
nhận dtype bằng probe CPU cùng cách dựng model. Train có BF16 autocast có thể
tạo semantic V BF16; không áp dụng điều đó cho eval. Con số này chưa tính cross
K/V, copy keys, Qwen self KV, encoder activations, allocator overhead hoặc
temporary SDPA buffers; không phải peak VRAM đo bằng
`torch.cuda.max_memory_allocated`.

Với CE chunk size1024, batch48 và stride `1024//48=21`, một tensor attention
logic kích thước `[48,4,21,4096]` FP32 khoảng63 MiB. Đây chỉ là working tensor
lý thuyết của probe, không phải toàn bộ graph. Hierarchical path dùng scatter
reduce/token normalization và có thể tạo temporary khác tùy kernel.

### 4.2. `run_pubmed_pair.sh` thực sự chạy gì

Script tại [`scripts/run_pubmed_pair.sh`](../src/eviseq_update_v3/scripts/run_pubmed_pair.sh)
đặt các mặc định:

- `CUDA_VISIBLE_DEVICES=0,1`, `NPROC_PER_NODE=2`;
- `BATCH_SIZE=48` là mỗi GPU;
- accumulation mặc định1 khi có hai worker, nên effective batch là
  `48×2×1=96` mỗi optimizer update;
- `MAX_GRAD_NORM=1.0`;
- AFMR value anchor, grounded copy, semantic hierarchical coverage, rank512,
  4 head×128, region64, partition/coverage/continuity bật, fusion
  norm-preserving;
- PPLX và Qwen embedding chạy tuần tự, cùng Qwen3 decoder;
- training một interface warmup + ba full epochs trong PubMed override;
- eval single-process sau mỗi run, mặc định split test và batch64;
- benchmark eval là greedy (`num_beams=1`, `do_sample=false`).

Temperature/top-p chỉ tồn tại ở `generate_sampled()` API
([`evaluation/generate.py:75-90`](../src/eviseq_update_v3/eviseq_update_v3/evaluation/generate.py:75));
config validator bắt eval benchmark greedy. Chúng không sinh candidate, draft,
hay self-improvement trong training.

Nếu chạy một GPU, script đổi accumulation mặc định thành2 để vẫn giữ effective
batch96; đó không phải cùng training trajectory với hai GPU. `DistributedBatchSampler`
dùng global batch96 để bucket rồi shard rank ở
[`data/sampling.py:39-75`](../src/eviseq_update_v3/eviseq_update_v3/data/sampling.py:39).
Do đó single-GPU batch48×accum2 và two-GPU batch48×accum1 có token-weighted
update target tương đương về lý thuyết, nhưng order/grouping, padding, dropout
random stream và floating point khác.

### 4.3. Checkpoint compatibility

V3 `architecture_spec.graph_version` là
`afmr_token_depth_lowrank_v3`, semantic graph có rank/head/planner/fusion
details tại [`training/checkpoint.py:18-81`](../src/eviseq_update_v3/eviseq_update_v3/training/checkpoint.py:18).
V2 checkpoint có graph/spec khác; test roundtrip đã xác nhận copy-only hoặc
semantic rank khác bị reject. Đây là bảo vệ đúng cho state key/graph chính.

Không nên xem v3 checkpoint là warm-start tương thích tự động với v2. Muốn so
fair, phải train v3 từ cùng pretrained encoder/decoder và cùng data protocol;
load v2 chỉ khi có adapter/state mapping được ghi rõ.

## 5. V3 giữ gì của v2, và giữ với điều kiện nào

| Thành phần v2 | V3 cuối | Điều kiện diễn giải |
|---|---|---|
| AFMR full-memory residual + value anchor | Giữ nguyên cơ chế v2, gồm value anchor key/value tách biệt | Refined key/source bias vẫn đổi hidden sau fine-tune |
| Qwen cross-attention mọi layer | Giữ, copied self projections + static cross gate | Query-cross gate v3 ban đầu tắt trong recipe cuối |
| Grounded copy theo source-only alignment | Giữ | Alignment không đọc reference; truncation boundary có thể bỏ token |
| Duplicate token-ID marginalization | Giữ | CE chính xác cho source occurrences trùng ID |
| CE-only training | Giữ | Không có contrastive/R-Drop/NEFTune/candidate generation |
| Chunked CE/checkpoint gradients | Giữ và mở rộng cho planner slices | Đã có dense/chunk gradient oracle |
| FP32 update + BF16 CUDA compute | Giữ | Eval CUDA cast BF16; real-Qwen benchmark chưa có |
| Static source cache | Giữ | Runner safe; public API stale-source risk còn tồn tại |
| Semantic 1×128 | Đổi thành 4×128 hierarchical | Params/cache/compute tăng; diversity chưa được chứng minh |
| Free semantic residual | Đổi norm-preserving tangent fusion | Giữ norm hidden; với hidden/state cố định P_copy vẫn giữ, nhưng không bảo đảm P_copy/R2 sau full fine-tune |
| Không có prefix plan | Thêm coverage/recent tracker | Learned usage, chưa phải fact/discourse planner |

Vì vậy mô tả chính xác là “V3 bảo vệ magnitude và thêm prefix-conditioned
hierarchical source read, trong khi giữ grounded copy/marginalized CE của v2”.
Mô tả “semantic branch không ảnh hưởng copy” hoặc “giữ nguyên R2” là quá mạnh
nếu nói về checkpoint sau full fine-tuning.

## 6. Fairness của các số ROUGE hiện có

Các con số lịch sử cần ghi điều kiện đầy đủ:

- baseline `eviseq_new`: một GPU, batch48, accumulation2, epoch4, clip1.0;
- update v2: điểm lịch sử không có metadata hoàn toàn matched;
- revised v3 script: hai GPU, batch48/GPU, accumulation1, effective batch96,
  một warmup + ba full, clip1.0, sampler global bucket khác.

Chênh lệch lịch sử v2 R2 so với baseline chỉ khoảng `+0.052` điểm theo thông
tin hiện có; chưa đủ nói là advantage ổn định. V3 chưa có PubMed score trong
review này. Cùng ROUGE backend là điều kiện bắt buộc: evaluator mặc định dùng
Python `rouge` 1.0.0, còn Perl ROUGE-1.5.5 là wrapper tùy
`ROUGE155_SCRIPT`. Hai backend, preprocessing khoảng trắng và detokenize có
thể cho số khác.

Để claim vượt T5Gemma hoặc decoder-only fine-tuned PubMed, cần tối thiểu:

1. cùng split và preprocessing/fingerprint;
2. mỗi model dùng tokenizer native; kiểm soát nội dung nguồn được nhìn thấy,
   ngân sách độ dài đầu ra và normalization khi chấm, đồng thời ghi rõ EOS/stop
   của từng model. Cùng số token không tự đồng nghĩa cùng lượng văn bản;
3. với ablation EviSeq, giữ batch/updates/seed/clip/precision và checkpoint
   selection cố định. Với T5Gemma/decoder-only, cho ngân sách tuning công bằng,
   báo đầy đủ hyperparameters/compute và dùng validation để chọn checkpoint;
   không buộc các họ model khác nhau phải dùng cùng learning rate;
4. cùng greedy decoding cho benchmark chính; temperature/top-p chỉ là bảng
   candidate riêng;
5. ROUGE-1.5.5 chạy cùng script/options nếu claim dùng 1.5.5;
6. nhiều seed hoặc paired bootstrap confidence interval;
7. báo R1/R2/L riêng và score theo source-length bucket;
8. manual/factuality evaluation và citation/entailment metric riêng trước khi
   claim giảm hallucination.

Không có test pass nào thay cho các phép so sánh trên. Đặc biệt, test cached
parity chỉ nói inference không đổi so với dense; test chunk gradient chỉ nói
loss implementation đúng; cả hai không phải evidence ROUGE.

### 6.1. Output probability, repetition penalty và candidate sampling

`_mix_logits()` không trả trực tiếp `log P_mix`; nó trả
`log P_mix + logsumexp(LM_logits)` tại
[`grounded_copy.py:378-387`](../src/eviseq_update_v3/eviseq_update_v3/modeling/grounded_copy.py:378).
Đây là một offset chung theo vocabulary, nên softmax của output vẫn đúng
`P_mix`. Đây không phải lỗi normalization.

Repetition penalty của Transformers được áp dụng sau đó trên raw scores ở
[`generate.py:15-20`](../src/eviseq_update_v3/eviseq_update_v3/evaluation/generate.py:15).
Raw-logit penalty không bất biến với việc cộng một offset chung; vì vậy hai
score tensors có cùng softmax trước penalty vẫn có thể cho lựa chọn khác sau
penalty. Đây là nhận xét từ công thức và code, không phải một benchmark hay
probe decoding riêng của root; nó là
contract của processor trên raw logits. Không nên dùng “P_copy giữ nguyên” để
đòi hỏi output sau repetition penalty giữ nguyên.

Temperature/top-p ở [`generate.py:75-90`](../src/eviseq_update_v3/eviseq_update_v3/evaluation/generate.py:75)
chỉ chạy trong `generate_sampled()`, sau các constraint scores. Benchmark script
dùng greedy; sampling là API candidate ngoài training, không phải self-improve.

## 7. Thứ tự xử lý được đề xuất

### Ưu tiên 1 — bảo đảm predictions đúng checkpoint trước khi chốt score

1. Chặn `LUNA-EVAL-001`: manifest/fingerprint checkpoint-config-data-tokenizer-
   generation; reject complete/partial predictions mismatch.
2. Tách output path theo checkpoint/encoder/seed/backend và lưu
   `resolved_config.yaml` cùng prediction.
3. Ghi provenance vào mỗi ROUGE report: checkpoint epoch/step, split, number of
   examples, backend, detokenization và hash predictions.

### Ưu tiên 2 — bảo vệ API và làm rõ cách diễn giải

1. Sửa stale cross-cache bằng explicit cache flag hoặc cache identity; thêm
   A→B regression test.
2. Lọc padding khỏi non-chat repetition/no-repeat history; thêm unequal-prompt
   test với pad=eos.
3. Khi resume, validate training/data manifest và world size/effective batch;
   nếu override, ghi lineage rõ.
4. Chạy source-length diagnostic: số region, số active head, context delta khi
   bật/tắt coverage/continuity.

### Ưu tiên 3 — ablation để biết v3 thật sự giúp gì

Giữ mọi điều kiện train/eval giống nhau và chạy ít nhất:

| Ablation | Câu hỏi |
|---|---|
| v2 1×128 | V3 tăng do capacity hay do planner/fusion? |
| 4×32 | Width128 có cần thiết không? |
| 4×128, partition off | Partition có giúp R1/L hay chỉ mất capacity? |
| planner off / coverage off / continuity off | Thành phần nào làm R2/R1 thay đổi? |
| fusion residual | Norm preservation có giúp score hay chỉ giữ scale? |
| semantic head gate off | Gate có suppress quá mạnh hoặc không học? |
| copy/AFMR frozen semantic-only | Có thể giữ copy/R2 khi chỉ mở read branch không? |

Mỗi run cần lưu loss train/validation, R1/R2/L của cùng checkpoint, copyable
token recall, source-length bucket, attention overlap/head activity và peak
VRAM/tokens/s trên GPU thật.

## 8. Verdict cuối của Luna

**Về correctness:** V3 cuối có đường gradient hợp lệ trong các test/probe hiện
có. Causal planner không nhìn label/future trong đường được kiểm tra; dense,
chunked CE, cached decoding, left padding, compaction/reset và token-weighted
DDP đều có bằng chứng tốt. Không phát hiện bug làm gradient không cập nhật
trọng số trong train chuẩn.

**Về kiến trúc:** AFMR/value-anchor + Qwen cross + grounded copy/marginalized CE
là nền tảng giữ được các cơ chế của v2. So với v2 1×128, v3 giữ 128 chiều/head,
tăng số head lên 4 và tổng rank lên 512;
hierarchical planner và tangent fusion giải quyết một số đường tác động mà v2
chưa có. Nhưng hard partition chưa đảm bảo phủ hiệu quả trên source ngắn,
planner chưa biết fact, multi-head chưa đảm bảo diversity, và norm-preserving
không bảo vệ logits/copy sau full fine-tune.

**Về thực nghiệm:** Chưa thể kết luận v3 vượt v2, decoder-only hoặc T5Gemma;
chưa thể claim giảm hallucination. Trước hết phải sửa evaluation provenance,
chạy matched ablation, kiểm tra ROUGE-1.5.5 cùng preprocessing, rồi mới dùng
điểm cuối để viết claim. Reuse predictions từ đúng checkpoint/config vẫn hợp
lệ; chỉ khi chưa xác minh được provenance thì không nên dùng score đó cho
claim. Eval kết thúc thành công tự nó không xác nhận được điều này.
