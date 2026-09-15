# DeepSeek-V4.1-Flash: trích xuất và phân tích kỹ thuật

Nguồn đọc: [DeepSeek_V41_Tech_Report.pdf](/Users/kieugiangbien/Downloads/Project/LLM2Seq/Technical_Report/DeepSeek_V41_Tech_Report.pdf).

Tài liệu có 51 trang. Text extraction dùng được cho toàn bộ PDF; phương trình và sơ đồ Causal Encoder-Decoder (CED) ở trang 9 đã được kiểm tra trực quan. Số trang trong tài liệu này là số trang PDF được in ở chân trang, không phải số dòng của bản text extraction.

Các con số và kết luận benchmark dưới đây là **tuyên bố của technical report**. Chúng chưa được tái lập độc lập trong repository này. Report không cung cấp một ablation chỉ bật/tắt CED trong khi giữ CSA2, FP4, MoE, dữ liệu và post-training cố định; vì vậy không được quy mọi cải thiện chất lượng cho CED.

## Synthesis

DeepSeek-V4.1-Flash là một multimodal MoE có 552B backbone parameters, 196B Engram parameters và context tối đa một triệu token. Ý tưởng trung tâm không phải tăng chất lượng bằng một loss mới, mà là đồng thiết kế ba lớp:

1. kiến trúc giảm số phép tính và kích thước KV cache;
2. cache precision và cache reuse;
3. hệ thống training/inference phục vụ workload agent dài và nhiều lần prefill.

CED chia một Transformer causal thành nửa encoder và nửa decoder. Nửa encoder tạo hidden anchor ở giữa mạng. Global K/V của các lớp decoder phía trên được chiếu từ anchor đó thay vì tính lại từ hidden của từng lớp. Local Sliding-Window Attention (SWA) vẫn được tính theo từng lớp để giữ độ sâu xử lý cục bộ. Với context dài, CED làm giảm chi phí prefill xấp xỉ một nửa.

CED chỉ là một phần của hệ thống. CSA2 tiếp tục chia sẻ global KV, indexer K và Top-K indices giữa các lớp. Hierarchical Sparse Indexer giới hạn miền tìm kiếm của indexer sau. FP4 giảm kích thước cache. SWA Bounded Replay chấp nhận trạng thái xấp xỉ để loại SWA KV khỏi persistent cache.

Ở post-training, report nói rõ không đưa ra thuật toán RL mới. Công sức chính nằm ở data synthesis, environment verification, asynchronous rollouts, controllable reasoning effort, model merging và on-policy distillation.

## Luận đề và cây lập luận

### Luận đề trung tâm

Report lập luận rằng có thể giảm mạnh chi phí serving context dài mà vẫn giữ năng lực của mô hình bằng cách tách các nguồn chi phí theo ba trục: kích thước mỗi KV entry, số token được giữ hoặc chọn, và số lớp phải tạo cache độc lập.

### Các nhánh hỗ trợ

- **CED giảm prefill:** nửa decoder không cần chạy global KV từ toàn bộ prompt; global KV được chiếu từ hidden ở nửa encoder (tr. 9).
- **SWA giữ độ sâu cục bộ:** local K/V vẫn lấy từ hidden hiện tại của từng lớp; sự đánh đổi là cần replay cho các token cuối (tr. 9–10, 20).
- **CSA2 giảm chi phí theo lớp:** Full, Reindex và Reuse chia sẻ KV/index và chỉ cập nhật selection khi cần (tr. 9–12).
- **FP4 và cache policy giảm bộ nhớ:** global KV được QAT ở FP4; SWA KV được giữ trong bộ nhớ ngắn hạn hoặc tái tạo thay vì lưu persistent (tr. 14, 19–20).
- **Hệ thống làm cho chia sẻ khả thi:** shadow indexer, pipeline payload và micro-batch state management giữ semantics và gradient khi các lớp dùng chung state nhưng nằm ở pipeline stage khác nhau (tr. 17–18).
- **Dữ liệu và post-training tạo phần lớn gain được báo cáo:** report dùng 45T token pretraining và nói post-training algorithm vẫn theo SFT → RL → OPD chuẩn (tr. 6, 25).

### Các điểm không được suy ra

- Không thể suy ra CED tự nó làm tăng ROUGE hoặc giảm hallucination.
- Không thể suy ra bounded replay luôn tương đương full forward; report thừa nhận đây là xấp xỉ.
- Không thể suy ra các kích thước 5120, 64 heads hoặc Top-512 phù hợp với mô hình nhỏ hơn.
- Không thể chuyển nguyên optimizer, RL harness hoặc cache runtime vào EviSeq chỉ bằng config.

## 1. Tổng quan mô hình

### Cấu hình chính

Report mô tả một language backbone 40 causal Transformer layers, gồm 20 lớp causal encoder và 20 lớp decoder. Hidden dimension là 5120. Mô hình có vision encoder, MLP projector, DeepSeekMoE, Single-PassmHC, Engram, DSpark, CED, CSA2 và Hierarchical Sparse Indexer (tr. 7–8, 21–22).

Model có 552B backbone parameters và 196B Engram parameters. Số parameter được kích hoạt mỗi token là 8B trong prefill và 16B trong decode. Đây là số activated parameters của một MoE rất lớn, không phải kích thước dense tương đương.

### Multimodal input

Vision encoder tạo spatial grid. Pixel-unshuffle 3×3 giảm số visual token chín lần trước khi MLP projector đưa chúng về hidden dimension của language backbone. Visual embeddings được chèn vào vị trí image token và đi chung với text từ đầu pretraining.

DeepSeek-ViT thay absolute positional embedding bằng 2D-RoPE, dùng linear patch projection thay convolution để tương thích với Muon, RMSNorm và SwiGLU. Vision encoder được pretrain contrastive với SigLIP rồi autoregressive fine-tune cùng một 4B MoE LLM; sau đó LLM bị bỏ và vision encoder được giữ lại (tr. 8, 22–23).

Để tránh text/image tokens cạnh tranh cùng một load-balancing bias, report dùng correction bias riêng cho từng modality. Bias chỉ thay đổi lựa chọn expert; routing score gốc vẫn dùng để tính trọng số output (tr. 8).

## 2. Causal Encoder-Decoder (CED)

### Vấn đề CED giải quyết

Agent thường gọi tool nhiều lần. Khi prefix không có trong cache, mỗi request phải prefill toàn bộ context. Với context dài, chi phí này lớn dù phần lớn prompt chỉ dùng để tạo cache.

CED được lấy cảm hứng từ YoCo. Khác với encoder-decoder T5, causal encoder ở đây vẫn dùng causal masking để có thể tái sử dụng ưu điểm của backbone decoder-only.

### Global attention path

Gọi tổng số lớp là `L`. Nửa dưới `L/2` là causal encoder. Gọi hidden ở lớp cuối encoder là `H_{L/2}`. Với mỗi lớp decoder `l > L/2`, report định nghĩa:

```text
C_l = H_{L/2} W_l^KV
Z_l = H_{L/2} W_l^Z
```

`C_l` chứa global KV entries của lớp `l`; `Z_l` là compression weights tương ứng. Projection có trọng số riêng theo lớp nhưng dùng chung hidden anchor.

Trong Transformer thông thường, global K/V của lớp `l` được tạo từ `H_l`. Trong CED, global K/V của phần decoder không đi qua `H_l`; vì thế chỉ cần xử lý nửa encoder trên toàn prompt để có global cache cho phần trên.

### Local SWA path

SWA vẫn tạo local K/V từ hidden hiện tại `H_l` ở mọi lớp. Điều này giữ computational depth cho thông tin cục bộ nhưng tạo một vấn đề: nếu muốn lấy SWA KV của decoder cho toàn prompt, phải chạy thêm nhiều lớp trên nhiều token.

Report giải quyết bằng Decoder SWA Bounded Replay: chỉ replay `n_win` token cuối của prompt. Trạng thái SWA tái tạo không hoàn toàn giống full forward; đây là trade-off compute–fidelity.

### Độ phức tạp

Với `N >> n_win`, report cho rằng prefill giảm từ:

```text
O(NL)
→ O(NL/2 + n_win × L/2)
≈ O(NL/2)
```

Mức “gần một nửa” là cho global prefill path trong bối cảnh dài; chi phí SWA replay và các kernel sparse vẫn phải được tính trong hệ thống thật.

### Luồng gradient của CED

Report không trình bày backward graph riêng, nhưng phương trình xác định được các đường chính:

```text
loss của global attention lớp trên
  → W_l^KV, W_l^Z
  → H_{L/2}
  → causal encoder phía dưới
```

Đường global K/V của lớp trên không truyền qua `H_l` để tạo K/V. Tuy nhiên `H_l` vẫn nhận gradient từ query, local SWA, FFN và residual của chính lớp đó.

Ở inference, cache là state đã materialize nên không có gradient. Ở training, các projection layer-dependent phải được tối ưu và hidden anchor phải còn trong graph đến khi backward hoàn tất.

### CED khác EviSeq như thế nào

EviSeq hiện tại có encoder Qwen3 riêng và decoder Qwen3 riêng. Decoder cross-attention đọc `H0/M`; grounded copy và semantic reader cũng dùng các source memory này. CED lại chia một stack causal duy nhất thành hai nửa, rồi tạo global K/V của nửa trên từ hidden giữa stack.

Điểm tương đồng chỉ là ý tưởng **anchor**:

- EviSeq: `H0` là value anchor cho source read/copy.
- CED: `H_{L/2}` là layer/cache anchor cho global K/V của decoder.

CED không có copy-mass-preserving mixture, không có semantic correction và không phải cơ chế giảm hallucination. Nó là một trục tối ưu serving khác với trục tối ưu ROUGE của EviSeq.

## 3. Compressed Sparse Attention 2 (CSA2)

CSA2 giảm chi phí theo ba trục:

1. **entry dimension:** GQA giảm số KV heads, MLA chia sẻ latent KV;
2. **sequence dimension:** nén nhiều token thành một entry và chỉ chọn Top-K;
3. **layer dimension:** chia sẻ cache hoặc selection giữa các lớp.

CSA2 bỏ overlap và absolute positional embedding của CSA cũ. Indexer K được chiếu từ main KV thay vì có một compression path tách biệt.

### Ba mode

- **Full:** tạo main KV, indexer Q/K và Top-K indices mới.
- **Reindex:** dùng main KV/indexer K từ Full gần nhất nhưng dùng indexer Q riêng để chọn Top-K mới.
- **Reuse:** dùng cả main KV và Top-K indices từ mode trước, không chạy indexer mới.

Mọi mode vẫn tạo main Q và SWA KV ở lớp hiện tại. Khi kết hợp CED, Full Mode của decoder tạo global KV từ `H_{L/2}`; Reindex và Reuse giữ nguyên.

### Hierarchical Sparse Indexer

Full Mode đầu tiên quét toàn bộ causal context, chọn Top-K và đồng thời chọn các block có điểm indexer cao nhất. Ví dụ report dùng 2.048 blocks, mỗi block 8 vị trí, tạo candidate pool tối đa 16.384 vị trí.

Reindex Mode sau đó chỉ chấm điểm trong candidate pool. Reuse Mode dùng selection gần nhất. Vì candidate pool có kích thước cố định, chi phí indexer của các lớp sau không còn tăng tuyến tính theo context length.

Điểm quan trọng cho training là candidate restriction được áp dụng giống nhau trong training và inference. Đây là “training-aware approximation”; nếu chỉ giới hạn candidate khi inference thì model sẽ học một search domain khác với domain lúc chạy thật.

Giới hạn: candidate pool có thể bỏ mất vị trí cần thiết. Report đưa risk này vào phần limitations nhưng không cung cấp ablation đầy đủ về selection error ở mọi context.

## 4. Các architectural extensions

### Single-PassmHC

mHC duy trì `n` residual streams và dự đoán các hệ số trộn `A_l`, `B_l`, `C_l`. Bản cũ cần nhiều kernel vì input mixing của block hiện tại phụ thuộc hệ số vừa tính từ chính block đó.

Single-PassmHC dịch input-mixing coefficient lùi một block:

```text
X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l)
```

Nhờ dependency được dịch, residual update, input mixing và coefficient prediction có thể fuse thành một kernel. Report nói độ suy giảm thực nghiệm không đáng kể. Trong pretraining họ vẫn giữ implementation multi-kernel; Single-PassmHC chủ yếu để tối ưu deployment. Mega-mHC giảm activation memory traffic khoảng một nửa so với implementation cũ.

Điểm chuyển giao: đây là tối ưu hệ thống có thay đổi nhỏ trong residual dependency, không phải một semantic mechanism. Không nên đưa vào EviSeq nếu chưa có lý do về memory traffic hoặc nếu cần giữ parity với pretrained checkpoint.

### Engram

Engram là conditional memory được truy cập bằng token n-gram, tách memorization khỏi computation của Transformer.

Report dùng hai module, tổng cộng 196B parameters. Mỗi module dùng n-gram orders 2, 3, 4; có 8 hash heads; embedding dimension tổng 2.048 cho mỗi order; mỗi hash table khoảng 16M entries với kích thước prime khác nhau. Module đặt ở layer 1 và 14.

Lookup index chỉ phụ thuộc input token sequence nên có thể prefetch trước khi pipeline microbatch chạy. Embeddings và key/value projections dùng FP8. Embedding gradients được buffer trong backward rồi gửi về rank sở hữu sau backbone backward. Sinkhorn normalization giảm chi phí ghi lại ma trận normalized.

Đây là bộ nhớ tham số khổng lồ cho long-tail lexical knowledge. Nó khác grounded copy: Engram trả embedding learned từ n-gram, còn grounded copy marginalize attention mass trên token nguồn.

### DSpark

DSpark là speculative decoding module được train sau pretraining. Drafter có 3 Transformer blocks, SWA window 128; một forward pass tạo base logits cho 5 vị trí draft song song. Markov head mô hình dependency giữa các draft token. Confidence head dự đoán acceptance probabilities và scheduler chọn verification length theo throughput.

Trong stage riêng, backbone frozen và chỉ DSpark được train. Trong post-training, DSpark tiếp tục train cùng backbone nhưng objective của DSpark không truyền gradient vào backbone. Điều này giữ speculative module đồng bộ với policy mà không làm objective phụ thay đổi backbone.

### FP4 Main KV Cache

Report áp dụng QAT cho global main KV cache, dùng MXFP4 E2M1 với một E4M3 scale cho mỗi 16 channels. Cache được quantize sau RoPE. SWA KV giữ FP8 vì nhạy với quantization.

Lập luận numeric của report: sau RMSNorm, latent KV 512 channels có L2 norm tối đa xấp xỉ `sqrt(512) ≈ 22.6`; magnitude quan sát trong training khoảng 10, thấp hơn range format được chọn. Đây là lập luận cho cache range, không phải bảo đảm mọi model hoặc mọi layer có cùng bound.

### Optimization

Report tách optimizer theo loại tham số:

- Muon cho linear transformation matrices;
- head-wise Muon cho Query và Key weights để mỗi head có preconditioner riêng;
- AdamW cho RMSNorm weights, bias và các non-matrix parameters;
- momentum update + Sinkhorn balancing cho Engram tables, token embeddings và prediction head.

AdamW dùng `beta1=0.9`, `beta2=0.95`, `epsilon=1e-20`, weight decay 0.1. Muon dùng momentum 0.95, weight decay 0.1 và rescale RMS update về 0.18. Sinkhorn normalization lặp số bước lẻ `K=11`, ngưỡng `tau=1e-3`.

Đây là optimizer/system co-design cho pretraining khổng lồ. Không có bằng chứng rằng Muon hoặc Sinkhorn sẽ tốt hơn AdamW cho fine-tuning PubMed 1.37B; chuyển thẳng sẽ làm thay đổi biến độc lập của thí nghiệm.

## 5. Training infrastructure

### Contrastive vision training

Vision encoder được train contrastive trước generative fine-tuning. Text và visual features phải all-gather giữa data-parallel ranks. Vì gradient text chỉ phụ thuộc visual features đã gather và ngược lại, report overlap all-gather với forward/backward thay vì chặn pipeline.

Vision encoder được disaggregate khỏi LLM parameter tree. Mỗi step tách thành vision forward, LLM forward/backward và vision backward, giảm interference giữa hai loại workload.

### Ultra-long multimodal loading

Image của một sequence dài được shard cân bằng giữa context-parallel ranks và mỗi image chỉ load một lần. Điều kiện để I/O được che bởi compute được viết theo per-token quantities; `N` bị triệt tiêu, nên bottleneck phụ thuộc bytes/token, compute/token và bandwidth hơn là chỉ phụ thuộc sequence length.

Trong RL rollout, image được chuyển incremental và output preprocessing CPU được cache trên distributed filesystem.

### CSA2 training với shared state

Khi source và consumer layer nằm ở pipeline stage khác nhau, module reuse trực tiếp không còn phù hợp stage-local execution. Report dùng:

- **shadow indexers:** replica nhẹ trên mỗi stage, một logical owner giữ parameters/checkpoint;
- **pipeline payload extensions:** gửi representation và sparse routing information qua point-to-point payload;
- **micro-batch shared-state management:** giữ state đến consumer cuối cùng, kể cả activation recomputation và backward, rồi release.

Điểm quan trọng về gradient là shared state không chỉ cần copy ở forward; lifetime và payload phải đủ dài để gradient aggregation có thể hoàn thành.

### Engram infrastructure

Engram tables được row-shard theo process group riêng, optimizer state tiếp tục shard. Lookup indices biết trước từ input nên prefetch trước pipeline. Embedding gradient được buffer và gửi về owner sau backbone backward. FP8 fetch và fused row/column normalization giảm memory traffic.

## 6. Inference system và cache policy

### Kernel flow và EPD

Report thiết kế inference với kernel fusion; Reuse Mode layer được báo cáo chạy khoảng 15 kernels khi prefill và 11 kernels khi decode. EPD disaggregation tách vision encoding, prefill và decode để scale/overlap độc lập.

### Persistent KV cache

So với V4, V4.1 loại SWA KV khỏi persistent cache và tiếp tục nén global KV. Global KV có lifetime dài và phù hợp SSD/host memory; SWA KV chỉ hữu ích trong thời gian ngắn của active session.

SWA KV được giữ trong distributed memory pool có TTL ngắn. Khi bị evict, hệ thống dùng bounded replay thay vì lưu lâu dài một state ít được tái sử dụng.

### Encoder SWA Bounded Replay

Khi prefix global KV còn nhưng encoder SWA KV mất, hệ thống replay `n_win` token cuối prefix cùng uncached suffix. Global KV đã cache được reuse; SWA KV của suffix được tạo mới.

Report thừa nhận state suffix phụ thuộc vị trí cache hit và không còn mathematically identical giữa các trường hợp. Đây là một approximation boundary cần stress test.

### Decoder SWA Bounded Replay

Trong CED, decoder global KV đã được chiếu từ encoder output. Trở ngại còn lại là decoder SWA KV. Hệ thống replay `n_win` token cuối prompt qua decoder layers, dùng SWA truncation, rồi chỉ dùng state này cho những decode đầu; state không được đưa vào prefix cache dài hạn.

Report nói thêm rằng post-training mô phỏng cùng replay để model thích nghi với approximation. Đây là một ví dụ quan trọng: nếu deployment có approximation có hệ thống, training phải thấy approximation tương ứng.

## 7. Pretraining

### Data construction

Corpus gồm text-only và multimodal data, sau xử lý dùng tỷ lệ token 7:1 giữa text và multimodal. Text model-generated hoặc machine-translated kém được lọc vì có information gain thấp và có thể thành duplication ngầm.

Multimodal pipeline gồm image-text pairs, interleaved web/PDF data và domain-specific data cho visual grounding, OCR, charts, image-code và computer-use trajectories. Image được lọc relevance, deduplicate theo semantic; interleaved documents qua heuristic/statistical filter rồi quality scoring.

Ultra-long documents được deterministic pre-split trước khi mixing. Packing được tối ưu để padding rate không quá `1e-4` theo report.

### Model setup

Thông số đáng chú ý:

- 40 Transformer layers, 20 encoder + 20 decoder;
- hidden dimension 5120;
- hai lớp đầu pure SWA;
- 18 encoder layers còn lại CSA2 compression ratio `m=2`;
- 20 decoder layers CSA2 `m=1`;
- indexer có 32 query heads, head dimension 128;
- main attention có 64 query heads, head dimension 512;
- query compression dimension 1280;
- Top-K sparse attention là 512;
- Hierarchical Sparse Indexer tối đa 2.048 blocks × 8 positions;
- SWA window `n_win=128`;
- mỗi MoE layer có 1 shared expert + 384 routed experts, mỗi token activate 6 routed experts;
- expert intermediate dimension 2304.

Các con số này phục vụ model 552B và không nên copy vào EviSeq. Điều đáng tham khảo là tách rõ query dimension, indexer dimension, cache entry và candidate pool; không phải giá trị tuyệt đối.

### Training setup

Report train sparse attention từ đầu ở sequence length 64K, không có dense-attention warmup, rồi mở rộng lên 1M ở mốc 34T token. Tổng pretraining data là 45T token. Batch giữ cố định 100.6M token.

Learning rate warmup 2.000 steps, giữ `2.6e-4` đến 28T token, cosine decay đến `2.6e-5` từ 28T đến 40T, rồi giữ giá trị đó đến 45T. Report nói không gặp instability.

Vision encoder frozen cho tới giai đoạn learning-rate decay; final normalization và projector vẫn trainable trước đó, sau đó mới unfreeze vision encoder với learning rate nhỏ hơn.

## 8. Base-model evaluation

Report so sánh V4.1-Flash-Base với V4-Flash-Base và V4-Pro-Base trên world knowledge, language reasoning, code/math, long context và multimodal.

Các số được nhấn mạnh gồm:

- MMLU-Pro: 74.1;
- SimpleQA-Verified: 42.3;
- HumanEval: 79.4;
- Codeforces: 3471;
- GPQA không nằm ở bảng base này nhưng được báo 90.9 ở post-training;
- LongBench-V2: 45.2;
- MMMU-Pro: 56.5;
- CVBench: 77.9;
- DocVQA: 95.6;
- RefCOCO average: 86.0.

Report kết luận V4.1 ngang hoặc hơn các model tiền nhiệm trong nhiều benchmark với activated parameter nhỏ hơn. Tuy nhiên Table 1 là so sánh end-to-end giữa nhiều thay đổi: CED, CSA2, FP4, Engram, data và training. Đây không phải bằng chứng causal cho riêng CED.

## 9. Post-training

### Công thức chung

Report dùng SFT → RL → OPD. Không có algorithmic innovation mới; gain được gán chủ yếu cho task synthesis, environment construction, filtering, difficulty calibration và verifiability.

### Task synthesis

Mỗi task được biểu diễn bằng `(problem, environment, verification system)`. Chất lượng có hai trục: difficulty và correctness. Trajectories sau mỗi RL run cung cấp evidence để audit lại task.

General-agent pipeline dựng mock tools, mô phỏng interface/API/behavior và đưa failure cases thật vào environment. Coding-agent pipeline chọn task từ session khó hoặc repository đủ điều kiện, dựng container có test, tạo evaluation points và dùng agent độc lập kiểm tra environment, factual errors, mismatch và hackability.

Đây là quy trình kiểm soát chất lượng dữ liệu RL, không phải self-improvement training bắt buộc cho EviSeq.

### RL trên nhiều scaffold

Rollout chạy trong sandbox, worker container và DSec. Rollout execution được tách khỏi preemptible GPU pool; khi trainer bị preempt, state được giữ để resume.

Để mở rộng compute qua nhiều run, report merge checkpoints từ các scaffold/config khác nhau rồi khởi tạo run tiếp theo. Cách này cộng các hướng tối ưu khác nhau nhưng có rủi ro trộn policy và cần đánh giá kỹ.

### DSec

DSec là platform sandbox quy mô lớn, hỗ trợ sharding, eventual-consistency placement, NUMA isolation, preemption-safe resume và command trajectory logging.

Node dùng local admission constraint để bù cho việc không có strong global consistency. Worker VM bị bind vào NUMA domain; report báo density tăng từ khoảng 1.000 lên hơn 2.500 live containers/node trước khi degradation đo được.

Agent misbehavior được xử lý bằng AppArmor, eBPF network policy và failure/repercussion signal khi agent crash hoặc reward hack.

### Controllable reasoning effort

Model nhận scalar effort `b` trong khoảng 1–100 qua prompt. Với cùng `(x,b)`, nhiều response tạo thành subgroup và reward được mean-center trong subgroup; response ở effort khác không trực tiếp so sánh.

Length penalty giảm theo hàm mũ theo effort:

```text
k(b) = k0 exp(-(b - b_min)/tau)
```

Effort cao bị phạt token ít hơn nên model có thể suy luận lâu hơn. Report dùng các tier `low=50`, `high=75`, `max=100`.

Điểm nổi bật của cơ chế này là điều khiển cost–quality bằng một scalar liên tục mà không đổi weights hoặc decoding config. Nó phù hợp reasoning/agent, không trực tiếp liên quan đến greedy PubMed summarization.

### Asynchronous post-training

Final approach dùng sample-level dispatch: khi đủ một GRPO group cho prompt kế tiếp thì dispatch, không chờ batch hoặc toàn prompt. Batch-level dispatch gây oscillation; prompt-level dispatch bị long-tail stall.

Hai vấn đề được tách:

- **length bias:** sample ngắn hoàn thành sớm và chiếm batch đầu;
- **off-policy staleness:** sample được tạo từ checkpoint cũ.

Hệ thống điều chỉnh per-dataset concurrency, có thể bỏ short sample sớm, giới hạn off-policy ratio và mask token quá stale trong loss.

Rollout có token-level interruption. KV cache và expert routing được persist theo token để resume sau checkpoint switch mà không prefill lại. OPD cuối dùng hơn 40 teacher models khác nhau theo domain và stage.

## 10. Kết quả post-training đáng chú ý

Report báo các con số sau trong Table 3 và Figure 9:

- GPQA Diamond: 90.9;
- HLE: 36.8, text-only subset 39.1;
- Codeforces rating: 3471;
- MathArena Apex: 65.6;
- Terminal-Bench 2.1: 90.6;
- Terminal-Bench 3.0: 30.0;
- Terminal-Bench 4.0: 31.2;
- DeepSWE v1.1: 74.2;
- ProgramBench Almost@1: 20.3 trong bảng so sánh;
- CyberGym: 88.1;
- AutomationBench: 54.8;
- Agents’ Last Exam: 31.8;
- Chartography: 78.9;
- BabyVision: 89.6.

Khi effort tăng từ 25 lên 100, report báo average Pass@1 của tám benchmark reasoning tăng 67.1% → 76.3%, DeepSWE 66.0% → 74.2%, Terminal-Bench 2.1 82.4% → 90.6%, đổi lại khoảng 2.5× output tokens. Gain lớn nhất nằm ở effort thấp đến trung bình; 60–80 đã lấy phần lớn accuracy với ít token hơn max.

Multi-agent evaluation dùng 172 task “golden” của ProgramBench và no-GPU subset của FrontierSWE v2. Ở deadline 8 giờ, ProgramBench Almost@1 được báo 30.04% multi-agent so với 20.39% single-agent. Ở 20 giờ, FrontierSWE v2 Mean@5 là 32.90% so với 28.20%. Report gọi kết quả này là preliminary và so sánh cấu hình mạnh nhất quan sát được.

## 11. Evidence ledger

### CED giảm prefill

- **Claim:** CED giảm gần một nửa prefill.
- **Evidence:** Global KV của lớp decoder trên được chiếu từ `H_{L/2}`; complexity được viết là `O(NL/2 + n_win L/2)`.
- **Location:** mục 2.2, PDF tr. 9.
- **Relationship:** công thức giải thích vì sao số lớp xử lý toàn prompt giảm.
- **Confidence:** Author's stated position.
- **Caveat:** Không có CED-only ablation; chi phí SWA replay và kernel/runtime vẫn phải đo.

### CED giữ năng lực nhờ local SWA

- **Claim:** CED giữ computational depth của local attention.
- **Evidence:** SWA K/V vẫn lấy từ hidden hiện tại ở từng layer.
- **Location:** mục 2.2, PDF tr. 9.
- **Relationship:** local path không bị rút gọn giống global path.
- **Confidence:** Source fact or data.
- **Caveat:** Bounded replay làm local state xấp xỉ khi cache miss.

### Hierarchical indexer làm chi phí lớp sau độc lập hơn context length

- **Claim:** Candidate pool cố định làm chi phí Reindex sau gần constant theo context length.
- **Evidence:** Full Mode chọn block pool; Reindex chỉ chấm trong pool; Reuse không index mới.
- **Location:** mục 2.3.2, PDF tr. 11–12.
- **Relationship:** giảm số vị trí cần score sau lượt Full đầu tiên.
- **Confidence:** Source fact or data.
- **Caveat:** Candidate pool có thể bỏ vị trí quan trọng.

### Post-training gain đến từ data pipeline

- **Claim:** Dưới optimizer cố định, scale/diversity/verifiability của synthesized data chiếm phần lớn gain.
- **Evidence:** Report nói không có post-training algorithmic innovation mới.
- **Location:** mục 5.1, PDF tr. 25.
- **Relationship:** tách data engineering khỏi thuật toán RL.
- **Confidence:** Author's stated position.
- **Caveat:** Đây là kết luận của hệ thống DeepSeek, không phải kết quả PubMed hay EviSeq.

### Bounded replay là một xấp xỉ có ranh giới

- **Claim:** Replay chỉ `n_win` token có thể đủ dùng.
- **Evidence:** Report báo suy giảm không đáng kể trong các thử nghiệm của họ.
- **Location:** mục 3.2.2, PDF tr. 20.
- **Relationship:** hỗ trợ bỏ persistent SWA KV.
- **Confidence:** Author's stated position.
- **Caveat:** Report cũng nói reconstructed state không mathematically equivalent và có thể lỗi ở boundary chưa test.

## 12. Hạn chế và phản biện

### Thiếu attribution cho CED

CED đi cùng CSA2, hierarchical indexer, FP4, Engram, DSpark, Single-PassmHC, MoE và data pipeline. Table 1/3 chỉ cho thấy hệ thống cuối cùng hoạt động; không cho biết CED đóng góp bao nhiêu.

### CED tối ưu serving hơn là chất lượng

CED giảm số phép tính prefill và cache, nhưng không thêm loss semantic, copy supervision, coverage hay factuality. Với bài toán PubMed, CED chỉ có lý do trực tiếp nếu source rất dài hoặc inference prefill là bottleneck.

### Approximation boundaries

Hai approximation chính là sparse candidate restriction và SWA bounded replay. Cả hai đều có thể tạo failure phụ thuộc vị trí cache hit, token bị loại hoặc long-tail evidence.

### Internal evaluation

Nhiều benchmark và kết luận “comparable/frontier” dựa trên internal framework, internal corpora hoặc harness riêng. Những kết quả này cần được coi là report evidence, không phải tái lập mở hoàn toàn.

### Scale mismatch

CED được train từ đầu với 45T token và 40-layer, 5120-hidden model. EviSeq đang fine-tune backbone khoảng 1.37B trên PubMed. Cơ chế cache projection layer-dependent không thể giả định sẽ giữ parity khi thêm vào checkpoint Qwen hiện có.

## 13. Điều có thể chuyển sang EviSeq

### Có thể chuyển giao

1. **Anchor ổn định:** thử một hidden source anchor duy nhất cho các projection source ở decoder layers, giữ H0 làm value anchor hiện tại.
2. **Projection theo layer:** nếu source projection bị lặp nhiều lần, dùng projection riêng theo decoder layer từ cùng anchor; đây là một ablation rõ ràng.
3. **Training-aware approximation:** bất kỳ candidate pool, source truncation hoặc cache replay nào được dùng khi inference phải xuất hiện trong training để tránh train/inference mismatch.
4. **Tách quality path và efficiency path:** đo ROUGE/NLL riêng với latency, prefill FLOPs và memory; không gán gain chất lượng cho tối ưu cache.
5. **Gradient lifetime:** nếu cache/state được reuse giữa layer hoặc microbatch, giữ state đến consumer cuối trong backward; không release ngay sau forward.

### Không nên chuyển thẳng

- Không gộp encoder và decoder EviSeq thành CED trong bản ROUGE hiện tại.
- Không thêm CED chỉ vì report báo benchmark tốt hơn; cần long-context profiling trước.
- Không copy hidden size, số heads, Top-K hoặc FP4 format của DeepSeek-V4.1-Flash.
- Không thêm DSpark, asynchronous RL, task synthesis hoặc OPD khi mục tiêu hiện tại là CE-only PubMed và không sinh candidates trong training.
- Không đổi optimizer sang Muon/Sinkhorn trong cùng thí nghiệm với thay đổi architecture.

### Nếu sau này cần thử

Thiết kế ít phá vỡ nhất là một variant riêng:

```text
H_anchor = H0 hoặc hidden giữa encoder
K_l, V_l = W_l^K(H_anchor), W_l^V(H_anchor)
```

Giữ grounded copy và semantic mixture hiện tại làm đường quality chính. Chỉ dùng anchor path cho cross-attention, không thay copy path. So sánh với full source K/V trên cùng checkpoint, seed và data.

Acceptance nên gồm ROUGE-1/2/L, validation NLL, source-visible evidence, copy precision, cross-attention entropy, prefill latency, peak memory và độ lệch output giữa dense/replay. Nếu ROUGE không tăng nhưng prefill giảm rõ, kết quả vẫn có giá trị nhưng phải viết là efficiency contribution.

## 14. Kết luận theo mức độ tin cậy

### Author's stated position

- CED giảm gần một nửa prefill.
- CSA2 + FP4 giảm global KV cache còn khoảng 890 bytes/token, xấp xỉ một phần tư V4.
- SWA Bounded Replay giảm persistent KV footprint còn xấp xỉ một phần tám V4.
- V4.1-Flash đạt năng lực reasoning/agentic mạnh với activated parameters thấp hơn.

### Source fact or data

- CED dùng `H_{L/2}` để tạo global KV của các lớp phía trên.
- SWA vẫn layer-local.
- CSA2 có Full/Reindex/Reuse.
- Hierarchical indexer áp dụng cùng candidate restriction trong training và inference.
- Pretraining dùng 45T token, sequence 64K rồi mở rộng 1M.
- Post-training dùng SFT, RL và OPD chuẩn theo report.

### Reasoned inference

- CED là một hướng tối ưu serving/cache, khác hướng tối ưu ROUGE của EviSeq.
- Gradient global KV của lớp trên quay về hidden anchor và projection theo layer, không quay về hidden lớp trên qua đường tạo K/V.
- CED đầy đủ sẽ phá parity của checkpoint EviSeq nếu thêm sau khi pretrained.
- Nguyên tắc đáng học nhất cho EviSeq là anchor reuse có training-aware approximation, không phải toàn bộ CED topology.

### Unverified

- CED có làm tăng ROUGE PubMed hay giảm hallucination hay không.
- CED có tốt hơn EviSeq về chất lượng khi cùng parameter/training budget hay không.
- Bounded replay có ổn định ở mọi loại tài liệu, đặc biệt facts nằm trước replay window, hay không.
- CED-only contribution trong các benchmark của report.

## 15. Câu hỏi recall và transfer

1. Vì sao CED được gọi là causal encoder-decoder dù có phần “encoder”? Hidden nào tạo global K/V của decoder trên?
2. Đường gradient của global KV decoder khác đường gradient của local SWA như thế nào?
3. Tại sao Hierarchical Sparse Indexer phải được áp dụng giống nhau trong training và inference?
4. Điểm nào của CED có thể mượn cho EviSeq mà không biến EviSeq thành một architecture khác?
5. Vì sao không thể dùng benchmark cải thiện của toàn bộ V4.1-Flash để kết luận CED tự nó tăng chất lượng?

## Page map

- PDF tr. 1–6: abstract, motivation, system-level thesis.
- PDF tr. 7–8: architecture overview và multimodal path.
- PDF tr. 9: CED và công thức `C_l/Z_l`.
- PDF tr. 9–12: CSA2 và Hierarchical Sparse Indexer.
- PDF tr. 12–15: Single-PassmHC, Engram, DSpark, FP4 cache, optimizer.
- PDF tr. 16–20: training infrastructure, cache management, bounded replay.
- PDF tr. 20–24: data construction, pretraining setup và base evaluation.
- PDF tr. 25–32: task synthesis, RL, DSec, effort control, asynchronous training và evaluation setup.
- PDF tr. 33–36: post-training results, scaffold transfer và multi-agent scaling.
- PDF tr. 37: conclusion và limitations.
- PDF tr. 38–45: references.
- PDF tr. 46–51: author list, scaffold details, reasoning-effort appendix và exponential penalty.

