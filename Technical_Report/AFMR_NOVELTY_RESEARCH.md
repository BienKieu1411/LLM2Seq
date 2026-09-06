# AFMR: củng cố novelty bằng cách tách thích nghi truy xuất khỏi biến đổi nội dung

Ngày nghiên cứu: 2026-09-05. Code đối chiếu: commit `6696530`.

Trạng thái: báo cáo nghiên cứu và đề xuất thí nghiệm. Biến thể kiến trúc trong mục 5 **chưa được triển khai vào model**. Không có kết quả AFMR mới, không tải model, không thay code/config đang chạy.

## 1. Kết luận và quyết định đề xuất

Hướng nên kiểm tra trước là **value-anchored retrieval adaptation**: giữ đường values từ final encoder state, còn nhánh AFMR chuyên thích nghi địa chỉ truy xuất và mức ưu tiên nguồn. Đặt giới hạn trực tiếp lên tác động của nhánh mới tới attention logits, thay vì chỉ giới hạn sigmoid gate của một residual có norm không bị chặn.

Câu hỏi trung tâm cho paper:

> Khi ghép hai backbone pretrained độc lập để tóm tắt, có cần biến đổi chung representation dùng để chọn nội dung và representation dùng để truyền nội dung, hay nên thích nghi cách đọc trong khi giữ một đường nội dung có anchor?

Đây là giả thuyết kiến trúc có thể kiểm chứng, không phải kết luận rằng code hiện tại sai. AFMR hiện tại vẫn có thể là phương án tốt hơn nếu việc thích nghi values đem lại lợi ích lớn. Chưa có ablation để biết.

Novelty không nên được đặt vào tên ba block hoặc tuyên bố “kết hợp kỹ thuật của các LLM mới nhất”. Đóng góp đáng bảo vệ hơn gồm: một phân tích về sự ghép chung hai vai trò ở interface; một cơ chế kiểm soát riêng đường truy xuất; và bằng chứng thực nghiệm về quality–cost của cơ chế đó trong ghép backbone cho summarization.

Không khuyến nghị đổi sang MoE, thêm CL/KD/ranking, thêm kế hoạch sinh tuần tự, hoặc thay self-attention pretrained trong lần thử này. Không có cơ sở đảm bảo bất kỳ hướng nào sẽ vượt T5Gemma hoặc được NAACL nhận.

## 2. Bản hiện tại thực sự làm gì?

Đối chiếu trực tiếp các file:

- `src/eviseq_new/eviseq_afmr/modeling/afmr.py`, đặc biệt `_depth_weights`, `_focus_prior`, `forward`.
- `src/eviseq_new/eviseq_afmr/modeling/decoder.py`, đặc biệt `CopiedCrossAttention._memory_kv` và `forward`.
- `src/eviseq_new/eviseq_afmr/modeling/controller.py`.
- `src/eviseq_new/eviseq_afmr/training/engine.py`.
- `src/eviseq_new/configs/afmr_base.yaml` và các task config.

AFMR tạo một tensor memory đã thích nghi và một prior nguồn:

\[
M=F_\phi(H^{L-m+1},\ldots,H^L;c),\qquad b=B_\phi(M;c).
\]

Trong mỗi decoder layer, cùng memory đi qua memory normalization rồi cả hai projection:

\[
K=\operatorname{Norm}_K(W_K\operatorname{Norm}_M(M)),\qquad
V=W_V\operatorname{Norm}_M(M).
\]

\[
A=\operatorname{softmax}(QK^\top/\sqrt{d_h}+b),\qquad O=AV.
\]

Như vậy, **projection K và V đã khác nhau**, nhưng representation đầu vào của chúng cùng được sửa bởi depth/feature branch. Không được viết rằng bản hiện tại “chưa tách K/V”.

| Thành phần | Hiện trạng code | Giới hạn của claim |
|---|---|---|
| Depth readout | Softmax theo depth ở từng source token, condition thêm controller | Không phải AttnRes nằm trong mọi backbone layer |
| Feature residual | Low-rank, phi tuyến SiLU, output factor zero-init; gate theo channel | Không phải chỉ một Linear, cũng không có bảo đảm norm residual luôn nhỏ |
| Focus prior | Pool cửa sổ token đa tỉ lệ, score rồi overlap-add về token | Không có local/medium/global self-attention như một số bản hình vẽ |
| Controller | Pooled source, pooled prompt embedding, source length và output budget | Task config đang để decoder prompt rỗng; nhánh prompt riêng không phân biệt document |
| Attention | Cross-attention đọc cùng một full memory với static source bias | Attention vẫn thay đổi theo decoder query; chỉ bias là static |
| Objective | CE-only trong cấu hình hiện tại | Không còn positive sentences, evidence InfoNCE hoặc salience auxiliary loss trong main graph |
| Full memory | Giữ các token còn lại sau tokenizer/truncation | Không đồng nghĩa toàn bộ tài liệu gốc đều còn visible |

Encoder vẫn nhận instruction trong `encoder_prefix`. Vì vậy decoder prompt rỗng không có nghĩa toàn model không có instruction. Nhưng không nên quảng bá “prompt-controllable bridge” như contribution đã được chứng minh khi instruction/budget chưa được biến thiên và đánh giá có kiểm soát.

`ARCHITECTURE_REVIEW.md` là tổng hợp report và có cả đề xuất lịch sử về slots/CL. Phần đó không phải đặc tả main graph AFMR hôm nay. Dùng code hiện hành và các cập nhật cuối `AFMR_IMPLEMENTATION_PLAN.md` để mô tả implementation.

## 3. Những tiền lệ phải đối chiếu để tránh novelty giả

### 3.1. Ghép pretrained models và chọn representation

**Warm-start encoder–decoder không mới.** Rothe và cộng sự nghiên cứu tận dụng pretrained checkpoints cho sequence generation, gồm lựa chọn encoder/decoder từ các loại checkpoint khác nhau. Đóng góp của mình cần nằm ở interface và kết quả ghép có kiểm soát, không phải “lần đầu dùng LLM làm encoder và decoder”. [Rothe et al., TACL 2020](https://aclanthology.org/2020.tacl-1.18/).

**T5Gemma cũng xuất phát từ adaptation của decoder-only models.** T5Gemma 2 tiếp tục dùng UL2 adaptation và thêm tied embeddings, merged self/cross attention. Vì vậy đối lập “T5Gemma không dùng LLM, còn mình dùng LLM” là sai. Bài toán gần hơn là task-level interface adaptation giữa checkpoint độc lập so với một backbone encoder–decoder đã được adapted trước. [T5Gemma](https://arxiv.org/abs/2504.06225), [T5Gemma 2](https://arxiv.org/abs/2512.14856).

**Learned layer mixing đã có tiền lệ lâu.** ELMo sử dụng các internal states thay vì chỉ layer cuối. AttnRes gần đây đưa selective depth aggregation vào residual computation của LLM. AFMR chỉ đọc vài final states một lần tại interface; không được nhận các lợi ích pretraining của AttnRes làm bằng chứng riêng cho AFMR. [ELMo](https://aclanthology.org/N18-1202/), [Attention Residuals](https://arxiv.org/abs/2603.15031).

### 3.2. Salience, focus và content planning

**Gated encoder-to-decoder selection đã có.** SEASS dùng selective gating để kiểm soát representation truyền sang decoder. Đây là tiền lệ trực tiếp cho lập luận “bridge giúp lựa chọn thông tin quan trọng”. [Selective Encoding, ACL 2017](https://aclanthology.org/P17-1101/).

**Global filtering cộng local selection cũng đã có.** Li và cộng sự tách filtering toàn cục khỏi chọn câu khi sinh; vì thế “document plan + local read” chưa đủ là một novelty độc lập. [Explicit Information Selection, EMNLP 2018](https://aclanthology.org/D18-1205/).

**Coarse-to-fine summarization không bắt đầu từ QSA/CSA.** Ling và Rush đã nghiên cứu attention chọn chunk rồi đọc word. Nếu đề xuất phân phối vùng và token conditional bên trong vùng, cần so sánh với hierarchical attention, không chỉ dẫn technical reports của model lớn. [Coarse-to-Fine Attention, NewSum 2017](https://aclanthology.org/W17-4505/).

**Focus prior cũng có nhiều dạng trước đây.** FAME đưa topical focus vào generation; Bottom-Up Summarization dùng content selector để ràng buộc attention. FAME không đồng nhất với additive source-key prior của AFMR, nhưng cả hai là related work bắt buộc về điều khiển nội dung. [Focus Attention, ACL 2021](https://aclanthology.org/2021.acl-long.474/), [Bottom-Up Summarization, EMNLP 2018](https://aclanthology.org/D18-1443/).

### 3.3. Các nguồn gần nhất cho đề xuất tách retrieval và content

**Key-Value Memory Networks** sử dụng encodings khác nhau cho addressing và output của memory read. Đây là nguồn cơ sở cho sự khác nhau về vai trò, đồng thời là lý do không được claim việc tách key/value bản thân nó mới. [Miller et al., EMNLP 2016](https://aclanthology.org/D16-1147/).

**MUDDFormer là đối chứng rất gần.** Nó tạo dynamic connections theo token và theo từng input stream Q/K/V/residual. Chỉ thêm nhiều depth routers riêng cho K và V sẽ dễ bị xem là adaptation của hướng này. Đề xuất ở đây khác ở việc cố ý giới hạn nhánh routing và giữ value bypass tại interface giữa hai backbone, chứ không tăng số dynamic streams trong toàn mạng. [MUDDFormer, ICML 2025](https://proceedings.mlr.press/v267/xiao25d.html).

**Low-Rank Attention Residuals** nghiên cứu routing keys chiều thấp nhưng giữ residual values full-dimensional. Nó làm rõ rằng descriptor để lựa chọn không nhất thiết trùng với nội dung được truyền. Bài này xét depth routing trong LLM, chưa chứng minh lợi ích cho summarization interface. Đây là preprint và phải được trích như preprint, không gán venue. [LR-AttnRes](https://arxiv.org/abs/2607.09694).

### 3.4. Technical reports: học nguyên lý gì và không suy diễn gì?

| Nguồn | Điều có thể học | Điều không được suy ra |
|---|---|---|
| [Kimi K3](https://arxiv.org/abs/2607.24653) và AttnRes | Đọc thông tin có chọn lọc theo depth thay vì accumulation cố định | Lấy nhiều layer hơn sẽ tăng ROUGE; AFMR đã triển khai nguyên AttnRes |
| [DeepSeek-V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), [mHC](https://arxiv.org/abs/2512.24880) | Ràng buộc đường truyền phải tác động lên đối tượng toán học rõ ràng | Sigmoid gate nhỏ tự động cho norm bound như doubly stochastic mixing |
| [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next) | Phân biệt retrieval, residual flow, quality và cost; dynamic read/write có vai trò riêng | Cần bê GDN, QSA hoặc bốn residual streams vào Qwen pretrained |
| [Instella-MoE](https://arxiv.org/abs/2609.00791) | Gating và kiến trúc cần được đánh giá cùng chi phí hệ thống | MoE hoặc Gated MLA sẽ trực tiếp tăng điểm PubMed |

Một phản biện mới quan trọng đến từ **mHC for PEFT**: trong setup fine-tuning frozen OLMo-2 của họ, cố định residual mixing ở identity thường có lợi, và mHC đơn lẻ không nhất quán vượt LoRA. Đây là cảnh báo rằng kết luận từ from-scratch pretraining không tự chuyển sang adaptation. Nó không chứng minh phải đóng băng mọi residual trong AFMR. [mHC for Parameter-Efficient Finetuning, preprint 2026](https://arxiv.org/abs/2607.18130).

Các report trong thư mục đã được dùng qua bản tổng hợp sẵn có; phiên nghiên cứu này đối chiếu thêm nguồn công khai chính thức và các phần liên quan của paper. Không tuyên bố đã đọc lại toàn bộ mọi PDF từ đầu đến cuối trong phiên này.

## 4. Những cách làm novelty mạnh hơn đã cân nhắc

| Hướng | Điểm có thể đóng góp | Rủi ro | Quyết định |
|---|---|---|---|
| Giữ AFMR, chứng minh principle bằng ablation tốt | Insight thực nghiệm về task-level adaptation, có thể đủ giá trị mà không thêm block | Nếu gain rất nhỏ và từng phần không giúp, câu chuyện yếu | Giữ làm reference bắt buộc |
| Value-anchored retrieval adaptation | Tách đường thay đổi attention khỏi đường mang values, kiểm soát được tác động | Có thể hạn chế adaptation cần thiết của values; có tiền lệ decoupling | Thử trước, có điều kiện |
| Region-mass-constrained attention / summary allocation | Phân biệt kế hoạch phân bổ và việc đọc token bằng marginal constraint thật | Gần hierarchical/OT attention; cần score reductions hoặc lặp, đe dọa SDPA và latency | Không chọn gần deadline |
| Thêm query-conditioned low-rank score | Relevance thay đổi theo decoder query | Query dependence vốn có trong cross-attention; có thể chỉ là bilinear correction | Không dùng làm main novelty |
| Differential attention hoặc DEX | Cơ chế xử lý attention noise có tiền lệ mạnh | Scope mới, thêm đường tính hoặc đổi values; chưa rõ bottleneck của mình | Related work/đối chứng sau, không ghép ngay |
| MoE, sparse hard top-k, KD/ranking | Có thể cải thiện capacity hoặc training signal ở bối cảnh phù hợp | Khác mục tiêu interface, thêm biến gây nhiễu và compute | Không chọn |

Differential Transformer và DEX đáng đọc khi phân tích nhiễu attention, nhưng không phải lý do để ghép thêm một block. DEX cũng nhấn mạnh khác biệt giữa thiết kế from-scratch và adaptation pretrained. [Differential Transformer](https://arxiv.org/abs/2410.05258), [DEX](https://arxiv.org/abs/2505.16333).

### Vì sao “source plan + query residual” chưa đủ?

Với correction dạng tuyến tính:

\[
q^\top k+(Uq)^\top(Vh)
=[q;Uq]^\top[k;Vh].
\]

Đây có thể được biểu diễn bằng dot-product attention trên feature mở rộng. Không có nghĩa mọi nonlinear query-conditioned design đều tương đương attention cũ; nhưng riêng phép cộng bilinear này chưa tạo một planning mechanism mới. Muốn gọi là plan phải định nghĩa thêm đối tượng như thứ tự nội dung, coverage state hoặc allocation constraint, và kiểm chứng chức năng đó.

Một prior tĩnh vẫn có thể hữu ích: nó mô hình hóa preference nguồn chung cho bản tóm tắt, trong khi QK xử lý nhu cầu ở từng token. Static không phải lỗi thiết kế tự thân.

## 5. Biến thể đề xuất: value-anchored, bounded retrieval interface

Tên trong báo cáo chỉ mô tả chức năng, chưa nên đặt acronym hoặc claim “first”. Đây là **một thay đổi vào cấu trúc đường truyền AFMR**, không thêm encoder, decoder, evidence slots hoặc một phase training.

### 5.1. Hai vai trò trong cùng một source-token table

Đặt:

\[
H_0=P_0(H^L),\qquad R=F_\phi(H^{L-m+1:L};c)-H_0.
\]

`P0` là anchor projection. Cùng hidden width có thể khởi tạo identity; khác width cần projection thích nghi. Identity theo tọa độ không chứng minh hai model có cùng semantic space.

Ở mỗi decoder layer, tạo anchor K/V:

\[
K_0=N_K(W_K N_M(H_0)),\qquad V_0=W_V N_M(H_0).
\]

Nhánh AFMR chỉ đề nghị sửa keys:

\[
K_* = N_K(W_K N_M(H_0+R)),\qquad E=\mathcal C(K_*-K_0),
\]

\[
K=K_0+E,\qquad V=V_0.
\]

`C` là phép chặn norm được định nghĩa bên dưới. Focus prior đa tỉ lệ có thể giữ từ AFMR trên `H0 + R` để làm ablation ít biến:

\[
b=B_\phi(H_0+R;c),\qquad
A=\operatorname{softmax}(Q(K_0+E)^\top/\sqrt{d_h}+b),\qquad O=AV_0.
\]

Không normalize lại `K0 + E` sau khi chặn mà vẫn nhận nguyên guarantee ở mục 6. Nếu thêm operation sau đó, cần chứng minh lại tác động lên logits.

Mỗi source token vẫn có một key và một value tương ứng. Đây không phải dual-bank attention: decoder chỉ thực hiện một attention read, không có hai output context rồi router chọn bank. Lúc chuẩn bị K/V có thể cần thêm tensor tạm; không được nói training memory cost bằng đúng bản cũ.

### 5.2. Tại sao không đơn giản là K/V projection vốn đã có?

Thay đổi nằm **trước projection**, ở computational graph của adapter:

- Hiện tại: cùng nonlinear multi-depth transformation tác động lên cả K và V.
- Đề xuất: transformation đó chỉ thay địa chỉ truy xuất; V lấy từ anchor final state.
- `W_K`, `W_V`, decoder và encoder vẫn train theo stage như trước. Không đóng băng toàn bộ content learning.

Với tham số riêng `phi` của routing bridge và cố định input/backbone/anchor projection, có `∂V0/∂phi = 0`. Đây là invariant cục bộ. Encoder dùng chung vẫn nhận gradient từ cả hai đường và có thể thay đổi `H0` trong full fine-tuning.

Nếu một key-only MLP trên final state, cùng ngân sách tham số, đã đạt kết quả ngang nhau, thì multi-depth/multi-scale story không được công nhận chỉ nhờ hình vẽ phức tạp hơn.

### 5.3. Câu chuyện thống nhất cho ba block

Ba block được xem là ba thao tác trong **một quá trình thích nghi truy xuất**:

1. Depth readout lấy descriptor phù hợp từ các tầng encoder.
2. Cross-space residual biến descriptor đó thành địa chỉ mà decoder có thể truy vấn.
3. Multi-scale focus cung cấp preference về vùng nguồn ở cấp toàn bản tóm tắt.

Values đi theo đường anchor riêng. Vì vậy lập luận chung là: *thích nghi lựa chọn nguồn mà không buộc cùng adapter phải viết lại nội dung nguồn*. Không cần trình bày ba kỹ thuật quen thuộc như ba invention độc lập.

Đây là hypothesis về inductive bias, không phải claim tăng expressive capacity. Biến thể cố ý **ít tự do hơn** ở value path để kiểm tra trade-off thích nghi và bảo toàn representation.

## 6. Cơ sở toán học có thể bảo vệ

### 6.1. Tách chính xác hai nguồn thay đổi của attention output

So sánh tại một layer, một query và cùng trạng thái backbone. Đặt reference output `o0 = a0 V0`. Với một shared-memory adapter tạo `a` và `V0 + ΔV`:

\[
o-o_0=(a-a_0)V_0+a\Delta V.
\]

Hạng thứ nhất đổi nơi đọc. Hạng thứ hai đổi nội dung được đọc. Đây là đẳng thức, không phụ thuộc việc adapter là tuyến tính hay phi tuyến.

Value anchor loại hạng thứ hai **đối với can thiệp của routing adapter**:

\[
o-o_0=(a-a_0)V_0.
\]

Đẳng thức không cho biết hạng `a ΔV` trong AFMR cũ là có hại. Nó có thể hữu ích; chính ablation phải quyết định. Nó cũng không bảo đảm factuality, vì values là representation học được và decoder còn nhiều tầng biến đổi.

### 6.2. Chặn tác động key residual ngay trong logit space

Giả sử mỗi query sau q-normalization thỏa `||q||2 ≤ Qmax`. Với RMSNorm có scale vector `gamma` và head width `d_h`, một upper bound là:

\[
Q_{\max}=\sqrt{d_h}\,\|\gamma\|_\infty.
\]

Bound này tính từ learned scale hiện tại, không lấy maximum quan sát trên một minibatch làm bảo đảm cho query tương lai. Nếu implementation không có q-norm tương ứng, không được dùng bound này tự động.

Đặt bán kính key residual:

\[
r=\frac{\epsilon_K\sqrt{d_h}}{\max(Q_{\max},\varepsilon)},\qquad
\mathcal C(u)=\frac{u}{\sqrt{1+\|u\|_2^2/r^2}}.
\]

Khi đó `||C(u)||2 ≤ r` và theo Cauchy–Schwarz:

\[
\left|\frac{q^\top E_i}{\sqrt{d_h}}\right|\leq\epsilon_K.
\]

Phép chặn trơn ở zero có đạo hàm identity; không tạo dead route như nhân đồng thời hai zero-initialized factors. Trong GQA, bound phải đúng cho mọi query head dùng chung key head. Thực hiện reduction FP32, xử lý trường hợp scale bằng zero và sai số BF16 khi kiểm thử.

### 6.3. Giới hạn thay đổi odds do interface

Đặt `delta_i = q^T E_i / sqrt(d_h) + b_i`. Với cùng tập token hợp lệ:

\[
\frac{a_i/a_j}{a_{0,i}/a_{0,j}}=\exp(\delta_i-\delta_j).
\]

Nếu key correction bị chặn bởi `epsilon_K` và prior có `max(b)-min(b) ≤ 2s`, thì:

\[
e^{-2(\epsilon_K+s)}
\leq\frac{a_i/a_j}{a_{0,i}/a_{0,j}}
\leq e^{2(\epsilon_K+s)}.
\]

Prior tanh của AFMR có range tối đa `2s` giữa content tokens. Sau log-mean-exp centering, zero nằm trong range của các content priors; do đó gán zero cho prefix/special tokens cũng không tăng range này. Padding bị mask hoàn toàn và không nằm trong phép so sánh odds.

Đây là bound **cục bộ lên ảnh hưởng của interface so với anchor**, không phải giới hạn toàn mô hình sau nhiều layers, không phải theorem về ROUGE và không phải chứng minh chống hallucination. Cap quá nhỏ có thể làm model không thích nghi đủ; cap lớn có thể làm bound gần như không hữu ích.

### 6.4. Quan hệ với mHC

mHC ràng buộc residual mixing bằng doubly stochastic matrices. Ở đây đối tượng bị ràng buộc là thay đổi logit do key residual. Hai cơ chế khác nhau. Nguồn cảm hứng là phải xác định rõ quantity cần kiểm soát, không phải dùng chữ “bounded” cho một gate rồi suy ra norm bound của cả mạng.

### 6.5. Kiểm tra số học đã thực hiện

Chạy bằng `bienkieu_env`, CPU, tensor FP64 ngẫu nhiên, không tải model:

- 100 trường hợp, `B=2`, query length 5, source length 13, head width 8.
- `epsilon_K=0.3`, prior strength 0.1.
- Key-logit perturbation không vượt 0.3.
- Range của log-odds perturbation tối đa đo được là 0.570547, dưới bound 0.8.
- Tiny vocabulary CE truyền gradient hữu hạn, khác zero tới route parameters.
- Values không có dependency trực tiếp vào route parameters khi giữ input và anchor path cố định.

Đây là sanity check của đề xuất toán học, **không phải test tích hợp AFMR, smoke B200 hoặc evidence tăng điểm**. Chưa có kiểm thử GQA/BF16/resume/cache cho biến thể này vì chưa implement.

## 7. Thiết kế gradient và hiệu năng phải giữ khi thử

Objective ban đầu giữ nguyên:

\[
\mathcal L=\mathcal L_{\mathrm{CE}}.
\]

Không cần positives, oracle labels, CL hoặc teacher để định nghĩa key correction và focus prior. Tín hiệu supervision vẫn là summary reference trong CE khi train; inference chỉ dùng source, instruction và generated prefix.

Đường gradient: `CE → decoder → A → K correction / focus prior`; đồng thời `CE → decoder → V0 → final encoder state`. Không đặt `detach()` ở route để giả lập “bảo vệ content”. Bảo vệ ở đây đến từ bypass topology, không phải cắt mọi gradient encoder.

Giữ zero-init của các output factors để bắt đầu từ anchor. Do zero-init, một số lớp sâu bên trong branch hoặc gates có thể có zero gradient ở bước đầu; cần kiểm tra chúng nhận gradient sau khi output factor bắt đầu khác zero, không đòi mọi parameter nonzero-gradient ngay lần backward đầu tiên.

Hiệu năng dự kiến theo graph, cần profile thực tế:

- Encoder và controller chạy một lần cho mỗi input.
- Tạo `K0`, `K*`, correction, rồi cache **một** `K` và **một** `V0` cho từng decoder layer.
- Không tính lại depth/multi-scale branch ở mỗi output token.
- Decode vẫn có một SDPA cross-attention call/layer, greedy và self-KV cache như trước.
- Chuẩn bị source KV đắt hơn vì tính hai key projections/norm paths. Training có thêm activation; persistent KV cache có thể giữ nguyên shape, nhưng không đồng nghĩa tổng VRAM hoặc thời gian bằng nhau.
- Không tạo attention map đầy đủ chỉ để tính cap. Cap key-side dùng q-norm bound nên chuẩn bị được trước decode.

Trước full run phải pass: anchor identity, masking/padding, zero/one-content-token cases, dimension mismatch, GQA head mapping, BF16 bound tolerance, CE gradient, cached/full greedy parity, EOS compaction, batch-size invariance trong sai số hợp lý, checkpoint round-trip và eval resume. Không thể nhận checkpoint shared-memory cũ rồi im lặng đổi value path như thể cùng kiến trúc.

## 8. Thí nghiệm cần có để câu chuyện không chỉ là lý thuyết

### 8.1. Xác nhận failure mode trước

AFMR chưa có kết quả full mới trong dữ liệu người dùng cung cấp. Không gọi AFMR thất bại dựa vào số của PCEB/dualbridge. Các mốc PubMed trước đó là:

| Bản do người dùng cung cấp | R1 | R2 | RL |
|---|---:|---:|---:|
| T5Gemma2-1B-1B | 49.580 | 21.990 | 45.463 |
| EviSeq PPLX cũ | 48.976 | 21.482 | 45.078 |
| EviSeq_new trước PCEB | 49.228 | 21.644 | 45.377 |
| PCEB | 49.231 | 21.583 | 45.312 |
| Dualbridge | 49.176 | 21.590 | 45.283 |

Các số là user-reported ROUGE-155, chưa kiểm toán raw predictions ở phiên này. Mức tăng/giảm vài phần mười chưa tự chứng minh nguyên nhân kiến trúc hoặc significance. Bản EviSeq_new trước PCEB còn thiếu 0.352/0.346/0.086 điểm; không cộng một “gain kỳ vọng” vào các số này để dự báo AFMR.

Trên validation, lấy một tập cố định để phân loại lỗi: bỏ sót fact, sai entity/number/negation, lặp, quá ngắn, và fact nằm ngoài input truncation. Nếu lỗi chủ đạo là truncation hoặc max output length, tách K/V không giải quyết được.

Can thiệp tắt prior hoặc đổi tạm V của checkpoint đang có chỉ dùng chẩn đoán độ nhạy. Nó có thể là off-distribution intervention; không thay thế ablation retrain có kiểm soát.

### 8.2. Ma trận kiểm chứng gọn

| ID | Variant | Câu hỏi |
|---|---|---|
| B0 | Final-state anchor + copied cross-attention, không AFMR | Interface tối thiểu đã mạnh tới đâu? |
| B1 | AFMR hiện tại, shared adapted input cho K/V | Reference thực tế cần vượt |
| B2 | Value anchor, key adaptation như mục 5 nhưng không norm cap | Tách đường values có giúp không? |
| B3 | B2 + key-logit bound | Bound có lợi thêm hay chỉ làm thiếu capacity? |
| C1 | Key-only MLP từ final state, cùng ngân sách adapter B3 | Có thực sự cần multi-depth? |
| C2 | Value-only adaptation, keys từ anchor | Có phải chính values mới cần adaptation? |
| C3 | B3 bỏ focus prior | Multi-scale prior có đóng góp thực chất? |

Không bắt buộc full-run bảy variant ngay. Giai đoạn đầu profile và pilot B1/B2/B3 trên cùng train subset và fixed validation, tránh chọn theo training CE đơn lẻ. Nếu B2/B3 không có tín hiệu, dừng hướng này; không tiếp tục thêm loss để bảo vệ hypothesis.

Sau pilot có triển vọng, chạy các đối chứng mạnh và full run cùng budget. Cap chỉ lựa chọn trên validation với grid nhỏ đã định trước; không có lý do lý thuyết nào chứng minh 0.2, 0.5 hay 1.0 là tối ưu. Các threshold chấp nhận overhead cũng phải được đặt trước, chẳng hạn tối đa 10% tăng time/step và decode latency; đây là tiêu chí engineering đề xuất, không phải số đo.

### 8.3. Fairness và độ tổng quát

- Giữ instruction nội dung tương đương T5Gemma, greedy, cùng dữ liệu và preprocessing ROUGE-155. Cùng max tokens chưa chắc cùng lượng text vì tokenizer khác; báo source truncation và output words.
- Theo yêu cầu hiện tại, đánh giá `last` sau cùng số epoch đã định. Nếu thêm kết quả `best`, công bố riêng và không chọn test checkpoint theo điểm test.
- Cố định protocol trên validation. Vì test PubMed đã được dùng phản hồi nhiều vòng, cần ghi nhận adaptive development; thêm dataset held-out khác để kiểm tra khả năng tổng quát.
- Nếu mục tiêu là summarization tổng quát, PubMed và arXiv chưa đủ khác miền. Ưu tiên thêm CNN/DM; nếu claim vượt ra ngoài văn bản tài liệu, cần dữ liệu dạng khác như dialogue.
- Có ít nhất hai cặp backbone nếu claim interface modular; tách ảnh hưởng pretrained encoder từ ảnh hưởng bridge.
- So sánh paired bootstrap khi có per-example predictions; multi-seed nếu đủ budget. Báo total parameters và trainable parameters, không chỉ đếm hai backbone hoặc active parameters.
- Báo R1/R2/RL cùng length ratio, repetition, đánh giá factual consistency trên subset, time/step, docs/s, generated tokens/s, peak VRAM. Attention mass/cross_res không phải bằng chứng nội dung được dùng đúng.

Vượt 1/3 ROUGE là mục tiêu thực nghiệm của người dùng, không phải điều kiện đủ về đóng góp khoa học hay nhận paper. Nếu chỉ thắng RL rất nhỏ nhưng R2 giảm và factuality kém, phải trình bày đúng trade-off.

## 9. Câu chuyện paper nên thống nhất như thế nào?

### 9.1. Thesis

> Summarization with independently pretrained backbones requires adapting source access, but this need not require the same interface to modify the value representations that carry source content.

Đây là câu đặt vấn đề và hypothesis. Chỉ nâng thành kết luận khi B1/B2/B3/C1/C2 có evidence.

### 9.2. Dàn ý Introduction theo một chuỗi nguyên nhân

1. **Bài toán.** Có nhiều pretrained encoder/LLM hữu ích, nhưng ghép chúng cho summarization cần một interface task-level có chi phí hợp lý. Cite warm-start và T5Gemma, thừa nhận prior art.
2. **Thách thức cụ thể.** Summarization vừa phải chọn ít thông tin, vừa giữ đúng nội dung được chọn. Shared nonlinear adapter ghép thay đổi attention với thay đổi value payload. Trình bày như một khả năng gây trade-off, không như lỗi đã chứng minh của mọi model.
3. **Insight.** Descriptor để quyết định đọc ở đâu có thể khác representation dùng để truyền nội dung. Dẫn KV-MemNN, MUDDFormer và LR-AttnRes; giải thích chính xác mình kế thừa và giới hạn ở đâu.
4. **Thiết kế.** Một source-token table đầy đủ, value anchor, depth/space/focus adaptation phục vụ retrieval, và logit perturbation bound. Không đổi native backbone self-attention, không cần oracle selection khi infer.
5. **Evidence.** Đưa kết quả quality–cost và controlled ablation thực sự có. Nếu chưa có thì để placeholder, không đưa score EviSeq cũ vào hàng AFMR mới.

### 9.3. Ba contribution có thể viết sau khi đủ kết quả

- Phân tích tách retrieval redistribution và value rewriting ở interface, kèm chẩn đoán thực nghiệm cho trường hợp ghép pretrained models.
- Một interface value-anchored có giới hạn tác động attention rõ ràng, giữ source-token alignment và cached greedy computation.
- Đánh giá có kiểm soát cho quality, factuality proxy và cost trên nhiều loại summarization/backbone, giải thích khi nào constraint giúp hoặc hại.

Không tự nhận contribution thứ ba là “extensive experiments” nếu chỉ có một PubMed run. Không tự nhận cross-attention mới từ đầu; phần cải tiến là interface constraint/topology và bằng chứng về tác dụng.

### 9.4. Phân bổ nội dung cho bản thảo 8 trang main

Đây là kế hoạch bố cục theo mục tiêu 8 trang của người dùng, không phải xác nhận quy định venue hiện hành:

- Introduction và contributions: khoảng 1 trang.
- Related work đặt đúng các đối thủ gần nhất: khoảng 0.75 trang.
- Interface formulation, proposed design, một proposition có điều kiện: khoảng 2 trang.
- Setup và protocol: khoảng 1 trang.
- Kết quả chính, ablation và quality–cost: khoảng 2 trang.
- Phân tích/failure cases, limitations, conclusion: khoảng 1.25 trang.

Chi tiết proof, toàn bộ hyperparameters, additional cases và bảng dài có thể để appendix. Kết quả chính và đối chứng chứng minh novelty không nên chỉ nằm trong appendix. Không lấp trang bằng citation không liên quan.

## 10. Reviewer sẽ phản biện gì?

| Câu hỏi | Trả lời được hiện tại | Evidence còn thiếu |
|---|---|---|
| “Chẳng phải attention đã có K/V riêng?” | Có. Khác biệt là adapter tác động bất đối xứng trước projection và có value bypass | C1/C2, parameter-matched comparisons |
| “Đã có MUDDFormer/LR-AttnRes rồi?” | Có; không claim phát minh decoupled routing. Scope là bounded cross-backbone source interface | Bằng chứng constrained topology có ích ngoài generic multiway adapter |
| “Tại sao phải bảo vệ values, chúng đã tốt chưa?” | Chưa biết. Value anchor là inductive-bias hypothesis | Retrained B1/B2/C2 và factual error analysis |
| “Bound bảo đảm summary đúng?” | Không. Nó chỉ giới hạn local attention perturbation | Không thay bằng guarantee toàn model; cần factuality evaluation |
| “Sao không để model tự học gate nhỏ?” | Gate nhỏ không chặn norm của branch. Cap logit là constraint khác | B2/B3 để kiểm tra constraint có thực sự đáng dùng |
| “Prompt-conditioned có thật không?” | Main instruction có trong encoder prefix, nhưng prompt branch riêng đang không biến thiên | Không claim prompt controllability nếu chưa làm controlled request tests |
| “Thêm constraint có giảm expressivity?” | Có thể có, đó là trade-off cố ý | Dataset/backbone robustness, cap sensitivity |
| “Inference lại chậm?” | Có thể giữ một cached K/V read; prefill và training thêm compute | B200 measured throughput/VRAM, không suy đoán từ FLOPs |
| “Cải thiện có phải do dài hơn?” | Chưa có kết quả để trả lời | Length statistics và validation-controlled length comparison |
| “Vài phần mười ROUGE có đáng kể?” | Không kết luận từ aggregate một seed | Paired comparison và seed variability |

## 11. Claim–evidence map và bước tiếp theo

| Claim | Evidence | Trạng thái |
|---|---|---|
| AFMR hiện tại dùng cùng adapted memory cho K và V | `afmr.py` và `decoder.py` | Đã xác minh code |
| Bounded gate hiện tại không tự bảo đảm bounded residual norm | Công thức gate nhân trainable branch | Đúng về toán |
| Proposed value bypass loại đường ảnh hưởng trực tiếp của route lên V | Graph và đẳng thức mục 6 | Đúng theo thiết kế đề xuất, chưa implement |
| Key cap cho upper bound lên logit/odds perturbation | Điều kiện q-norm, Cauchy–Schwarz, softmax odds; 100 toy checks | Có phân tích và sanity check, chưa tích hợp |
| Routing/value coupling làm AFMR giảm ROUGE | Chưa có controlled ablation | Chưa được chứng minh |
| Proposed variant nhanh như AFMR | Chỉ cùng shape persistent KV và số attention reads | Cần profiling |
| Proposed variant tăng ROUGE hoặc vượt T5Gemma | Chưa có training/eval | Không được claim |
| Novelty đủ cho NAACL | Không có tiêu chí máy móc hoặc bảo đảm nhận | Cần kết quả và review hoàn chỉnh |

Bước tiếp theo được khuyến nghị: giữ nguyên run AFMR hiện hành; thu validation/error profile; nếu triển khai variant thì chỉ thay topology K/V trước, kiểm thử độc lập, sau đó ablate cap. Không sửa nhiều hyperparameters, đổi loss và đổi architecture trong cùng một run.

Kết luận cuối: một thiết kế ít module hơn nhưng trả lời rõ **cần thích nghi phần nào, bảo toàn phần nào, và giới hạn ảnh hưởng ra sao** có câu chuyện khoa học tốt hơn việc liên tục thêm block theo report mới. Hướng value-anchored routing đáng thử vì bám vào code hiện tại, có đối chứng và điều kiện bác bỏ rõ, nhưng chưa có quyền được gọi là tốt hơn trước khi chạy thí nghiệm.
