# EviSeq: hướng thử tiếp từ foundation model và world model

Ngày đối chiếu: 2026-09-07. Code nền local: `a1ce056`. Đây là nghiên cứu và đề xuất có điều kiện, không phải kết luận về run đang chạy trên server. Chưa có checkpoint/predictions của run đó trong workspace để xác định nguyên nhân thất bại.

Cập nhật triển khai theo yêu cầu tiếp theo: ứng viên shared source read ở mục 10 đã được triển khai trong [src/eviseq_update](../src/eviseq_update/README.md), package riêng với config bật mặc định. 115 local tests đã qua; offline smoke chạy hai training stages, checkpoint và evaluation với nhánh mới bật. Chưa có đánh giá ROUGE/GPU của bản update. Các ứng viên khác vẫn là đề xuất nghiên cứu.

Cập nhật sau kết quả người dùng cung cấp: EviSeq đạt R1/R2/RL = 49.626/21.901/45.895. Target mới đề xuất là ít nhất +1 điểm tuyệt đối ở cả ba metric so với mốc T5Gemma 49.580/21.990/45.463, cùng protocol. Mục 10 mô tả ứng viên nhỏ từ phân tích graph: dùng lượt đọc của copy head để condition trực tiếp nhánh LM. Sau yêu cầu mở rộng nguồn tham khảo, mục 11–14 bổ sung khảo sát, hai ứng viên kiến trúc khác và thứ tự thử có điều kiện. Chưa xác nhận dataset/checkpoint/resolved config của điểm số vừa cung cấp; config PubMed local chỉ là căn cứ thiết kế, không phải bằng chứng về run server.

Đã đối chiếu ba bản tổng hợp có sẵn: `ARCHITECTURE_REVIEW.md`, `AFMR_IMPLEMENTATION_PLAN.md`, `AFMR_NOVELTY_RESEARCH.md`. Không đọc lại toàn bộ năm PDF technical report cũ. Các nguồn bổ sung bên dưới được kiểm tra trên paper gốc; phần chuyển giao sang EviSeq là giả thuyết của báo cáo này.

## 1. Quyết định đề xuất

Nếu run hiện tại chưa đạt mục tiêu, ưu tiên xác định nó thất bại ở đâu, sau đó thử một thay đổi. Sau khảo sát mở rộng: nếu nguồn nhìn thấy đã đủ, thử source read ở output head trước; nếu thiếu ý trong nguồn đã đọc, xét đọc phân cấp; nếu mất ý do truncation, ưu tiên mở rộng nguồn. Value calibration, adaptation objective và state là các nhánh có điều kiện, không phải danh sách module cần cộng dồn.

Foundation models ghép encoder và LM là nguồn tham khảo trực tiếp hơn world models điều khiển robot. World models cung cấp ý tưởng về biểu diễn và trạng thái; kết quả robotics/video không chứng minh gain ROUGE cho summarization. Không có cơ sở để hứa một kiến trúc mới chắc chắn vượt T5Gemma.

## 2. Những điểm cụ thể trong graph hiện tại

- `modeling/afmr.py`: nếu encoder và decoder cùng hidden width, `base_projection` là `Identity`. AFMR tạo memory M bằng depth/feature residual, còn value memory H0 lấy từ final encoder state qua base projection.
- `modeling/decoder.py`: mỗi layer vẫn có learned `memory_norm`, `k_proj`, `v_proj`, `o_proj`; K nhận M và V nhận H0. Không được mô tả V là không có khả năng học alignment. Thiếu gì, nếu có, là một phép thích nghi phi tuyến dùng chung trước value projections; chưa biết nó có cần thiết không.
- Cùng hidden size không đồng nghĩa cùng hệ tọa độ ngữ nghĩa. Đồng thời, khác hệ tọa độ không tự chứng minh implementation hiện tại sai: các projection và backbone đều có thể thích nghi trong full fine-tuning.
- Source prior được tính một lần cho document. Attention QK và copy query thay đổi theo decoder hidden state, nên model đã có truy xuất theo prefix. Điểm chưa explicit là một state riêng theo dõi nguồn đã dùng; decoder self-attention vốn đã lưu thông tin prefix.
- Copy có context encoder, lexical embedding, decoder query và gate. Nó không chỉ đếm token nguồn. Tuy nhiên, không có ràng buộc bảo đảm giữ đúng quan hệ tác nhân–kết quả hoặc phủ đủ các ý.
- Main objective là token mixture CE. Một epoch interface warm-up hiện tại vẫn là học source→summary, không phải pretraining denoising tổng quát của interface.

Một số mô tả trong report ngày 05/09 là lịch sử: prompt decoder hiện đã có chat instruction và code hiện đã triển khai value anchor/copy. Không dùng nguyên report cũ làm mô tả graph đang chạy.

## 3. Các kiến trúc bổ sung và mức độ liên quan

| Nguồn | Cơ chế đã được paper mô tả | Ý tưởng có thể kiểm tra trong EviSeq | Giới hạn |
|---|---|---|---|
| [Flamingo, NeurIPS 2022](https://arxiv.org/html/2204.14198v2), §2.1–2.2 | Perceiver Resampler; chèn gated cross-attention/dense giữa các layer LM đóng băng; tanh gate scalar khởi tạo 0 | Đối chứng lịch mở interface và mức bảo toàn decoder pretrained | EviSeq đã có cross gate. Đổi sigmoid sang tanh không tự tạo đóng góp mới; không suy ra cần nén toàn bộ source |
| [BLIP-2, ICML 2023](https://arxiv.org/html/2301.12597v3), §3 | Q-Former kết nối frozen encoder/LLM; hai giai đoạn representation và generative learning | Học interface thành một nhiệm vụ rõ ràng trước khi full fine-tune | Bản gốc dùng 32 learned queries, Q-Former 188M tham số và nhiều objective. Không bê nguyên bottleneck vào PubMed |
| [LLaVA-1.5, CVPR 2024](https://arxiv.org/html/2310.03744v2), §3.3 | Connector MLP hai layer cùng recipe dữ liệu/alignment | Đối chứng connector phi tuyến đơn giản; kiểm tra vai trò của value calibration | Vision-language là miền khác; EviSeq đã có MLP residual ở K path, không phải chưa có MLP |
| [V-JEPA 2, 2025](https://arxiv.org/html/2506.09985v1), §2–3 | Dự đoán latent của vùng video bị mask với target EMA/stop-gradient; hậu huấn luyện predictor theo action | Học representation chứa thông tin cần dự đoán, thay vì chỉ tối ưu một vector pooled theo chủ đề | Không có bằng chứng trực tiếp cho token-level summarization; thêm target encoder/loss tăng chi phí |
| [VL-JEPA, bản v2 2026](https://arxiv.org/html/2512.10942v2), §2 | X-encoder, predictor theo query, Y-encoder và Y-decoder; dự đoán embedding đích, train với InfoNCE | Một predictor cho nội dung câu kế tiếp, decoder vẫn diễn đạt bằng token | Bản gốc không phải CE-only. Không được nói chỉ gắn MSE lên controller là đã triển khai VL-JEPA |
| [Large Concept Models, 2024](https://arxiv.org/html/2412.08821v1), §2, §3.1.2 | Dự đoán sentence embeddings trong SONAR rồi giải mã thành text; có thử summarization | Điều khiển ở cấp câu, đồng thời giữ đường token cho tên riêng/số liệu | Paper báo hạn chế fluency theo CoLA; không có đảm bảo latent tốt sẽ cho câu tự nhiên |
| [DreamerV3, bản v2 2024](https://arxiv.org/html/2301.04104v2), Learning algorithm | RSSM có recurrent state và stochastic latent; actor/critic học từ imagined trajectories | State explicit biểu diễn tiến độ sử dụng thông tin nguồn | Decoder đã có lịch sử; phải chứng minh state mới thêm chức năng. Không có action/reward environment tương ứng sẵn trong EviSeq |
| [LeWorldModel, preprint 2026](https://arxiv.org/html/2603.19312v1), §2 | Predictor latent theo action; MSE + SIGReg chống collapse, không dùng EMA/stop-gradient | Nếu nghiên cứu latent predictor, kiểm tra collapse và mục tiêu biểu diễn ngay từ đầu | Gaussian prior chưa có bằng chứng phù hợp với biomedical semantics; không phải một regularizer nên bật mặc định |
| [TD-MPC2, ICLR 2024](https://arxiv.org/abs/2310.16828) | Learned latent dynamics và tối ưu trajectory cho continuous control | Bài học về đánh giá rollout nhiều bước | Không ưu tiên cho CE-only greedy summarization: planning/search và reward model làm đổi bài toán và ngân sách |

T5Gemma 2 đã có trong report cũ nên không tính là nguồn mới, nhưng là đối chứng quan trọng: paper mô tả adaptation với UL2 và continued pretraining quy mô lớn, cùng tied embeddings và merged attention. Vì vậy, phần chênh lệch với EviSeq có thể đến từ sự chuẩn bị của backbone/interface trước fine-tuning, không chỉ từ bridge. Đây là khả năng cần kiểm tra, không phải nguyên nhân đã xác định. [T5Gemma 2, §3 và Conclusion](https://arxiv.org/html/2512.14856v1).

## 4. Trước khi sửa: xác định failure mode

Chốt metric chính và protocol trước vòng thử tiếp. Nếu đã nhìn test để chọn hướng sửa, ghi nhận test-guided development; dùng validation cho các quyết định tiếp theo. Epoch 3 thua một điểm số riêng lẻ chưa đủ xác định nguyên nhân hoặc kết luận toàn bộ kiến trúc thất bại.

| Quan sát | Chẩn đoán cần làm | Hướng sửa có cơ sở hơn |
|---|---|---|
| Python ROUGE thấp nhưng Perl tương đối ổn | Chấm lại cùng predictions/reference với preprocessing chung | Sửa protocol/formatting, chưa cần thêm module |
| Tóm tắt thiếu nội dung nằm ngoài source đã truncate | Kiểm tra visible text và vị trí evidence của mẫu lỗi | Sửa input coverage; bridge không truy cập được token chưa encode |
| Train CE giảm, validation CE/generation xấu dần | Xem đường học, độ dài, độ trôi chảy và chênh lệch train/val | Regularization/LR/stage schedule trước khi tăng capacity |
| Đổi source sang document khác nhưng CE/output ít đổi | Source-swap diagnostic, cả khi copy hoạt động và khi copy bị vô hiệu hóa có kiểm soát | Kiểm tra interface/source conditioning, value calibration hoặc adaptation objective |
| Token/thuật ngữ đúng hơn nhưng câu gượng hoặc lặp | Copy gate, posterior copy responsibility, output length và đọc mẫu mù | Đối chứng train copy-off; nếu có lặp nguồn rõ thì thử coverage |
| Tên/số đúng nhưng sai nhóm điều trị, hướng kết quả, quan hệ giữa các câu | Annotate lỗi role binding, evidence support và độ phủ ý | Thử state theo câu/nguồn sau khi xác nhận lỗi không do truncation |
| Train chậm/OOM nhưng chất lượng chấp nhận được | Profile actual kernels, allocations, CE/copy head và source length | Tối ưu chỗ đo được; resampler chỉ khi chi phí memory thực sự là nút thắt |

Source-swap: giữ decoder prefix và labels của document A, đổi toàn bộ tensor nguồn sang B; với copy phải đổi cả token IDs và alignment theo B. Tính delta CE trên cùng target tokens. Đây là can thiệp ngoài phân phối để đo độ nhạy, không phải benchmark chất lượng hoặc loss mới; teacher forcing có thể che sự phụ thuộc nguồn, nên xem thêm generation trên mẫu nhỏ. Không có threshold delta CE phổ quát để kết luận model bỏ qua source.

Đừng chỉ log giá trị gate. Với cross-attention, xem norm của phần residual sau gate so với residual stream. Với copy, trọng số g không phải tỷ lệ token thực sự được copy. Có thể đo trách nhiệm của nhánh copy với một token v:

`responsibility(v) = g P_copy(v) / ((1-g) P_LM(v) + g P_copy(v))`.

Với reference token, đây là diagnostic likelihood; với generated token, đây vẫn là phân bổ xác suất, không phải truy vết một quyết định copy rời rạc. Không suy ra attention mass là chứng cứ faithful explanation.

## 5. Ứng viên có điều kiện: kiểm tra value calibration nhỏ

Điều kiện: value-anchored run thua shared-memory control hoặc source conditioning yếu, sau khi loại trừ lỗi dữ liệu/optimization. Đây là hypothesis về expressive power của interface, không phải sửa gradient.

Giữ K từ AFMR M. Thử một adapter V độc lập, dùng chung cho các decoder layers:

```text
H0 = base_projection(H_final)
Hv = H0 + alpha * W_up SiLU(W_down RMSNorm(H0))
K_l = key_norm_l(Wk_l memory_norm_l(M))
V_l = Wv_l memory_norm_l(Hv)
```

Khởi tạo W_up bằng 0, alpha dương nhỏ. Đây là cách bắt đầu từ graph value-anchor hiện tại: backward đầu W_up có thể học, rồi gradient tới W_down ở các bước sau. Không khởi tạo đồng thời alpha=0 và W_up=0 vì hai thừa số 0 có thể khóa nhánh. Một gate nhỏ vẫn không đảm bảo norm adapter luôn nhỏ; cần đo norm hoặc định nghĩa clipping nếu mục tiêu là bound thật.

Với width 1024 và rank 128, hai matrix có 262.144 tham số, chưa kể norm/gate. Tính Hv một lần cho source, tiếp tục dùng K/V cache một lần mỗi layer. Vẫn phải đo VRAM và latency, không tuyên bố chi phí bằng 0.

Để tách biến: lần đầu giữ input copy-context là H0 như cũ, chỉ sửa value stream của decoder. Nếu sau này cho copy nhận Hv, đó là ablation tiếp theo. Retrieval residual AFMR vẫn không trực tiếp sửa Hv, nhưng giá trị không còn là final-state anchor thuần như main graph hiện tại.

Khác với Wv_l có sẵn: adapter thêm biến đổi phi tuyến trước phép chiếu và chia sẻ giữa các layer. Nếu nó không vượt shared-memory hoặc một connector đơn giản cùng số tham số, không có lý do giữ thêm module.

Nguồn cảm hứng là learned connector của LLaVA/BLIP-2; công thức cụ thể trên là đề xuất cho EviSeq, không phải kiến trúc hay kết quả được hai paper đó xác nhận.

## 6. Thay đổi cách học: adaptation bằng denoising từ train source

Điều kiện: nhiều biến thể bridge đều conditioning yếu hoặc underfit; có ngân sách thử training recipe. Lấy ý tưởng giai đoạn alignment của foundation models và denoising adaptation của T5Gemma, nhưng thử ở quy mô phù hợp dự án.

Pilot có thể dùng source của train split: che một span/câu và yêu cầu khôi phục từ context còn nhìn thấy, rồi fine-tune source→summary. Nguồn bị che phải bị loại khỏi encoder input và copy alignment; target không được len vào copy memory. Có thể dùng text delimiter đã hỗ trợ bởi tokenizer, không bắt buộc thêm sentinel vocabulary.

Hai đối chứng phải rõ: (a) tổng số update hoặc GPU-hours được giữ cố định, dành một phần cho denoising thay vì summarization; (b) nếu thêm update thì baseline cũng được cấp ngân sách tương ứng. Không lấy dữ liệu validation/test vào adaptation. Vẫn là token CE ở từng giai đoạn, nhưng đã đổi training recipe và phân phối dữ liệu; không được gọi protocol hoàn toàn giống run hiện tại.

Không gộp value adapter và denoising trong thí nghiệm đầu tiên. Nếu cả hai thay đổi cùng lúc, không biết lợi ích đến từ kiến trúc hay thêm tín hiệu học.

## 7. Nếu lỗi là lặp/thiếu ý: state explicit có kiểm soát

Ý tưởng có thể mượn từ Dreamer là state thay đổi theo lịch sử, nhưng phiên bản gần summarization nhất là coverage. Coverage đã có tiền lệ trực tiếp ở [Pointer-Generator Networks, ACL 2017](https://aclanthology.org/P17-1099/); không nên đổi tên nó thành world model để claim novelty.

Thử nhỏ nhất: lưu mức sử dụng các vị trí/vùng nguồn trước bước sinh hiện tại và dùng nó điều chỉnh scores của copy head. Giữ source memory đầy đủ và LM branch. Trạng thái này có thể làm giảm việc đọc lại nguồn, nhưng phải cho phép một vùng cung cấp nhiều fact; phạt coverage quá mạnh có thể làm mất thông tin đúng.

Thiết kế phải causal: state tại bước t chỉ dùng các bước trước t. Nếu update từ attention đã điều chỉnh, output head sẽ cần recurrent scan theo thời gian; không thể tùy tiện dùng cumsum của attention tương lai hoặc detach state trong train rồi gọi là cùng graph. Cache state phải được chọn lại khi finished-row compaction và có test teacher-forcing/cached parity.

Nếu muốn state đó sửa bias bên trong decoder, không thể lấy final decoder hidden của chính bước t rồi đưa ngược vào các layer trước trong cùng forward. Cần causal staging hoặc thêm lượt tính; phải báo chi phí. Vì vậy thử output-head coverage trước sẽ ít xáo trộn hơn, dù nó không tự giải quyết mọi lỗi quan hệ.

Chỉ giữ nếu giảm lặp hoặc tăng coverage có ý nghĩa trên validation mà không làm factuality/fluency xấu đi. Attention entropy thấp hơn tự nó không phải thành công.

## 8. Hướng nghiên cứu sâu hơn: semantic predictor theo câu

VL-JEPA/LCM gợi ý học “nội dung câu kế tiếp” rồi để LM diễn đạt. Một thiết kế cho EviSeq có thể là:

```text
full source memory + previous generated sentences
    → predictor → predicted semantic state z_k
    → token decoder conditioned on source and z_k
```

Reference sentence có thể được encode làm target phụ khi train, nhưng decoder phải nhận z_k được predictor dự đoán, không được nhận embedding câu reference mà inference không có. Predictor trước câu k chỉ thấy các câu trước k; không pooling future decoder states. Khi inference, prefix của predictor đến từ câu model thực sự sinh. Phải chấp nhận và đo exposure bias, không gọi teacher forcing là rò rỉ chỉ vì dùng các câu trước trong train.

Nếu thêm latent matching, đây là nhánh thí nghiệm thay đổi ràng buộc CE-only hiện tại. Frozen target encoder/EMA và stop-gradient là lựa chọn thiết kế có chủ đích, khác với ngắt nhầm gradient trên generation path. VL-JEPA gốc dùng InfoNCE; LeWM dùng SIGReg; không trộn cả hai chỉ để tránh collapse. Với target embeddings học được, cần kiểm tra variance/effective rank và khả năng phân biệt cặp fact tối thiểu: tăng/giảm, drug/placebo, số 12/21.

Vẫn giữ CE và đường source-token để bảo toàn tên/số. Similarity embedding có thể đánh giá gần nhau hai câu chỉ khác một con số hoặc phủ định; loss latent đẹp không thay thế kiểm tra factuality. LCM có đánh giá summarization nhưng không mặc nhiên tốt hơn về fluency. Tách tầng ngữ nghĩa và diễn đạt là hypothesis; chưa có gain PubMed trong dự án này.

Hướng này cần thêm module, có thể thêm encoder lúc train, cache/state ở cấp câu và chi phí train tuần tự hơn. Xếp sau các thử nghiệm nhỏ, chỉ làm nếu mục tiêu chuyển sang nghiên cứu content planning và ngân sách cho phép.

## 9. Thứ tự thí nghiệm và quy tắc giữ/bỏ

1. Để run hiện tại hoàn tất; lưu đúng config/checkpoint/predictions. Không đổi architecture của checkpoint bằng sửa YAML.
2. Chấm Perl ROUGE chung, xem phân phối độ dài và đọc mù một tập validation cố định. Chọn mẫu cả ngẫu nhiên lẫn nhóm lỗi; báo riêng hai nhóm, không dùng nhóm chọn lỗi để ước lượng error rate toàn dataset.
3. Dựng các control cùng backbone và training recipe: connector đơn giản; AFMR shared-memory LM-only; value-anchor LM-only; value-anchor + copy. Có thể chạy pilot trên train subset cố định trước khi full-run, nhưng không đổi subset giữa variants.
4. Chọn đúng một nhánh theo bảng failure mode. Với copy, force g=0 trên checkpoint là diagnostic; muốn kết luận tác dụng kiến trúc cần bản được train copy-off từ đầu. Tương tự can thiệp residual bằng 0 không thay thế trained ablation.
5. Giữ thay đổi nếu metric chính validation và loại lỗi nhắm tới cải thiện, rồi kiểm tra lại trên seed khác nếu có thể. Paired bootstrap đo uncertainty theo mẫu; không thay thế uncertainty do seed training.
6. Báo trade-off trên cả R1/R2/RL, fluency/factuality, parameters, GPU-hours và inference cost. Không chọn một metric thuận lợi sau khi đã nhìn kết quả để tuyên bố thắng.

Value calibration nhỏ vẫn là ứng viên nếu diagnostic/control cho thấy value anchor hạn chế model. Sau kết quả cập nhật, ứng viên thử trước được cụ thể hóa ở mục 10. Nếu lỗi nổi trội là copy lặp hoặc role binding thì chuyển hướng tương ứng; không dùng cùng một bản sửa cho mọi thất bại. World model đầy đủ, Q-Former nén toàn source, diffusion decoder hoặc actor–critic không phải bước mặc định tiếp theo.

## 10. Ứng viên cho target mới: dùng cùng lượt đọc nguồn cho sinh và copy

### 10.1. Điểm cần kiểm chứng trong graph

Trong `GroundedCopyHead.loss`, LM logits được tính từ `lm_head(states[valid])`. Copy attention tạo xác suất token nguồn và context của gate; context đó chưa được đưa trực tiếp vào LM head. Decoder hidden đã được condition bởi cross-attention trong backbone, nên không được gọi nhánh LM là không đọc nguồn. Đề xuất này bổ sung một cạnh từ **chính lượt đọc của output copy head** tới nhánh LM.

Với target token y không có trong các copy token đủ điều kiện:

```text
P_copy(y) = 0
L(y) = -log P_LM(y | h) - log(1-g)
```

Giữ h và source features cố định để phân tích đường cục bộ: copy alignment nhận gradient qua g, nhưng nhánh vocabulary likelihood chưa có đường trực tiếp qua alignment. Tín hiệu `-log(1-g)` khuyến khích bớt copy; nó không phân biệt token không-copyable y nào cần được diễn đạt từ nguồn. Khi gate weights đang khởi tạo 0, gradient từ gate về copy query/keys bằng 0 ở những vị trí này. Khi gate học, gradient gián tiếp này có thể khác 0; đây không phải bug detach hoặc toàn bộ encoder mất gradient.

Giả thuyết: để copy attention đồng thời đọc source context cho LM sẽ cung cấp tín hiệu lexical generation cho việc chọn nguồn, kể cả khi phải paraphrase. Nó có thể tăng độ thống nhất giữa nội dung đọc và từ được sinh. Không suy ra R2 thấp hiện tại do nguyên nhân này, và không có lower bound cho gain ROUGE.

### 10.2. Computational graph đề xuất

Giữ encoder, AFMR, decoder cross-attention và phân phối copy a_t hiện tại. Thêm semantic values có rank r, căn theo cùng decoder-source tokens bằng sparse overlap alignment sẵn có:

```text
U_j = overlap_pool_j(W_value RMSNorm(H0))
a_t = existing_copy_attention(h_t, keys, source_bias, mask)
c_t = sum_j a_tj U_j
beta_t = sigmoid(w_beta [q_t ; RMSNorm(c_t)] + b_beta)
h'_t = h_t + beta_t W_out RMSNorm(c_t)
P_LM = softmax(lm_head(h'_t))
P(y_t=v) = (1-g_t) P_LM(v) + g_t sum_{j: token_j=v} a_tj
L = token CE of this mixture
```

Giữ g_t và các input của nó như cũ trong ablation đầu. beta_t điều khiển semantic residual, g_t điều khiển xác suất copy; không buộc hai gate bằng nhau. Chỉ tính a_t một lần và dùng cho cả hai output paths. Context U_j từ source representation, không chỉ từ lexical token embedding. Nhờ vậy, cùng token xuất hiện ở hai vị trí có thể có value khác nhau theo ngữ cảnh encoder; việc chọn đúng vị trí vẫn phải được học.

Ví dụ mục tiêu: nguồn nói “did not significantly reduce mortality”, model có thể học cách viết “no significant mortality reduction” bằng LM branch đồng thời giữ tên thuốc/số liệu bằng copy. Đây là ví dụ minh họa mong muốn, không phải prediction đã quan sát hoặc bảo đảm xử lý phủ định.

### 10.3. Initialization, gradient và chi phí

- Với hidden width 1024, rank r=128: W_value và W_out có tổng 262.144 tham số; thêm gate/norm vẫn xấp xỉ 0,263M. Không tăng decoder vocabulary/head kích thước lớn.
- Khởi tạo W_out bằng 0, beta ở giá trị dương nhỏ như 0,05; W_value khởi tạo bình thường. Lúc khởi tạo h'=h nên output giữ nguyên về toán học. Giữ RNG của modules chung khi tạo module mới để so sánh initialization công bằng.
- Backward đầu W_out có thể học; gradient vào W_value/beta và vào a_t qua nhánh mới bắt đầu sau khi W_out khác 0. Không khởi tạo đồng thời output factor và một multiplicative gate bằng 0.
- CE đi qua LM head → h' → c_t → a_t và U_j. Không detach alignment/semantic values trong main training. Vẫn có CE qua copy mixture và qua backbone như trước.
- Khi không có source token đủ điều kiện, ép semantic residual bằng 0 và dùng LM fallback chính xác. Không chỉ dựa vào softmax trên một hàng bị mask toàn bộ.
- U_j được chuẩn bị một lần mỗi source; thêm một tensor [B,J,r] và một phép attention-value multiplication. Chi phí thêm khoảng O(B T J r), không có attention score thứ hai nhưng cũng không phải miễn phí.
- Training giữ chunked CE và chỉ materialize vocabulary logits ở supervised positions. Phải sửa đồng bộ dense path, chunked path và inference; CopyState cần thêm values và hỗ trợ row selection khi compaction. CUDA/BF16 và peak VRAM phải đo ở batch thực tế.
- Không thêm recurrence, lookahead hoặc label-dependent alignment. Với teacher forcing, h_t chỉ thấy prefix trước target tương ứng; inference dùng chính prefix đã sinh. Giữ một encoder pass và một decoder pass.

### 10.4. Probe toán học đã chạy

Đã chạy bằng PyTorch với GroundedCopyHead hiện tại và một LM head tuyến tính ngẫu nhiên, B=2, T=3, source length=5, hidden=24, rank=8. Toàn bộ target ID 31 không nằm trong source IDs 4–8. Decoder hidden và source keys được coi là input độc lập; gate ở initialization để cô lập đường generation về copy alignment.

| Kiểm tra | Kết quả |
|---|---:|
| Head hiện tại: norm gradient copy query | 0 |
| Head hiện tại: norm gradient source keys | 0 |
| Context-read branch đang hoạt động: norm gradient copy query | 0.022294 |
| Context-read branch đang hoạt động: norm gradient source keys | 0.002972 |
| Context-read branch đang hoạt động: norm gradient semantic values | 0.006863 |
| W_out=0: chênh lệch loss với head hiện tại | 0 |

Probe chỉ chứng minh một khác biệt của đường gradient và initialization trên tensor nhỏ. Không phải huấn luyện EviSeq, không xác nhận GPU performance, không chứng minh sửa được factuality hoặc đạt target ROUGE. Probe này được thực hiện trước bước triển khai `eviseq_update`; các kiểm thử implementation được ghi trong README của bản đó.

### 10.5. Prior art và thí nghiệm quyết định

Đưa source context vào vocabulary distribution đã có trong pointer-generator gốc, §2.1–2.2, equations 3–4; không claim đó là cơ chế mới. Generalized Pointer Generator cũng nghiên cứu giới hạn của exact-copy và việc edit token được trỏ tới. Đề xuất hiện tại là cách tích hợp source read với pretrained LM head của EviSeq bằng residual nhỏ, không phải bản triển khai nguyên paper nào. [See et al., ACL 2017](https://arxiv.org/html/1704.04368v2), [Shen et al., EMNLP 2019](https://aclanthology.org/D19-1390/).

Các control nên có: A = graph hiện tại; B = cùng graph cộng semantic read vào LM; C = adapter cùng ngân sách tham số chỉ nhận decoder hidden, để kiểm tra gain có phải do thêm capacity chung. Giữ seed, dữ liệu, số update, prompt, decoding và preprocessing. Chưa ghép value-calibration hoặc coverage vào B.

Nếu B tốt hơn A và C trên validation, xác minh theo nhóm copyable/non-copyable targets, fluency, entity/number/negation errors, độ dài và repetition; sau đó chạy lại seed khác và full protocol. Nếu B chỉ tăng copy rate/độ dài mà chất lượng thực tế không tốt hơn, không kết luận cơ chế đạt mục tiêu. Target số để chốt full evaluation là R1≥50.580, R2≥22.990, RL≥46.463; dùng kiểm định phù hợp cho ba metric đã định trước và phân biệt uncertainty theo mẫu với theo seed. Các vòng chọn cấu hình tiếp theo dùng validation, ghi nhận việc đã xem test hiện tại.

Rủi ro chính: thêm một lượt source read có thể dư thừa với cross-attention hiện có; một attention dùng chung cho exact copy và paraphrase có thể nhận mục tiêu xung đột; context pooled có thể trộn fact không tương thích; residual có thể đẩy LM về văn phong nguồn. Các khả năng này là lý do phải có A/C và xem lỗi thực, không chỉ theo dõi gradient.

## 11. Khảo sát mở rộng ngoài các technical report có sẵn

Phạm vi tra cứu gồm summarization, long context, retrieval/fusion, content planning, học ghép backbone, tối ưu sequence, factuality, recurrent memory và world models. Dùng paper gốc trên ACL Anthology, arXiv, proceedings/publisher và tài liệu tác giả. Có tìm công trình 2025–2026; không lấy ngày crawl của search engine làm ngày xuất bản. Đây là khảo sát có chọn lọc rộng, không phải tuyên bố đã đọc mọi nguồn trên Internet hoặc systematic review hoàn chỉnh.

Mức đọc trong bảng: **M** = đã kiểm tra phần phương pháp liên quan của toàn văn; **T** = đã sàng lọc abstract/mô tả hoặc đoạn phương pháp trên nguồn gốc, chưa audit toàn bộ thí nghiệm/code. M không có nghĩa đã tái lập paper. Bảng gồm 29 công trình, trong đó hai pointer papers đã được dùng ở mục 10. Cộng chín nguồn ở mục 3 và T5Gemma 2 là 39 công trình được đối chiếu ở các mức độ khác nhau; không tính các report cũ vào con số này.

### 11.1. Đọc nguồn, copy và cấu trúc tài liệu

| Công trình | Mức | Điều rút ra để thiết kế | Giới hạn khi chuyển sang EviSeq |
|---|---|---|---|
| [Pointer-Generator, ACL 2017](https://arxiv.org/html/1704.04368v2) | M | Context attention đi vào vocabulary distribution; coverage theo lịch sử | Tiền lệ trực tiếp của mục 10, không chứng minh thêm context sẽ thắng pretrained cross-attention hiện tại |
| [Generalized Pointer Generator, EMNLP 2019](https://aclanthology.org/D19-1390/) | T | Nghiên cứu khả năng biến đổi từ được trỏ tới thay vì chỉ exact-copy | Chưa audit implementation; không đồng nhất phép edit embedding với residual được đề xuất |
| [FAME, ACL 2021](https://aclanthology.org/2021.acl-long.474.pdf) | T | Source-conditioned vocabulary bias có thể hỗ trợ chọn từ đúng chủ đề | AFMR đã có source prior nhưng không phải cùng cơ chế; chủ đề phù hợp chưa bảo đảm quan hệ đúng |
| [LongT5, Findings NAACL 2022](https://arxiv.org/html/2112.07916) | M | Local attention kết hợp global representations; khảo sát độ dài đầu vào, có PubMed | Gain đi cùng độ dài và pretraining; không so trực tiếp điểm paper với run 4K của dự án |
| [PEGASUS-X, 2022](https://arxiv.org/abs/2208.04347) | T | Block-local attention, global tokens và thêm pretraining trên chuỗi dài | Tăng context trong YAML không tái tạo được quá trình adaptation của paper |
| [Fusion-in-Decoder, EACL 2021](https://aclanthology.org/2021.eacl-main.74/) | T | Encode các passage riêng rồi fusion khi decode | Bài gốc là QA. Decoder vẫn đọc nhiều token; không giải quyết mọi chi phí bằng chunking |
| [CachED, TACL 2025](https://aclanthology.org/2025.tacl-1.58.pdf) | M | Chunked encoding + fusion, cache gradient tại encoder output rồi recompute từng chunk | Gradient đúng cho graph chunked, không tương đương full encoder attention. Chưa tái lập trên EviSeq |
| [Top-down/Bottom-up Summarization, 2022](https://arxiv.org/html/2203.07586) | M | Biểu diễn đoạn tương tác toàn cục rồi cập nhật token; có benchmark khoa học | Bản AvgPool phù hợp để tham khảo đơn giản. AdaPool dùng tagger/nhãn từ reference; OracleAdaPool không phải kết quả triển khai được |
| [GSum, NAACL 2021](https://aclanthology.org/2021.naacl-main.384/) | T | Nguồn cùng guidance như câu được chọn có thể giúp generation | Cần predictor guidance và tính chi phí; reference-derived oracle guidance khi test không phải đối chứng hợp lệ |
| [FROST, TACL 2021](https://arxiv.org/html/2104.07606) | M | Cùng decoder sinh entity plan trước summary bằng MLE | Tăng chuỗi sinh, có lỗi plan; đã có tiền lệ rõ cho planning, không gọi latent plan là mới chỉ vì đổi biểu diễn |
| [Entity Coverage Control, Findings NAACL 2022](https://aclanthology.org/2022.findings-naacl.40/) | T | Control code liên quan entity precision, có thử PubMed | Entity có trong nguồn vẫn có thể bị gán sai quan hệ; intermediate Wikipedia training là một thay đổi dữ liệu |
| [Structure Information for Scientific Summarization, 2025](https://arxiv.org/abs/2505.14179) | T | Nhận diện chức năng phần Background/Methods/Results/Discussion | Bản gốc cần classifier/dataset cấu trúc. Không mặc định thêm annotation này vào protocol hiện tại |

### 11.2. Cách học và cách chọn đầu ra

| Công trình | Mức | Điều rút ra để thiết kế | Giới hạn khi chuyển sang EviSeq |
|---|---|---|---|
| [PEGASUS, ICML 2020](https://arxiv.org/abs/1912.08777) | T | Khôi phục các câu quan trọng đã bị lấy khỏi nguồn | Gợi ý adaptation gần summarization; masked target không được còn trong copy memory |
| [PRIMERA, ACL 2022](https://aclanthology.org/2022.acl-long.360/) | T | Pretraining khuyến khích gom thông tin quan trọng giữa nhiều document | PubMed single-document khác setting; không chuyển nguyên Entity Pyramid rồi suy ra gain |
| [BRIO, ACL 2022](https://arxiv.org/html/2203.16804) | M | CE + ranking các candidate bằng chính generator; bản BRIO-Mul giữ khả năng sinh | Đổi objective, cần tạo/chấm candidates. Kết quả paper dùng beam; chưa có bằng chứng gain tương tự với greedy EviSeq |
| [SimCLS, ACL 2021](https://aclanthology.org/2021.acl-short.135/) | T | Generate candidates rồi một model khác chấm/chọn | Thêm reranker và inference compute; không gọi là cùng protocol greedy một lần |
| [SARA, ACL 2025](https://aclanthology.org/2025.acl-long.1236/) | T | Adaptive decoding cân bằng source, salient context và prior | RL điều khiển decoding cùng nhiều phân phối điều kiện; không phải một layer CE-only miễn phí |

BRIO là nguồn mạnh để phản biện giả định “chỉ cần thay kiến trúc”. Trong bảng 2 của paper, baseline BART được tác giả chấm lại đạt 44.29/21.17/41.09 trên CNN/DM, BRIO-Mul đạt 47.78/23.55/44.57. Đây là kết quả của paper trên dataset/recipe đó, không phải ước lượng gain cho PubMed. Nhánh thử BRIO của dự án phải báo riêng thay đổi objective và compute, đồng thời cho T5Gemma cơ hội dùng recipe tương ứng nếu claim lợi thế kiến trúc.

### 11.3. Kiến trúc lớn hơn, hiệu suất và generation nhiều bước

| Công trình | Mức | Điều rút ra để thiết kế | Vì sao chưa ưu tiên cho target hiện tại |
|---|---|---|---|
| [LOCOST, EACL 2024](https://aclanthology.org/2024.eacl-long.69/) | T | SSM encoder-decoder cho long summarization | Paper nêu trade-off chất lượng/tiết kiệm memory; hiệu suất tốt không đồng nghĩa ROUGE cao hơn |
| [Titans, 2025](https://arxiv.org/abs/2501.00663) | T | Neural long-term memory bổ sung attention | Đổi cơ chế memory/học của backbone; không có kết quả fine-tune EviSeq trong nguồn này |
| [Gated DeltaNet, 2024](https://arxiv.org/abs/2412.06464) | T | Gating và delta update cho memory; hybrid với attention | Cần recipe/kernel/backbone phù hợp, không phải thay attention pretrained tùy ý |
| [Gated DeltaNet-2, preprint 2026](https://arxiv.org/abs/2605.22791) | T | Tách erase/write trong linear attention | Thí nghiệm pretraining LM không chứng minh đổi bridge nhỏ sẽ tăng summarization |
| [Discrete Diffusion/CrossMamba, Findings NAACL 2025](https://aclanthology.org/2025.findings-naacl.352/) | T | Noising và backbone phải phù hợp cho conditional generation | Thay decoder/objective/inference. Paper vượt các diffusion baselines không có nghĩa vượt mọi autoregressive model |
| [Context-Aware Hierarchical Merging, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.289/) | T | Giữ nguồn làm căn cứ khi hợp nhất summaries trung gian | Nhiều lượt sinh, có thể tích lũy lỗi; latent pooling không phải cùng pipeline |
| [FRAME/SCOPE, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1094v2.pdf) | T | Fact extraction → selection → organization → writing | Pipeline meeting summarization bằng LLM; thêm nhiều lượt và đổi ngân sách, chưa là bằng chứng cho module nhỏ |

### 11.4. Đo đúng loại lỗi trước khi chọn module

| Công trình | Mức | Cách dùng trong nghiên cứu EviSeq |
|---|---|---|
| [Lost in the Middle, TACL 2024](https://aclanthology.org/2024.tacl-1.9/) | T | Gợi ý chia lỗi theo vị trí evidence. Paper dùng QA/retrieval; không suy ra EviSeq chắc chắn có cùng positional bias |
| [ARC, EACL 2026](https://aclanthology.org/2026.eacl-long.167/) | T | Phân biệt thiếu ý với sai fact; xem coverage theo vai trò nội dung trong legal/scientific documents |
| [QAFactEval, NAACL 2022](https://aclanthology.org/2022.naacl-main.187/) | T | QA-based signal bổ sung kiểm tra source support; chất lượng question/answerability ảnh hưởng metric |
| [SummaC, TACL 2022](https://aclanthology.org/2022.tacl-1.10/) | T | NLI theo cặp câu rồi aggregate, dùng như diagnostic factuality |
| [Stress Testing Factuality Metrics, ACL 2026](https://aclanthology.org/2026.acl-long.1472/) | T | Metric cho short summaries có thể không ổn trên long documents; cần kiểm tra bằng mẫu đọc mù và fact tối thiểu |

Không có metric phụ nào trong bảng tự chứng nhận một bản tóm tắt y sinh là đúng. Đọc mẫu để phân biệt: thiếu kết quả, nhầm nhóm, đảo chiều tăng/giảm, sai số, lặp và câu gượng. Đây là đánh giá mô hình, không phải diễn giải y khoa.

## 12. Hai ứng viên kiến trúc bổ sung sau khảo sát

### 12.1. Đọc nội dung ở cấp vùng rồi cấp token

**Điều kiện:** evidence cần thiết đã nằm trong nguồn nhìn thấy, nhưng model thường chọn nhầm ý/thiếu ý hoặc trộn các vị trí chứa cùng thuật ngữ. Đây là giả thuyết khác với vấn đề exact-copy.

AFMR đã có nhiều cửa sổ nguồn, song source prior của nó được tính một lần từ document controller. QK trong decoder/copy đã thay đổi theo prefix; phần mới cần kiểm chứng là **lựa chọn vùng rõ ràng theo từng bước sinh**, không phải thêm attention động lần đầu.

Pilot chỉ sửa output head, giữ cross-attention backbone. Giả sử token nguồn đủ điều kiện j thuộc một vùng m(j), chia bằng câu hoặc đoạn ngắn; để pilot dễ tái lập có thể dùng block không chồng lấn. Từ source H0 tạo region summaries S_m, rồi dùng h_t đọc vùng và token:

```text
S_m = mean_pool_{j in region m}(source_context_j)
pi_tm = softmax_m(q_region(h_t) dot k_region(S_m) / sqrt(r))
e_tj = q_token(h_t) dot k_token(source_context_j) / sqrt(r)
b_tj = softmax_{j within each region}(e_tj)
a_sem_tj = pi_t,m(j) * b_tj
c_sem_t = sum_j a_sem_tj * v_sem(source_context_j)
h'_t = h_t + beta_t W_out RMSNorm(c_sem_t)
```

Các vùng tạo thành partition của token hợp lệ thì a_sem là một phân phối chuẩn hóa trên nguồn. Mask vùng rỗng trước region softmax; nguồn rỗng dùng LM fallback. Đây là công thức đề xuất của báo cáo, không phải bản chép của LongT5/Top-Down. Hai paper đó cung cấp tiền lệ về granularity, không xác nhận thiết kế output head này.

Trong lần thử đầu, giữ copy attention hiện tại riêng với a_sem để khả năng copy đúng token không bị ép dùng cùng mục tiêu với việc chọn ý để diễn đạt. So với mục 10, đây là biến thể phức tạp hơn và thêm attention scoring; chỉ thử sau khi shared-read chưa đủ hoặc cho thấy xung đột. Muốn biết hierarchy có ích, phải có control **semantic attention phẳng độc lập** cùng rank/ngân sách, không chỉ so với model chưa có semantic head.

Không hard top-k loại token, không cần reference-derived region label, không thêm latent loss hay rollout. CE truyền qua cả hai tầng softmax; các vị trí t có thể tính song song khi teacher forcing. Source values/region keys được cache một lần; h_t chỉ dùng prefix hợp lệ. Chi phí attention thêm O(B T (J+M) r), ngoài projections; tăng cả memory lẫn compute. W_out=0 và beta dương nhỏ giữ output ban đầu, như mục 10.

Rủi ro: region pooling làm mất chi tiết, chia vùng sai ranh giới fact, ưu tiên vùng không đồng nghĩa đúng quan hệ. Phân cấp áp đặt tổng attention mass ở cấp vùng nên có thể làm kém token attention phẳng. Không coi heatmap tập trung hơn là bằng chứng tóm tắt tốt hơn. Nếu thiếu quan hệ giữa các vùng còn rõ sau pilot này, mới xét region self-attention/top-down update; chưa cộng vào pilot đầu.

### 12.2. Encode nhiều đoạn và giữ đầy đủ gradient

**Điều kiện:** lỗi thiếu ý chủ yếu do evidence nằm ngoài source bị truncate. Local `configs/afmr_base.yaml` đặt max_source_length=4096; con số này có cả phần encoder prefix trong giới hạn của pipeline, nên không đồng nghĩa 4096 token nội dung. Cần đối chiếu actual resolved config và text sau preprocessing của run server.

Hướng tham khảo gần nhất là FiD/CachED: encode nhiều đoạn từ một tài liệu, ghép token memories để decoder/copy truy cập. Chưa cần một model đi tìm tài liệu bên ngoài. Cần giữ ánh xạ đoạn/vị trí sang raw source cho copy, tránh mất offset hoặc copy lặp do overlap. Việc đổi full attention sang chunked encoding là một thay đổi biểu diễn cần học, không chỉ một tối ưu memory.

Với EviSeq, không bê nguyên pseudocode chỉ cache final encoder state: AFMR đọc nhiều layer taps. Đặt ranh giới cache tại **mọi tensor encoder mà downstream sử dụng**, gồm taps/final state cần thiết, rồi recompute và vector-Jacobian product đủ tất cả các nhánh. Nếu cùng tensor được dùng hai vai trò phải cộng gradient đúng, tránh đếm hai lần. Giữ RNG/dropout và autocast nhất quán giữa hai forward; optimizer chỉ step sau khi tổng gradient encoder/bridge/decoder/copy hoàn tất. Đây là yêu cầu thiết kế riêng cho graph hiện tại.

Gradient caching có thể detach ở lượt tạm nhưng phải truyền cached gradient trở lại lượt recompute; detach rồi không replay là lỗi. Trước long run, cần so loss và từng nhóm gradient với graph chunked giữ nguyên autograd trên batch nhỏ; so full encoder attention là một câu hỏi kiến trúc khác. Thiết kế production còn cần accumulation, DDP, clipping và checkpoint resume đúng thứ tự; chưa triển khai các phần này.

Chỉ chunk encoder không làm nguồn vô hạn miễn phí: còn memory taps, bridge, cross K/V, copy keys/values và chi phí T×J. Pilot 8K đủ để kiểm tra giả thuyết trước khi xét 16K. Chunking cũng mất contextualization giữa đoạn trong encoder; decoder phải tổng hợp, hoặc một thí nghiệm sau thêm tương tác cấp vùng.

Để đánh giá công bằng, chạy hai câu hỏi tách biệt: cùng vùng nguồn 4K để xem ảnh hưởng chunking; và tăng vùng nguồn 8K với cả EviSeq/T5Gemma được cấp input/compute tương ứng. Cùng con số token giữa hai tokenizer có thể nhìn thấy lượng text khác nhau: cần báo cả giới hạn token lẫn vùng văn bản thực tế. Nếu baseline không chịu được ngân sách đó, báo trade-off tài nguyên/chất lượng thay vì quy toàn bộ gain cho bridge.

## 13. Quyết định và thí nghiệm để tiến tới mức hơn rõ

Mốc làm việc: 50.580/22.990/46.463, tức còn thiếu +0.954/+1.089/+0.568 so với điểm EviSeq được cung cấp. Không có paper nào cho phép dự đoán các gain này bằng cách cộng kết quả từ model khác.

| Thứ tự có điều kiện | Thử nghiệm | Câu hỏi được trả lời | Khi nào giữ |
|---|---|---|---|
| Trước khi train variant | Lưu config/predictions; đọc mẫu validation ngẫu nhiên và nhóm lỗi; đo độ dài/visible evidence | Mất nguồn, chọn ý hay diễn đạt là vấn đề nổi trội? | Có bằng chứng đủ để chọn một nhánh; không chẩn đoán chỉ từ R2 |
| Nguồn đủ, thử nhỏ trước | Current vs shared-read §10 vs hidden-only adapter cùng capacity | Chính source read thêm vào LM có ích không? | Validation generation và loại lỗi mục tiêu cải thiện hơn cả hai controls |
| Evidence bị cắt | Chunk/fusion §12.2: 4K control rồi 8K | Thêm coverage có giải quyết thiếu ý? | Tăng coverage/chất lượng xứng với compute; baseline được đối chiếu cùng điều kiện |
| Evidence đã thấy nhưng chọn ý kém | Hierarchical semantic read §12.1 vs flat semantic read | Granularity vùng có ích ngoài việc thêm một attention? | Lợi ích giữ khi so cùng rank/compute và không làm factuality xấu đi |
| Conditioning yếu qua nhiều variant | Value calibrator §5 hoặc denoising §6, mỗi lần một biến | Representation interface hay cách học bị hạn chế? | Hơn controls với cùng update budget |
| Cho phép đổi objective | BRIO-style CE + candidate ranking trên train | Model đã sinh được candidate tốt nhưng xếp xác suất chưa đúng? | Generator tự sinh tốt hơn ở cùng decoding; tính cả candidate-generation training cost |
| Có lỗi lặp/state rõ | Coverage §7; kế tiếp mới cân nhắc plan §8/FROST | State/plan giải quyết được lỗi nào mà prefix model chưa xử lý? | Giảm lỗi mục tiêu với chi phí và chất lượng được đo |

Không cần train toàn bộ bảng trước khi có quyết định. Chọn một pilot theo lỗi, cấu hình được chọn bằng validation; thất bại của một pilot phải cập nhật giả thuyết. Chỉ ghép hai thay đổi sau khi mỗi thay đổi đã có bằng chứng riêng. Shared-read được ưu tiên vì có thay đổi graph cụ thể và ít chi phí; không có nghĩa nó có mức gain lớn nhất. Nếu truncation chi phối lỗi, mở rộng input được đưa lên trước.

Với nhánh BRIO, dùng candidates sinh từ train sources, ranking dựa trên train references; không tạo preference từ test. Score sequence phải dùng chính log-prob của LM/copy mixture và normalization đã định trước. Giữ CE để bảo toàn khả năng sinh. Không dùng kết quả best-of-k có reference oracle để báo score deployment. Có thể giữ greedy lúc đánh giá để so recipe, nhưng đó là giả thuyết cần kiểm chứng vì kết quả gốc dùng beam.

Pilot một seed là bước sàng lọc, không chứng minh thắng. Bản được chọn cần đánh giá lại ở các seed đã định trước (mục tiêu thực dụng: 3), báo mean/spread và paired uncertainty theo document. Muốn claim hơn rõ theo mốc này: điểm tăng ít nhất 1 ở cả ba metric, bằng chứng chênh lệch dương qua kiểm định/CI đã chọn trước, và không đổi protocol để lấy metric thuận lợi. Uncertainty do sample và do seed phải báo riêng. Đã xem test hiện tại thì ghi nhận test-guided development; các quyết định tiếp theo dùng validation và tránh chọn epoch bằng test.

## 14. Những điều chưa được chứng minh

- Chưa có predictions của run hiện tại trong workspace để xác định nguồn gây ra khoảng cách điểm số; không lấy predictions lịch sử của model/dataset khác thay thế.
- Chưa có ablation train copy-off, shared-read, hierarchy hoặc chunked EviSeq cho run này. Probe mục 10 chỉ kiểm tra đường toán học trên tensor nhỏ.
- Chưa đo latency/VRAM/GPU-hours của các ứng viên; số tham số nhỏ không đồng nghĩa attention theo source dài là rẻ.
- Survey không xác lập novelty. Shared context, hierarchical attention, entity planning và coverage đều có prior art; muốn đóng góp nghiên cứu cần chứng minh một thiết kế/tương tác mới và ablations tương ứng.
- Bản nền `eviseq_new` được giữ nguyên. Shared source read đã có trong `eviseq_update`; những thiết kế phân cấp/long-context/objective khác trong tài liệu chưa được triển khai.
