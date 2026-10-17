# Cập nhật: tận dụng encoder, bridge và decoder để điều khiển độ dài

**Đề xuất chính được sửa thành conditioning ở cả hai phía, phân công hai nhiệm vụ khác nhau.** Encoder chuẩn bị biểu diễn nguồn theo độ dài cần viết; bridge điều chỉnh cách đưa thông tin nguồn sang decoder; decoder diễn đạt bản tóm tắt theo yêu cầu số từ. Decoder-prompt-only trở thành đối chứng, không còn là phương án chính duy nhất.

Đây là thiết kế nghiên cứu, chưa tích hợp runtime. Báo cáo đầu [2026-10-17_design.md](2026-10-17_design.md) giữ lại như snapshot trước khi người dùng yêu cầu tận dụng cả hai prompt. Phần dưới thay thế khuyến nghị “encoder prefix chung, thử decoder prompt trước”.

## 1. Cơ sở của việc chia nhiệm vụ

[CTRLsum, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.396/) kết hợp guidance phía encoder với prefix decoder; đối chứng cho các tác vụ điều khiển nội dung ủng hộ việc phối hợp hai phía. Length control trong paper dùng bin và số keyword, không phải yêu cầu chính xác W từ. [GSum, NAACL 2021](https://aclanthology.org/2021.naacl-main.384/) cũng tách guidance và nguồn, nhưng dùng thêm encoder/cross-attention riêng. Không sao chép cấu trúc đó.

[LAAM, ACL 2022](https://aclanthology.org/2022.acl-long.474/) là tiền lệ sát hơn cho việc điều khiển thông tin được đọc theo ngân sách: cơ chế cụ thể sửa cross-attention, không phải prompt encoder. Paper có pretraining riêng; ta chỉ lấy nguyên tắc budget-aware source access. [Nguyen et al.](https://www.techscience.com/iasc/v38n1/55289/html) đã đưa length embeddings vào cả encoder và decoder. **Đưa độ dài vào hai phía không thể tự nhận là ý tưởng hoàn toàn mới.**

Các nguồn ủng hộ hướng thiết kế chung; không nguồn nào chứng minh cặp prompt tiếng Việt dưới đây vượt SEAM hiện tại. Nguồn độc lập chủ yếu là nghiên cứu học thuật, cộng kiểm tra code local và code tác giả; chưa có ba loại bằng chứng độc lập chứng minh tính ưu việt của SEAM.

## 2. Hai prompt cụ thể

Encoder nhận instruction ngắn trước source:

```text
Đọc văn bản để chuẩn bị bản tóm tắt khoảng {target_words} từ.
Ưu tiên các ý chính và chi tiết cần thiết để diễn đạt chính xác.
Văn bản:
{source}
```

Decoder nhận instruction khác, cùng một W:

```text
Viết bản tóm tắt khoảng {target_words} từ dựa trên văn bản đã cung cấp.
Giữ các thông tin quan trọng, viết rõ ràng và chỉ trả về bản tóm tắt.
```

Decoder prefix có thể giữ `Bản tóm tắt:\n`, tách khỏi instruction. Không cần đưa source thô sang decoder. Tên biến là thiết kế interface dự kiến; code hiện chưa render `{target_words}` theo sample.

Điểm khác nhau về vai trò: encoder được hướng dẫn **đọc/ưu tiên thông tin**, decoder được hướng dẫn **viết và tuân thủ độ dài**. Cùng W nhưng không cần lặp nguyên văn mọi quy tắc. Không yêu cầu encoder sinh danh sách ý, không sinh reasoning, không có summary trung gian hoặc nhãn do model tự tạo. [Entity-planning](https://aclanthology.org/2021.tacl-1.88/) là tiền lệ cho tách lựa chọn nội dung và diễn đạt, nhưng cách sinh entity chain của paper đó không được đưa vào đây.

Nếu backbone encoder cần task marker theo model card, giữ marker đã được kiểm chứng của adapter rồi thêm instruction này. Không bịa special token `summarize`, không áp chat template decoder cho mọi embedding encoder. Các prompt trên cần kiểm nghiệm; embedding backbone chưa được bảo đảm hiểu được instruction chọn nội dung chỉ nhờ zero-shot prompting.

## 3. Tác dụng phải xuất hiện ở ba vị trí

Với x là source, W là số từ yêu cầu, K là hard token cap:

```text
H_W       = Encoder([P_encoder(W); x])
c_W       = Controller(pool_content(H_W), pool_embed(P_decoder(W)), K)
(M_W,b_W) = Bridge(H_W, c_W)
y         ~ Decoder(P_decoder(W), M_W, b_W), hard_cap=K
```

Ở graph hiện tại, controller nhận **embedding token của prompt decoder**, không phải hidden states đã contextualize qua decoder. Không nên viết rằng bridge đã đọc hiểu toàn bộ instruction bằng một decoder riêng.

- Encoder: W xuất hiện trước source, nên có thể tác động lên biểu diễn token nguồn.
- Bridge: pooled source đã phụ thuộc W, đồng thời nhận embedding prompt decoder. Control state đi vào depth/feature gates và focus prior. Ta dùng đúng đường này trước khi thêm module mới.
- Decoder: nhận cả instruction đầu ra và memory/bias đã được chuẩn bị theo cùng yêu cầu.

Đây là tận dụng khả năng **điều kiện hóa riêng quá trình biểu diễn nguồn và quá trình sinh**, với bridge làm giao diện học được. Không kết luận decoder-only không làm được điều khiển tương tự; decoder-only cũng có thể nhận yêu cầu và tái sử dụng prefix KV. Chưa có kết quả so sánh để gọi ưu thế này là tăng điểm đã được chứng minh.

### Vì sao chưa thêm MLP độ dài ngay?

W đã có đường vào H, vào c và vào decoder. Thêm MLP mới đồng thời sẽ làm khó tách tác dụng của encoder prompt. Nhánh số học riêng vẫn có giá trị nếu model kém tuân thủ chữ số hoặc bridge ít nhạy với W. Khi đó thêm đặc trưng word-count cạnh nhánh token-budget hiện tại; **không đổi đơn vị của `output_budget`**. Thiết kế này vẫn tận dụng bridge mà không bắt buộc tăng số module trước khi có bằng chứng.

### Source có bị rút gọn không?

Không hard-select hoặc xóa source theo W trong thiết kế này. Bridge vẫn giữ các vị trí nguồn hợp lệ; focus prior chỉ thay đổi ưu tiên. Nhờ vậy candidate dài có thể tìm thêm chi tiết trong nguồn đã mã hóa. Điều đó không bảo đảm phủ ý tốt và cũng không làm chi phí cross-attention giảm theo W: memory vẫn dài như cũ.

Encoder causal: instruction phải ở trước source để các token nguồn có thể nhìn thấy W. Không lấy hidden của prefix đầu chuỗi làm “kế hoạch toàn văn” vì dưới causal mask nó chưa nhìn thấy phần source phía sau. Với encoder bidirectional, giữ đúng attention behavior của backbone; không ép mọi encoder sang một kiểu mask để hỗ trợ prompt.

## 4. Liên hệ với GRPO và full-actor update

Mỗi nhóm dùng cùng `(source, W, encoder_prompt, decoder_prompt, actor_version)` và sinh G summary. GRPO dùng chất lượng nội dung cùng chi phí over/under; CE nhỏ tiếp tục dùng nhãn thật. Reward và word-counter giữ thiết kế ở [báo cáo đầu](2026-10-17_design.md#5-reward-tăng-nội-dung-trong-khi-giảm-sai-số-độ-dài).

Reward không cần khả vi. Gradient của log-probability đi từ token decoder qua cross-attention/copy, bridge và encoder. Full-actor update mới cho encoder học cách thay đổi H theo prompt; chỉ đồng bộ decoder sau GRPO sẽ làm rollout không còn cùng policy với scorer.

**Giới hạn của group-relative learning:** nếu G candidate đều có cùng length cost, `cost_i - mean(cost_group)=0`; phần phạt độ dài không đóng góp vào advantage, dù tất cả đều quá dài hoặc quá ngắn. Không khắc phục được chỉ bằng tăng lambda. Log cả cost dispersion và tỷ lệ group không có tín hiệu phân biệt length. Matched-instruction CE/warm-up cung cấp tín hiệu ngoài group advantage; rollout cần đủ đa dạng. Nếu thay sampler để tăng đa dạng, vẫn phải giữ đúng behavior log-prob và kiểm tra chi phí. Không tự thêm absolute baseline rồi gọi đó là GRPO nguyên bản.

Khi dùng gold CE, W phải khớp độ dài của nhãn thật sau policy xử lý truncation đã công bố. Không randomize yêu cầu ngắn/dài rồi tính nguyên CE trên một nhãn duy nhất không khớp. Inference lấy W từ người dùng. Nếu dùng độ dài gold test cho benchmark phải công bố điều kiện đó và cấp cùng thông tin cho mọi đối chứng.

Chỉ huấn luyện trên W lấy từ gold không bảo đảm điều khiển các W phản thực tế trên cùng source. Khi đánh giá yêu cầu ngắn/vừa/dài, phải phân biệt mức trong phân phối train và mức ngoài phân phối, đồng thời kiểm tra hard cap có khả thi. Nếu sau này thêm các request ngoài độ dài gold vào RL, mask CE cho các request không khớp; không chế nhãn summary mới bằng cách cắt gold giữa câu.

Với R2, giả thuyết là source access theo ngân sách giúp chọn các chi tiết/quan hệ có thể diễn đạt trọn vẹn trong W, thay vì nhồi nội dung rồi dừng vì cap. **Không có loss mới bảo đảm model giữ được quan hệ hoặc tự tạo một plan**, và việc điều khiển đúng độ dài không đủ để bảo đảm R2 tăng. Vẫn phải kiểm tra R1/RL và sai lệch độ dài riêng.

## 5. Tận dụng encoder-decoder để giảm chi phí rollout

Trong `no_grad` rollout, với cùng x, W và prompts, H/M/b và copy state là phần không phụ thuộc candidate. Có thể tính một lần rồi dùng cho G decoder continuations. Mỗi candidate vẫn có decoder KV riêng và trạng thái sinh riêng.

Đây là cơ hội tối ưu, **chưa phải tính năng được xác nhận trong GRPO hiện tại**: caller đang repeat source G lần và vLLM dùng `n=1` mỗi row. Không bật prefix caching chung để thay thế: copy state và memory cần được phục hồi đúng.

Khi W nằm trong encoder prompt, khác W làm encoder input thay đổi, không được reuse H chỉ theo source. Cùng W nhưng encoder prefix/model/tokenizer/source span/actor version khác cũng phải coi là request khác. Actor đổi trọng số thì cache encoder/bridge cũ mất hiệu lực.

Khi chấm differentiable log-probs, phải recompute encoder/bridge với autograd hoặc chia sẻ một graph còn sống đúng vòng backward. Không dùng detached rollout memory thay cho encoder trong actor loss. Tối ưu graph sharing giữa score microbatch là việc riêng, không đánh đổi độ đúng gradient để tiết kiệm encoder pass.

## 6. Điều kiện triển khai bắt buộc

1. `target_words` là một field có kiểu, giữ xuyên qua schema, TRL sampling, sharding, group check và output metadata. Hai prompt render từ **cùng field**, không lấy riêng hai config có thể lệch nhau.
2. Tách ngân sách source content và chi phí prompt. Hiện prefix+source cùng dùng `max_source_length`. Khi thêm prompt, không âm thầm lấy mất thêm đoạn cuối source. Với đối chứng, giữ cùng visible source text/span và kiểm tra tổng context thật của encoder. Không thể giữ số token source cũ nếu encoder đã chạm giới hạn context mà không dành chỗ riêng; phải ghi nhận giới hạn đó.
3. Prefix encoder thay đổi theo record: content-mask và copy offsets dùng đúng chiều dài ký tự prefix **của record**, kể cả ký tự Unicode và task marker. Chỉ source có thể được copy; instruction không được coi là evidence.
4. Giữ instruction encoder trong attention validity nhưng tách khỏi source content mask. Hiện prefix vẫn là memory có thể được decoder attend tới; đó là control metadata, không phải thông tin từ tài liệu.
5. Prompt decoder theo W có thể khác width. Native rollout/scoring hiện yêu cầu prompt không padding cùng width: cần bucketing hoặc một đường padding/position IDs được kiểm chứng, không chỉ sửa template YAML.
6. Native, gold CE, old/current/reference scorer, vLLM và validation phải dùng cùng text renderer, prefix thực tế, prompt IDs, source span và token caps. Serving có per-request encoder prefix nhưng GRPO convenience caller hiện chưa mang đủ biến theo sample.
7. Cache key phải phân biệt rendered encoder prompt và decoder prompt, word-count value, source span, tokenizer/model identity và actor version. Không bật cache nếu copy state chưa có vòng đời đúng.
8. Kiểm tra behavior/current likelihood trước update trên 1 GPU và 2 GPU; sync full actor, group không bị chia giữa ranks và không reuse cache qua policy update.

## 7. Đối chứng tối thiểu

SFT checkpoint hiện tại là điểm tham chiếu. Ba biến thể sau cùng dùng GRPO+gold CE, cùng reward/caps/data/updates:

- **D:** encoder instruction chung; chỉ decoder nhận W. Bridge vẫn nhận prompt decoder như kiến trúc hiện tại.
- **E:** encoder nhận W; decoder nhận instruction tóm tắt chung. W chỉ đến decoder qua memory/bridge.
- **ED, phương án chính:** encoder và decoder nhận cùng W theo hai vai trò trên.

D so với ED đo phần thêm của source-side instruction. E so với ED đo việc nhắc yêu cầu trực tiếp ở decoder. Không gọi D là “decoder-only model”: nó vẫn là SEAM encoder-decoder. Đối chứng LLM decoder-only là một so sánh riêng và cũng phải nhận cùng W.

Sau đó, chỉ khi có lý do từ kết quả, so ED với ED+numeric-word-feature của bridge. Không thêm đồng thời entropy target, hard pruning, plan generator hay một model semantic judge.

Giữ cùng visible source, công bố prompt tokens và compute thực tế; “cùng số update” không tự có nghĩa cùng FLOPs. Không chọn checkpoint chỉ theo số từ hoặc một joint scalar. Dùng validation để chọn các checkpoint thỏa nội dung/độ dài, giữ test để báo cáo.

Chẩn đoán cần có: độ dài sinh khi cùng source được yêu cầu W ngắn/vừa/dài, MAE với request, R1/R2/RL, cap/EOS rate, và thay đổi source_bias/attention kèm chất lượng. **Attention thay đổi không chứng minh đã chọn đúng thông tin**, không đặt mục tiêu cơ học “W ngắn thì entropy phải thấp hơn” làm tiêu chí thành công.

## 8. Tự phản biện

**Hai prompt chỉ tăng số lần nhắc?** Có thể. Vì vậy kiểm tra ED với D/E, giữ phần source giống nhau; chưa gọi đây là module planning mới.

**Embedding encoder có làm theo instruction không?** Chưa được bảo đảm. Cần full fine-tuning và kiểm tra độ nhạy H/score theo W, không suy luận từ model card retrieval sang summarization.

**Chọn ít nội dung có làm R1/RL tụt?** Có thể, kể cả vẫn giữ full memory. Đó là lý do không hard-prune, không áp entropy constraint, và phải đánh giá từng metric.

**Có ưu thế tốc độ chắc chắn so với decoder-only không?** Không. Reuse encoder theo group là tối ưu có thể kiểm chứng trong SEAM; decoder-only cũng có KV reuse. Đo wall-time, token throughput và VRAM trên cùng GPU trước khi claim.

## Kết luận thiết kế

Phương án chính là **dual-role prompts + bridge hiện có + GRPO content/length + gold CE**, với reuse encoder trong rollout như hạng mục tối ưu độc lập. Mục tiêu là học **đọc theo độ dài cần viết và viết theo nội dung đã đọc**, không chỉ học dừng sớm. Ưu thế về điểm số và việc có cần thêm word feature vào bridge vẫn là hai câu hỏi thực nghiệm.
