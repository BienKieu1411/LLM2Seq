# Reward độ dài tận dụng giao diện encoder-decoder

Trạng thái: thiết kế và phản biện, chưa tích hợp runtime hoặc kiểm thử model. Tiếp nối [hai prompt phân vai](2026-10-17_dual_prompt_update.md).

## Quyết định

Giữ reward chính đo **ROUGE và sai số số từ theo đúng chỉ tiêu**. Bổ sung một thí nghiệm reward phía kiến trúc: cùng candidate, cùng yêu cầu W và decoder prompt, so likelihood của **teacher SEAM SFT đóng băng** khi nhận nguồn đúng và nguồn đối chứng. Gọi đây là **source discrimination**, không gọi là factuality hoặc coverage đã được chứng minh.

Phần source discrimination chỉ được bật sau kiểm tra offline, vì nó tăng compute và có thể thưởng copying/teacher bias. Reward baseline vẫn là đối chứng bắt buộc. Không thưởng trực tiếp attention, copy gate hoặc vector norm của actor đang học.

## 1. Độ dài cần khớp chỉ tiêu thực tế

Ký hiệu W là requested words, n là số từ candidate theo evaluator NFKC Unicode. B là ngân sách sai số **tính bằng từ**, được khai báo trước, không phải trần token sinh.

```text
D_len = abs(n - W) / B
Q     = 0.25*R1_F1 + 0.50*R2_F1 + 0.25*RL_F1
```

Trọng số Q là giả thuyết ưu tiên R2, phải so với trọng số bằng nhau. Không thay evaluator Perl bằng proxy training.

Với các ngưỡng người dùng đã quy định, B có thể là WikiLingua 43, Vietnews 38, XLSum 40, LegalText 72, Book 59, ViM 98, VLSP2022 34. Khi W đúng bằng reference word count và dùng cùng aggregation/counting, `mean(D_len)<=1` tương đương đạt ngưỡng MAE của dataset đó. Nếu W do người dùng truyền khác reference, đây là constraint theo request, không còn tương đương MAE với reference.

Đây là sửa đổi so với chỉ phạt `abs(n-W)/W`: cùng tỷ lệ sai 10% có nghĩa khác nhau ở summary 30 và 450 từ, trong khi benchmark đang quy định MAE tuyệt đối. Không dùng tolerance dead band nếu muốn tối ưu chính xác MAE này. Signed over/under vẫn được log riêng; mặc định chi phí hai phía bằng nhau vì MAE đối xứng.

Domain/nhóm không biết thì dùng B cố định do config/request khai báo. Không tự suy ra nhãn domain, không dùng kết quả test để chọn B hay lambda. Nếu đổi metric sang length score bất đối xứng thì đó là thí nghiệm khác.

## 2. Phần riêng cho giao diện encoder-decoder

Teacher T cố định là full SEAM SFT, gồm encoder, bridge, decoder/cross-attention và copy nếu bật. Nó không thuộc optimizer. Chọn nguồn đối chứng x_neg khác tài liệu, cùng ngôn ngữ/domain và gần độ dài input; giữ nguyên W, hai instruction, candidate token IDs và hard cap K.

```text
ell_true = mean_active_token log p_T(y_t | y_<t, x,     P_enc(W), P_dec(W))
ell_neg  = mean_active_token log p_T(y_t | y_<t, x_neg, P_enc(W), P_dec(W))
delta    = ell_true - ell_neg
S_src    = tanh(delta / tau)
```

`tau>0` được cố định sau calibration trên train/probe split, không tính lại theo candidate group. Teacher dùng temperature=1, dropout tắt và full mixture likelihood. Mean theo active decoder action tokens, gồm EOS theo contract scorer, để tránh raw log-likelihood ratio chỉ lớn vì sinh nhiều token. Đây là đơn vị subword cho likelihood; word count cho length vẫn riêng.

Điểm tận dụng kiến trúc là **thay đổi đường thông tin encoder/bridge trong khi giữ nguyên decoder context**. Nếu candidate chỉ là câu mẫu phổ biến không phân biệt nguồn, margin có thể nhỏ. Nếu candidate phù hợp nguồn đúng hơn nguồn khác, margin có thể tăng. Điều đó không chứng minh mọi claim trong candidate là đúng: tên riêng/chi tiết hiếm hoặc copying cũng có thể tạo margin lớn.

Phải tái tạo cả memory, bias, value memory và copy state từ x_neg. Chỉ thay cross-attention memory mà vẫn giữ copy state nguồn đúng sẽ làm phép đối chứng sai. Tránh all-zero/all-masked memory trong lần thử đầu: nó có thể ngoài phân phối và kích hoạt generate-only fallback của copy.

Nguồn đối chứng chọn từ train, có ID/hash khác, loại near duplicates, giữ mapping seed/version để tái hiện. Batch size 1 vẫn dùng được một reservoir train cố định; không lấy nhầm source của test làm negative. Matching quá dễ khiến score chủ yếu đo topic; nhiều negative tốn thêm compute. Không giả định một negative là đủ tốt khi chưa kiểm tra.

## 3. Công thức thí nghiệm tổng hợp

```text
R_i      = Q_i + eta*S_src_i - lambda_d*D_len_i
L_total  = L_GRPO(R) + 0.05*L_gold_CE
```

`eta=0` là baseline; `eta>0` là nhánh thử nghiệm sau diagnostic gate. Có thể thử mức nhỏ như 0.05, nhưng đây không phải giá trị được paper chứng minh. S_src bị chặn trong [-1,1]; không để source term lấn át mục tiêu ROUGE. Hệ số CE cũng cần kiểm chứng, không được coi là bảo đảm câu tự nhiên.

Pilot giữ lambda cố định. Nếu triển khai dual theo domain/nhóm:

```text
lambda_d <- clip(lambda_d + dual_lr*(global_mean(D_len | d) - 1), 0, lambda_max)
```

Cập nhật một lần sau optimizer step bằng global sum/count, giữ lambda cố định trong rollout/group/scoring và lưu state checkpoint. Không có domain trong một step thì không cập nhật domain đó. Constraint trên train không bảo đảm constraint validation/test được thỏa.

GRPO vẫn lấy advantage theo nhóm cùng source/W. Nếu mọi candidate có cùng length cost, cost bị triệt tiêu khi centering, dù tất cả đều sai length. Cũng tương tự với một source score gần như không phân biệt candidate. Log dispersion của từng thành phần; tăng lambda/eta không sửa được thành phần hằng trong group. CE khớp instruction và rollout diversity là các nguồn tín hiệu cần kiểm tra.

Đây là sequence reward. Không chia thành token rewards rồi khẳng định tương đương TRL GRPO hiện tại; token credit cần một objective và kiểm chứng riêng.

## 4. Cơ sở paper và phần tự đề xuất

- [CMDP summarization](https://aclanthology.org/2021.tacl-1.72/): cơ sở tách content reward và requirement costs. Công thức MAE/B và SEAM GRPO ở trên là adaptation.
- [ROUGESal](https://aclanthology.org/N18-2102/): tiền lệ content-aware reward và RL/CE mixing; paper không định nghĩa reward SEAM này.
- [Source coverage reward](https://aclanthology.org/2022.acl-long.351/): tiền lệ dùng nguồn trong reward và cảnh báo scale/ROUGE trade-off. Không mặc định full-source recall là mục tiêu tốt cho summary ngắn.
- [CPMI](https://aclanthology.org/2022.emnlp-main.399/): so source-conditional với target-only likelihood ở decoding. **Wrong-source teacher margin ở đây không phải CPMI, không phải mutual information được ước lượng đúng, và chưa được paper đó kiểm chứng như GRPO reward.**
- [TRAC 2026](https://aclanthology.org/2026.findings-acl.1873/): có length/content token-level reward và PRM theo abstract; không đưa PRM vào recipe này khi chưa đọc/kiểm chứng full method.

Không có ba nguồn độc lập xác nhận công thức tổng hợp mới là hiệu quả cho SEAM. Chỉ các thành phần có tiền lệ; hiệu quả kết hợp là giả thuyết thực nghiệm.

## 5. Vì sao không lấy attention/encoder similarity làm reward ngay?

Source_bias là prior học được, không phải phân phối attention đã quan sát. Native cross-attention dùng SDPA và không trả weights. Tăng raw attention, copy probability hay norm có thể chỉ là model thay cơ chế nội bộ để tự lấy thưởng.

[Attention is not Explanation](https://aclanthology.org/N19-1357/) đưa counterexamples, còn [phản biện](https://aclanthology.org/D19-1002/) cho thấy cần test theo model và định nghĩa. Không dùng một trong hai paper để khẳng định mọi attention đều vô nghĩa hoặc đều đáng tin.

Frozen encoder similarity ổn định hơn actor self-reward nhưng chưa định nghĩa được “bao nhiêu fact ứng với W từ”. Nếu encoder đang fine-tune cùng actor, `detach()` chỉ ngắt gradient trong forward hiện tại, không làm assessor bất biến qua optimizer updates. Snapshot cố định mới tránh được reward drift kiểu đó.

Không mặc định dùng cosine similarity với toàn source: nó dễ đo topicality, không kiểm tra phủ định/số liệu, và forcing full-source recall xung đột với summary ngắn. Semantic coverage nên là diagnostic trước, chỉ chuyển thành reward sau khi có evidence units và calibration đáng tin.

## 6. Chi phí và đường gradient

Teacher không sinh thêm summary: chấm lại G candidate đã có bằng teacher forcing. Tuy vậy nó thêm **hai lượt decoder scoring mỗi candidate**, cùng encoder/bridge cho hai nguồn. Có thể reuse H/M/copy state giữa G candidate cùng source/W trong no-grad reward path; decoder vẫn phải score từng candidate. Đây chưa phải tối ưu đã có trong code.

Nếu KL reference đã được load, có thể dùng chung trọng số teacher; chỉ reuse correct-source log-probs khi thật sự cùng teacher, prompt, token mask, temperature=1 và preprocessing. Config hiện beta=0 thì không có frozen reference; cần thêm full teacher hoặc resource-aware diagnostic, không gọi đó là reward miễn phí. Không tự bật KL chỉ để tạo teacher.

Actor loss vẫn recompute source/bridge với autograd. Reward teacher hoàn toàn `no_grad`; không backprop reward qua teacher, không dùng detached teacher memory thay actor memory. vLLM chỉ rollout; sync full actor sau update và hủy cache phụ thuộc trọng số. Không cần teacher trong inference sản phẩm sau training.

## 7. Diagnostic gate trước khi bật eta

Chấm cùng source/W trên các candidate: reference/paraphrase tốt, câu mẫu chung, summary tráo source, summary sai tên/số/phủ định, bản copy dài và bản bị cắt. Không sinh nhãn mới để SFT; dùng dữ liệu kiểm tra được con người hoặc quy tắc xác định và ghi rõ perturbation.

Yêu cầu: margin không chỉ phản ánh độ dài, tên riêng hoặc overlap; thứ hạng không ưu tiên copying/sai fact hơn summary tốt; tín hiệu không gần hằng trong group. Nếu không đạt, giữ eta=0, không cố tăng trọng số để ép metric có tác dụng.

So ba cấu hình trên cùng dual-prompt SEAM: (A) content+length baseline, (B) thêm source-discrimination term sau gate, (C) B nhưng đóng băng encoder actor để kiểm tra lợi ích update encoder. C là đối chứng kiến trúc, không phải model decoder-only. Theo dõi ROUGE, requested/reference MAE, cap rate, source-margin dispersion, wall-time và VRAM; không chọn theo reward tổng một mình.

## Kết luận phản biện

Ưu thế hữu ích của encoder-decoder là có giao diện nguồn có thể can thiệp/chấm đối chứng độc lập với decoder prompt, và gradient actor có đường đi về encoder. Reward vẫn cần được đánh giá bằng chất lượng đầu ra. Decoder-only cũng có thể dùng source-conditional reward; không claim tính độc quyền cho kiến trúc.

Khuyến nghị: dùng MAE-normalized length cost trong baseline; source-swap teacher margin là nhánh custom reward đáng thử **sau diagnostic**, không phải mặc định đã được xác nhận tốt hơn. Chưa triển khai source reward, teacher loading hoặc dual update mới trong runtime.
