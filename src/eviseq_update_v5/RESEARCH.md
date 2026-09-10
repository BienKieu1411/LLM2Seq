# Cơ sở nghiên cứu và lựa chọn kiến trúc v5

## 1. Luận điểm nghiên cứu

Giả thuyết chính là **semantic read có thể hữu ích nhưng cách buộc toàn bộ vocabulary prediction đi qua hidden đã sửa có thể tạo đánh đổi**. Điểm v2 cao hơn new ở R2 gợi ý thử giữ read này; chưa đủ kết luận nó là nguyên nhân. V5 kiểm tra giả thuyết bằng một architecture có đường base rõ và semantic branch riêng trong phân phối xác suất.

Đây không phải giả thuyết “ROUGE-L thấp vì thiếu planner”. ROUGE-L liên quan subsequence overlap với reference, không trực tiếp đo quality của content planning. Cũng không có bằng chứng từ ba điểm tổng cho thấy attention noise, hallucination hoặc truncation là nguyên nhân chính.

## 2. Các nguồn gần nhất với quyết định

### 2.1. Mixture of Softmaxes: tiền lệ bắt buộc

Yang và cộng sự phân biệt trộn **probabilities sau softmax** với trộn contexts/logits trước softmax. Paper đưa ra MoS và thử nghiệm language modeling; nó cũng ghi nhận chi phí output tăng. Đây là prior art trực tiếp, không được claim probability mixture là phát minh của v5.[^1]

Áp dụng ở đây có cấu trúc cụ thể: hai representations là cùng `h` và `h+delta_source`, chung vocabulary matrix, chung copy probability, và hệ số semantic có upper bound. Không suy ra EviSeq đã gặp “softmax rank bottleneck”, cũng không dùng kết quả perplexity của RNN để dự báo ROUGE. Control hidden interpolation cùng số tham số là bắt buộc để kiểm tra chính vị trí fusion.

### 2.2. Pointer-generator và generalized pointer

Pointer-generator kết hợp vocabulary distribution và attention mass trên token nguồn; coverage trong paper gốc còn dùng một loss riêng. Generalized Pointer Generator nghiên cứu khả năng sửa token được align thay vì chỉ exact-copy.[^2][^3]

Bài học cho v5 là giữ copy làm một đường lexical riêng, còn nguồn có thể hỗ trợ vocabulary generation theo đường khác. Semantic read độc lập của v2 phù hợp câu hỏi đó. Không sao chép coverage loss, không thêm target aligner và không gọi semantic residual hiện tại là implementation của Generalized Pointer Generator.

### 2.3. Gated Attention, 2025

Qiu và cộng sự thử vị trí/granularity của gating; gating sau SDPA theo head cho kết quả tốt trong các mô hình được pretrain của họ. Paper phân tích nonlinearity và sparsity, với lợi ích phụ thuộc vị trí gate.[^4]

Chuyển giao hợp lý là xác định chính xác gate điều khiển đại lượng nào và liệu normalization có xóa tác dụng đó không. V5 đặt router ở mixture probabilities nên context RMS không thể triệt tiêu tỷ trọng mixture. **Đây không phải gating SDPA của paper**; không mượn claim attention-sink-free hoặc speed overhead của họ cho v5. V4 đã có post-norm head gate; vì vậy thêm gate tương tự lần nữa không phải giải thích mới cho regression.

### 2.4. T5Gemma 2, 2025

T5Gemma 2 khởi tạo từ Gemma 3, tiếp tục pretrain với UL2, dùng tied embeddings và merged self/cross attention. Paper báo một ablation chỉ giữ cross-attention ở global layers làm giảm chất lượng; merged attention là đánh đổi hiệu suất/tham số, không phải thắng tuyệt đối về chất lượng.[^5]

V5 giữ source access ở mọi layer. Không đổi sang merged attention vì Qwen decoder hiện đã ghép với encoder khác; sửa cấu trúc đó sẽ thay nhiều giả định pretrained. Ưu thế của T5Gemma còn có thể nằm ở quá trình adaptation trước PubMed, không chỉ block architecture. Bản thiết kế này không thêm UL2 hay continued pretraining để giải quyết khoảng cách đó.

### 2.5. mHC và bằng chứng ngược khi fine-tuning

mHC ràng buộc residual mixing giữa nhiều streams bằng cấu trúc doubly stochastic và có thiết kế hệ thống cho pretraining. Một nghiên cứu PEFT sau đó trên frozen OLMo-2 cho thấy giữ residual mixing identity thường có lợi; mHC riêng không nhất quán vượt LoRA.[^6][^7]

Bài học là **ràng buộc phải có đối tượng toán học rõ, và lợi ích pretraining không tự chuyển sang fine-tuning**. V5 chỉ dùng lập luận convex mixture để chặn xác suất; không có Sinkhorn, Birkhoff projection hoặc bảo đảm spectral của mHC. Kết quả PEFT cũng không chứng minh v5 cần đóng băng trunk.

### 2.6. Differential Transformer, 2024/ICLR 2025

Differential attention dùng hiệu hai attention maps để giảm nhiễu; đây là thay đổi attention operator được nghiên cứu trong foundation models.[^8] Nó là hướng cạnh tranh thực sự nếu xác nhận nguồn irrelevant lấn evidence.

Không chọn làm v5 chính: chưa có attention/predictions chứng minh failure này; nó thay cả retrieval lẫn scale của context, khó tách khỏi vấn đề output fusion. Trọng số hiệu có thể âm nên không dùng trực tiếp làm copy probability. Nếu thử sau, chỉ đặt ở semantic read và so với flat attention cùng cost; không cộng vào v5 trước khi kiểm tra mixture.

### 2.7. Attention Residuals và LongT5

Attention Residuals thay cách tổng hợp thông tin theo depth trong backbone; LongT5 dùng local/transient-global encoder attention cho input dài.[^9][^10] Hai nguồn cung cấp hướng giải quyết khác nhau: chọn representation và mở rộng context.

AFMR đã có depth readout ở interface; nó không tương đương AttnRes toàn backbone. V5 không tăng depth taps chỉ vì một model lớn được lợi. LongT5 không chứng minh region-softmax ở output head tốt hơn flat read. Nếu evidence bị truncate, việc sửa output fusion không thể khôi phục phần source chưa encode; mở rộng source sẽ là nhánh kiến trúc riêng, cần budget/visibility control.

## 3. Đối chiếu các technical reports trong dự án

[ARCHITECTURE_REVIEW](../../Technical_Report/ARCHITECTURE_REVIEW.md), [AFMR_NOVELTY_RESEARCH](../../Technical_Report/AFMR_NOVELTY_RESEARCH.md) và [AFMR_FOUNDATION_WORLD_MODEL_NEXT_STEPS](../../Technical_Report/AFMR_FOUNDATION_WORLD_MODEL_NEXT_STEPS.md) là nguồn tổng hợp local. Các mô tả Kimi K3, Qwen3.8-Flash-Next, Instella-MoE và DeepSeek-V4 trong chúng được dùng như lịch sử thiết kế; không coi chúng là thí nghiệm EviSeq hoặc tuyên bố đã tái kiểm chứng toàn bộ PDF.

| Bài học trong report local | Quyết định cho v5 | Phần không chuyển sang |
|---|---|---|
| Giữ đường pretrained và thêm route có giới hạn | Giữ `P_base` trong output; bound mixture weight | Không thay residual topology toàn backbone |
| Phân biệt retrieval descriptors và content values | Giữ `M/H0`, semantic QK riêng và copy riêng | Không tuyên bố tách K/V là novelty |
| Dynamic read có thể hữu ích hơn tăng width đơn thuần | Router theo prefix + semantic context | Không suy ra cần 4 streams/4 heads |
| Sparse selector phải được kiểm chứng | Giữ mọi source token hợp lệ | Không hard top-k, nén source hoặc pruning cache |
| World models/latent prediction có thể tổ chức thông tin | Chưa dùng trong v5 | Không EMA target encoder, InfoNCE, latent rollout hay auxiliary reconstruction |
| Quality và cost cần đo cùng nhau | Thêm profile vocabulary readout và CE memory vào plan | Không gọi module ít tham số là miễn phí |

V-JEPA/LCM/Dreamer trong báo cáo local là hướng tham khảo xa hơn. Việc chuyển latent prediction/planning sang bài toán này sẽ cần mục tiêu học hoặc state assumptions mới; không dùng để tô tên “world model” cho CE-only summarizer.

## 4. So sánh các ứng viên

| Ứng viên | Cơ chế kỳ vọng | Vấn đề chưa giải quyết/rủi ro | Quyết định |
|---|---|---|---|
| V2 với scalar residual gate nhỏ hơn | Giảm mức semantic perturbation | Vẫn một vocabulary distribution; có thể mất gain semantic | Control, không chọn làm đề xuất chính |
| Flat semantic 4×128 | Tăng capacity đọc nguồn | Head redundancy, cost, không giữ base probability | Chưa ưu tiên khi chưa có bằng chứng rank thiếu |
| Null/sentinel semantic attention | Cho phép giảm source-read mass | V2 đã có scalar gate; RMS context có thể xóa mass; source length ảnh hưởng null competition | Không thêm một cơ chế gating thứ ba |
| Value calibrator trước cross-attention | Tăng khả năng alignment | Sửa shared hidden và copy gián tiếp ở mọi layer, chưa có source-conditioning diagnosis | Nhánh dự phòng khi xác nhận alignment yếu |
| Fusion trước final FFN/norm | Cho pretrained nonlinear block xử lý delta | Muốn giữ hidden base cho copy phải fork block hoặc đổi trunk; thêm chi phí và confound | Không chọn ở vòng này |
| Differential semantic read | Khử phần attention chung/irrelevant | Chưa biết noise là bottleneck; có thể bỏ signal hữu ích | Dự phòng có điều kiện |
| Hierarchy/coverage cải tiến | Phân bổ vùng và tránh lặp | Bằng chứng local chưa ủng hộ; usage ≠ fact completion | Bỏ khỏi main candidate |
| Probability mixture + common copy | Cho semantic tăng xác suất mà vẫn giữ đường base hiện hữu | Router có thể yếu/collapse, shared trunk vẫn trôi, thêm vocabulary compute | **Ứng viên chính** |

Không xếp hạng theo độ mới của năm xuất bản. Các nguồn 2025–2026 giúp phản biện chuyển giao, trong khi prior art cũ hơn xác định đúng phép toán đang dùng. Thay một cơ chế đã biết bằng tên mới sẽ không tạo novelty.

## 5. Ranh giới đóng góp học thuật

Không claim mới cho warm-start encoder–decoder, copy mixture, probability mixture, semantic attention độc lập hay sigmoid gate riêng lẻ. Điểm có thể nghiên cứu là **trade-off giữa source adaptation và khả năng giữ vocabulary prediction trong một summarizer ghép checkpoint**, với shared-copy decomposition và bound ở output distribution.

Đóng góp chỉ đứng vững nếu có: v5 hơn new/v2 cùng recipe; mixture hơn hidden-fusion/constant-router controls; nguồn semantic thật sự hữu ích hơn adapter chỉ đọc hidden; chi phí được báo; gain lặp lại qua seeds. Nếu các controls ngang nhau, mô tả kết quả là adaptation thực dụng của prior art, không claim một nguyên lý kiến trúc mới đã được xác lập.

## Nguồn

[^1]: Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, William W. Cohen. *Breaking the Softmax Bottleneck: A High-Rank RNN Language Model*. ICLR 2018, §2.4–2.5, §3.4. [Paper](https://arxiv.org/html/1711.03953v3).
[^2]: Abigail See, Peter J. Liu, Christopher D. Manning. *Get To The Point: Summarization with Pointer-Generator Networks*. ACL 2017, §2.1–2.3. [Paper](https://arxiv.org/html/1704.04368v2).
[^3]: Xiaoyu Shen et al. *Improving Latent Alignment in Text Summarization by Generalizing the Pointer Generator*. EMNLP-IJCNLP 2019. [Publication](https://aclanthology.org/D19-1390/).
[^4]: Zihan Qiu et al. *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*. NeurIPS 2025; mechanism §2, ablations §3–4. [Paper](https://arxiv.org/html/2505.06708v1), [proceedings](https://papers.neurips.cc/paper_files/paper/2025/file/904e89bb4e632e75fb47f093b620b257-Paper-Conference.pdf).
[^5]: Biao Zhang et al. *T5Gemma 2: Seeing, Reading, and Understanding Longer*. Technical report, December 2025, §2–3/Table 1. [Paper](https://arxiv.org/html/2512.14856v1).
[^6]: Zhenda Xie et al. *mHC: Manifold-Constrained Hyper-Connections*. Technical report, December 2025, §3–4. [Paper](https://arxiv.org/html/2512.24880v1).
[^7]: Valentijn Oldenburg et al. *Manifold-Constrained Hyper-Connections for Parameter-Efficient Finetuning*. Preprint, July 2026; abstract and §4.3–4.4. [Paper](https://arxiv.org/html/2607.18130v1). Evidence from frozen OLMo-2 PEFT, not PubMed full fine-tuning.
[^8]: Tianzhu Ye et al. *Differential Transformer*. October 2024, ICLR 2025, §2. [Paper](https://arxiv.org/html/2410.05258v1).
[^9]: Moonshot AI. *Attention Residuals*. 2026. [Official repository and linked report](https://github.com/MoonshotAI/Attention-Residuals). Applied here only as related work for depth aggregation.
[^10]: Mandy Guo et al. *LongT5: Efficient Text-To-Text Transformer for Long Sequences*. Findings of NAACL 2022. [Publication](https://aclanthology.org/2022.findings-naacl.55/).

Các nguồn trên xác nhận cơ chế/tiền lệ tương ứng; công thức v5, quyết định chọn/bỏ và kỳ vọng tác động lên ROUGE là phân tích thiết kế riêng, chưa được các paper này thử nghiệm.

## 6. Deep-research update sau phản biện chéo

Lượt nghiên cứu bổ sung đã đọc các trang gốc của Mixture of Softmaxes, Pointer-Generator, Generalized Pointer Generator, Gated Attention, T5Gemma 2, ReZero, DeepNet, BRIO và phân tích copy/paraphrase. Nó không thay thế audit code hoặc tạo bằng chứng ROUGE mới.

### 6.1. Quyết định routing

Mixture of Softmaxes hỗ trợ việc giữ nhiều vocabulary distributions rồi trộn ở probability space, nhưng paper cũng báo chi phí output tăng và hiệu quả giảm khi thêm quá nhiều mixture do overfit.^11 V5.1 vì vậy chỉ có hai readout dùng chung `W_lm`, không tăng head/rank vô hạn, và bắt buộc có `hidden_interpolation` control.

Hai reviewer cảnh báo nested gate của v5 làm semantic mass thành `(1-g)beta`;
reviewer còn lại cảnh báo simplex tự do có thể làm mất copy mass. Quyết định
dung hòa là copy-mass-preserving capped simplex: giữ `pi_copy=g`, cấp semantic
mass cộng thêm với cap `alpha≤1-g_route-generate_reserve`, trong đó
`g_route=stop_gradient(g)`, và giữ `P0` làm phần còn lại. Cấu trúc này có thể
tái tạo `new` khi `delta=0`, đồng thời không khóa semantic branch bởi một phép
nhân thứ hai. `generate_reserve` là reserve chứ không phải floor tuyệt đối.

### 6.2. Quyết định K/V và residual

Generalized Pointer Generator cho phép chỉnh token được align thay vì chỉ exact-copy.^12 V5.1 chỉ mượn nguyên tắc tách alignment khỏi abstraction; không đưa relation embedding hay target aligner vào main.

Gated Attention cho thấy tác dụng của gate phụ thuộc vị trí: gating ở output SDPA tạo nonlinearity/sparsity, còn một số vị trí sau projection không đem lại tác dụng tương tự.^13 Vì v4 đã có head gate/post-norm, v5.1 không lặp lại gating trong backbone. Main giữ `K=H0,V=H0` để bảo toàn tín hiệu v2; `K=M,V=H0` là ablation trực tiếp cho giả thuyết AFMR retrieval key tốt hơn.

ReZero trực tiếp nghiên cứu residual scalar khởi tạo bằng zero; DeepNet nghiên
cứu scaling/initialization để ổn định Transformer rất sâu.^14,^15 Đây chỉ là
động lực khái niệm cho endpoint `Wo=0`, không phải bằng chứng trực tiếp cho
`Wo=tiny` trong EviSeq. Hiện tượng Q/K/V học chậm khi `Wo=0` vẫn là giả thuyết
phải đo; plan có thêm `Wo=tiny` control. V2 cũng có
`W_out=0` và inner gate khoảng `0.05`, nên không được tuyên bố v5 có gradient
semantic yếu hơn v2 theo tỷ lệ `(1-g)/alpha` nếu chưa đo theo optimizer step.

### 6.3. Đánh giá và phạm vi novelty

T5Gemma 2 báo tied embeddings, merged attention và cross-attention ở mọi layer; ablation chỉ đặt cross-attention ở global layers giảm khoảng 1.3 điểm trong bảng của họ.^16 Đây là cơ sở để giữ source access ở mọi layer, nhưng không phải bằng chứng v5.1 sẽ thắng PubMed.

Phân tích “To Point or Not to Point” cho thấy hệ thống abstractive có trade-off giữa pointing và paraphrase.^17 Do đó plan ghi copy precision, non-copyable CE, source-visible bigrams và output length thay vì suy diễn từ ROUGE. BRIO dùng candidate-quality ranking; SimCLS huấn luyện scorer/reranker reference-free. Cả hai đều nằm ngoài vòng CE-only để không trộn objective.^18,^19

SDACL (Huang et al., Neurocomputing 2024) được đọc từ [bản PDF đã cung cấp](/Users/kieugiangbien/Downloads/Paper/VAI/Summarization/SDACL.pdf) và đối chiếu DOI.^20 Phương pháp dùng semantic similarity giữa summary và từng câu nguồn, chọn positive/negative theo salience và gán soft distance-aware weights. Nó phù hợp làm hướng contrastive sau này, nhưng không được đưa vào v5.1; main phải chứng minh architecture trước.

### 6.4. Ranh giới claim

Không gọi capped simplex là phát minh mixture, không gọi copy head là
pointer-generator mới, và không gọi output router là Gated Attention của Qiu
et al. Điểm có thể viết thành đóng góp là một decomposition cụ thể cho
summarizer ghép checkpoint: **bounded source-conditioned semantic correction,
copy responsibility giữ riêng, probability-space anchor và kiểm chứng
trade-off bằng controls**. Claim này chỉ hợp lệ nếu E/F hơn A/B/H/I với cùng
protocol và lặp lại qua seed; D chỉ là parity endpoint, không phải candidate
chất lượng.

## Ghi chú khóa plan sau audit

Các điểm công thức và protocol đã được chốt lại trong
[ARCHITECTURE_TARGET_AUDIT.md](ARCHITECTURE_TARGET_AUDIT.md): main là
copy-mass-preserving capped simplex (không phải independent simplex),
`Wo=tiny_rms_1e-3` là candidate còn `Wo=0` chỉ parity, independent-control dùng
`log(g)` thay vì `logit(g)`, và C17 kiểm tra `dP/dg` thay vì đòi `dL/dg` trùng
copy-only. Những sửa này là điều kiện triển khai rút ra từ code review, không
phải kết quả thực nghiệm mới.

[^11]: Z. Yang et al., “Breaking the Softmax Bottleneck: A High-Rank RNN Language Model,” ICLR 2018, [arXiv](https://arxiv.org/html/1711.03953v3).
[^12]: X. Shen et al., “Improving Latent Alignment in Text Summarization by Generalizing the Pointer Generator,” EMNLP-IJCNLP 2019, [ACL Anthology](https://aclanthology.org/D19-1390/).
[^13]: Z. Qiu et al., “Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free,” 2025, [arXiv](https://arxiv.org/html/2505.06708v1).
[^14]: T. Bachlechner et al., “ReZero is All You Need: Fast Convergence at Large Depth,” 2020, [arXiv](https://arxiv.org/html/2003.04887).
[^15]: H. Wang et al., “DeepNet: Scaling Transformers to 1,000 Layers,” 2022, [arXiv](https://arxiv.org/html/2203.00555).
[^16]: B. Zhang et al., “T5Gemma 2: Seeing, Reading, and Understanding Longer,” 2025, [arXiv](https://arxiv.org/html/2512.14856v1).
[^17]: M. Wilber, W. Timkey, M. van Schijndel, “To Point or Not to Point: Understanding How Abstractive Summarizers Paraphrase Text,” Findings ACL 2021, [ACL Anthology](https://aclanthology.org/2021.findings-acl.298/).
[^18]: Y. Liu et al., “BRIO: Bringing Order to Abstractive Summarization,” ACL 2022, [ACL Anthology](https://aclanthology.org/2022.acl-long.207/).
[^19]: Y. Liu and P. Liu, “SimCLS: A Simple Framework for Contrastive Learning of Abstractive Summarization,” ACL-IJCNLP 2021, [ACL Anthology](https://aclanthology.org/2021.acl-short.135/).
[^20]: Y. Huang et al., “Sentence salience contrastive learning for abstractive text summarization,” Neurocomputing 593 (2024) 127808, [DOI](https://doi.org/10.1016/j.neucom.2024.127808).
