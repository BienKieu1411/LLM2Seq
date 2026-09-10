# Quyết định kiến trúc AFMR

## Phạm vi và quyết định

Audit hợp nhất, gồm bằng chứng, invariant tensor/xác suất/gradient/DDP và
điều kiện bác bỏ, nằm ở [ARCHITECTURE_TARGET_AUDIT.md](ARCHITECTURE_TARGET_AUDIT.md).

Tài liệu này chốt hướng triển khai sau hai vòng phản biện chéo của ba `luna_worker` và một lượt nghiên cứu lại các công trình gốc về mixture-of-softmax, pointer-generator, gating, ổn định residual và summarization. Architecture gate là R1/RL cao hơn `eviseq_new` và R2 cao hơn `update_v2`; system stretch là thêm R2 cao hơn T5Gemma sau khi tái lập protocol. Chưa có kết quả AFMR để bảo đảm các gate này.

Quyết định chính là dùng **copy-mass-preserving capped simplex** ở output
vocabulary. Semantic branch được cấp một trọng số riêng, còn giá trị forward của
copy gate `g` của `new` được giữ nguyên. Đây là phương án trung gian có kiểm
soát giữa hai đề xuất đối nghịch: capped three-way simplex hoàn toàn tự do và
semantic-outside-copy. Nó cho phép semantic route có mass cộng thêm thay vì bị
nhân với `(1-g)`, nhưng không để semantic router lấy mass trực tiếp từ copy
route. Tên này cố ý phân biệt với independent simplex: `pi_copy=g`, base
floor dương và semantic mass dương không thể cùng bảo đảm khi `g` gần một.
Main ưu tiên giữ copy mass; control F dùng trade-off khác để đo floor thật.

Thiết kế main AFMR:

    P0 = softmax(z0),       z0 = W_lm h
    Ps = softmax(zs),       zs = W_lm (h + delta)
    Pcopy = marginalized copy distribution

    alpha_raw = alpha_max * sigmoid(r_sem) * semantic_evidence
    g_route = stop_gradient(g)
    alpha = min(alpha_raw, max(0, 1 - g_route - generate_reserve))

    pi_copy = g
    pi_sem  = alpha
    pi_base = 1 - g - alpha

    P = pi_base P0 + pi_sem Ps + pi_copy Pcopy

`alpha_max=0.20`, `generate_reserve=0.05` là giá trị pilot; router bias được
đặt để `alpha` ban đầu khoảng 0.05. `generate_reserve` không phải floor tuyệt
đối của `pi_base` khi `g` lớn. Nếu `Wo=0`, `delta=0`, do đó `Ps=P0` và `P`
khôi phục phân phối copy của `new` ở cùng checkpoint, prefix và source.

Main semantic reader giữ topology đã được kiểm chứng ở v2: `H0n=RMS(H0)`,
`K=RMS(Wk(H0n))`, `V=Wv(H0n)`, flat rank 128, một head, không
hierarchy/planner/coverage. Variant `K=M,V=H0` được chạy riêng để kiểm tra
giả thuyết AFMR retrieval memory giúp định địa chỉ; không đưa vào main trước
khi có validation evidence.

## Bằng chứng và phản biện chéo

Ba agent đã đọc cùng một đặc tả v5 và phản biện lẫn nhau.

- `critic_gradient` chỉ ra semantic mass cũ là `(1-g) × beta × a`. Với
  `Wo=0`, `Ps=P0`; router và Q/K/V không có tín hiệu phân biệt ở backward đầu
  qua expert residual, còn `Wo` vẫn có gradient qua vocabulary logits. Agent này
  chọn capped three-way simplex và đề xuất `K=M,V=H0`.
- `critic_architecture` phản biện rằng simplex tự do có thể lấy mất copy mass, làm mất R2. Agent này chọn semantic-outside-copy giữ `g`, `K=H0,V=H0`, và đề xuất null slot chỉ sau khi có chẩn đoán.
- `critic_eval_novelty` đồng ý nested gate là rủi ro, nhưng yêu cầu floor cho base/copy, hard source-empty fallback, gauge `C=Z0`, cùng protocol đánh giá công bằng. Agent này chọn capped simplex nhưng giữ `K=H0` trong main.

Điểm đồng thuận là bỏ nested gate và semantic gate kép, không quay lại complexity của v4, khóa source-empty behavior, và tách ảnh hưởng probability fusion khỏi decoding gauge. Bất đồng về K/V được giải quyết bằng `K=H0,V=H0` cho main và `K=M,V=H0` làm ablation. Bất đồng về `Wo` được giải quyết bằng hai cấu hình được định trước: `zero` để kiểm tra parity và `tiny` để kiểm tra tốc độ đánh thức Q/K/V; không chọn dựa trên test.

## Cơ sở từ nghiên cứu gốc

Mixture of Softmaxes chứng minh việc trộn các phân phối sau softmax có thể mở rộng họ phân phối so với một projection-softmax duy nhất, nhưng chi phí tính toán tăng và quá nhiều mixture có thể overfit.^1 Điều này ủng hộ việc giữ hai readout chung vocabulary head, đồng thời giải thích vì sao AFMR phải có control hidden interpolation và profile chi phí.

Pointer-generator cho thấy copy distribution và generation distribution có thể cùng tồn tại để tăng khả năng tái tạo token nguồn mà vẫn sinh từ mới.^2 Generalized pointer còn cho phép chỉnh token được align, cho thấy copy và semantic abstraction không nên bị buộc vào cùng một attention map.^3 V5.1 giữ copy head của `new` độc lập và chỉ cho semantic branch sửa vocabulary distribution.

Nghiên cứu gated attention gần đây cho thấy vị trí và granularity của gate quyết định tác dụng; sigmoid gate sau SDPA có thể thêm nonlinearity và sparsity, trong khi gate đặt sai vị trí không đem lại lợi ích tương tự.^4 Vì vậy AFMR không gọi router output là “gated attention” và không lặp lại head gate/hierarchy của v4.

T5Gemma 2 dùng adaptation từ decoder-only sang encoder-decoder, tied embeddings và merged attention; ablation của họ báo giảm khoảng 1.3 điểm trung bình khi chỉ đặt cross-attention ở global layers.^5 Điều này củng cố quyết định giữ cross-attention ở mọi layer trong trunk EviSeq, nhưng không cho phép suy ra AFMR sẽ vượt T5Gemma từ kiến trúc output riêng.

ReZero trực tiếp nghiên cứu residual scalar khởi tạo bằng zero; DeepNet nghiên cứu scaling/initialization cho Transformer rất sâu.^6,^7 Đây là động lực khái niệm cho endpoint `Wo=0`, không phải bằng chứng trực tiếp cho `Wo=tiny` trong EviSeq. Zero-init vẫn có thể làm semantic Q/K/V học chậm ở bước đầu; vì vậy plan yêu cầu đo gradient và có variant `Wo` rất nhỏ.

Phân tích pointer-generator cho thấy độ trừu tượng và độ trung thành là một trade-off có thể đo được, không thể suy ra từ ROUGE riêng lẻ.^8 BRIO dùng candidate-quality ranking, còn SimCLS huấn luyện scorer/reranker reference-free.^9,^10 V5.1 không đưa candidate generation, contrastive loss hay reranking vào training; chúng chỉ là hướng nghiên cứu sau khi architecture CE-only đã được chứng minh.

Tài liệu SDACL được cung cấp trong dự án dùng sentence-level semantic similarity và salience distance-aware contrastive loss để phân biệt câu nguồn quan trọng và nhiễu.^11 Phương pháp này không được đưa thẳng vào AFMR vì sẽ trộn thay đổi architecture với training objective. Nó chỉ được giữ làm hướng vòng sau, sau khi main architecture vượt các baseline trong cùng CE recipe.

## Đặc tả module

Trunk giữ native encoder, AFMR, decoder cross-attention và grounded copy của `new/v2`. `H0=bridge.value_memory`, `M=bridge.memory`; AFMR tiếp tục tạo:

    H0: value anchor
    M : retrieval key memory
    b : source prior

Semantic reader main:

    q = Wq RMS(h)
    H0n = RMS(H0)
    Ks = RMS(Wk(H0n))
    Vs = Wv(H0n)
    A = masked_softmax(q Ks^T / sqrt(r) + semantic_prior_scale * b)
    c = A Vs
    u = RMS(c)
    delta_raw = Wo u
    delta = smooth_relative_RMS_cap(delta_raw, h, rho=0.10)

Không dùng semantic gate `.05` thứ hai. RMS cap là giới hạn duy nhất của residual
trong main. `Wo=tiny_rms_1e-3` là candidate chính; `Wo=0` chỉ là endpoint
control/parity. Hàm cap không phải placeholder: dùng `RMS(x)=sqrt(mean_hidden(x²))`
ở FP32, `t=max(.10*RMS(h),1e-6)`, `r=RMS(delta_raw)`,
`delta=tanh(r/t)/max(r/t,1e-6)*delta_raw`, với `delta=0` khi `r=0`; C5/C8
kiểm tra bound và gradient hữu hạn.

Control `legacy_semantic` phải bật lại `semantic_gate_init=.05` đúng graph v2;
`inner_gate=false` chỉ là quyết định của main AFMR.

Router chỉ nhận prefix-conditioned features: `[q, u, stop_gradient(g_raw_logit),
stop_gradient(copy_entropy_norm)]`. `g_raw_logit` là output trước sigmoid của
legacy gate; entropy được chuẩn hóa theo `log(max(2,n_valid))`. Không nhận
reference, gold token, ROUGE, future target hoặc generated candidates. Nếu
semantic source mask rỗng thì `semantic_evidence=0` với shape `[B,T,1]`,
tạo từ `content_mask.any(dim=-1)[:,None,None].float()` rồi broadcast theo `T`;
không dùng copy mask `[B,W]`.
Nếu copy mask rỗng thì `g=0`; nếu cả hai rỗng thì `P=P0`. `g_route` chỉ detached
ở feature/cap; shared trunk vẫn có thể làm thay đổi copy sau optimizer updates.

Để tách decoding gauge, đặt:

    Z0 = logsumexp(z0)
    output_logits = log(P) + Z0

`C=(1-beta)Z0+beta Zs` của v5 cũ chỉ còn là diagnostic. Probe `repetition_penalty=1.0` và setting hiện hành phải chạy song song.

## Thứ tự implementation

1. Cố định common protocol: commit, tokenizer, source policy, split manifest, effective batch, optimizer steps, dtype và evaluator.
2. Dựng `base_only` tương đương `new` và `legacy_semantic` tương đương `update_v2` trong package riêng.
3. Tách semantic reader rank128 và kiểm chứng chính xác thứ tự
   `H0n=RMS(H0)`, `K=RMS(Wk(H0n))`, `V=Wv(H0n)`, cùng mask, cap, dtype và cache.
4. Implement mixture NLL ở log-domain; tuyệt đối không cộng ba CE riêng.
5. Implement `copy_mass_preserving_capped_simplex` với hard masks,
   `generate_reserve`, detached `g_route` và diagnostics.
6. Thêm variant `independent_capped_simplex` để phản biện trực tiếp; dùng
   `l_base=log(1-g_route)+r_base`, `l_copy=log(g_route)+r_copy`, `l_sem=-20+r_sem`
   (không dùng `logit(g)`), và variant này không phải main.
7. Thêm checkpoint schema, DDP token normalization, accumulation, cache lifecycle và generation gauge.
8. Chạy smoke/acceptance tests trước khi train; sau đó pilot một seed trên validation.
9. Chỉ full-train các variant qua pilot; checkpoint chọn bằng validation, không chọn test.

## Ma trận ablation bắt buộc

- `new/common-control`: copy-only.
- `update_v2/common-control`: flat semantic residual, một vocabulary distribution.
- `AFMR-zero-parity`: copy-mass-preserving capped simplex, `K=H0,V=H0`,
  `Wo=0`, chỉ dùng cho endpoint/smoke.
- `AFMR-tiny`: như trên nhưng `Wo` calibration deterministic để correction
  `1e-3 RMS(h)`, candidate training chính được đăng ký trước.
- `AFMR-independent`: capped three-way simplex có base floor, copy gate feature và semantic mass độc lập.
- `AFMR-KM`: candidate E nhưng `K=M,V=H0`, source prior giảm về 0 hoặc hệ số nhỏ để kiểm tra double-focus.
- `hidden-interpolation`: cùng semantic reader nhưng `softmax(W_lm(h+lambda_hidden delta))`, với `lambda_hidden` khóa trước, để kiểm tra probability fusion có giá trị riêng không.
- `constant-alpha`: alpha cố định `0.05` (và nếu pilot đăng ký trước thì `0.10`), cùng evidence/cap với main.

Không thay nhiều biến trong cùng một run. Null slot, entropy/confidence gate và copy-protected residual chỉ được thêm sau khi main ablation đã cô lập được nguyên nhân.

## Acceptance và tiêu chí dừng

Trước training phải đạt:

- Forward endpoint `Wo=0` trùng `new` trong tolerance; v2 endpoint so probability/loss/gradient, còn raw logits chỉ so khi control dùng gauge `Zs`.
- Mixture NLL đúng với oracle FP32/FP64 trên token copyable, non-copyable, duplicate IDs và mask rỗng.
- Backward endpoint đầu chỉ yêu cầu `Wo` hữu hạn (legacy base/copy/trunk vẫn
  nhận gradient); sau optimizer update, candidate tiny phải có Q/K/V/router
  gradient khác zero trên batch có signal.
- 1 GPU và 2 GPU cho cùng canonical global batch, token denominator và scheduler step cho update gần nhau.
- Dense/chunked, teacher-forced/incremental cache và compaction cho kết quả tương đương.
- BF16/FP32 không có NaN; các reduction/gate/logsumexp thực hiện ở FP32.
- Gauge `log(P)+Z0` giữ endpoint sau repetition penalty; temperature/top-p chỉ áp lên final mixture khi sampled generation.
- Checkpoint/resume và config fingerprint không chấp nhận rank, gauge hoặc mode mismatch.
- Với `g_route` detached, kiểm tra Jacobian `dP/dg=Pcopy-P0` và alpha không tạo
  đường gradient trực tiếp vào `g`; log `dL/dg` thực tế riêng, không yêu cầu nó
  khớp copy-only; gradient shared trunk được báo riêng.
- Fixed-small-set 16–32 examples phải giảm CE rõ ràng trước khi chạy PubMed;
  canonical manifest replay phải cho 1/2 GPU cùng update IDs, denominator và
  scheduler step.

Thắng kiến trúc được xét bằng E/F so với A/B cùng recipe; thắng hệ thống so
với mốc lịch sử và T5Gemma được báo riêng. Chỉ gọi là thắng hệ thống nếu vượt
mốc `new` ở R1 và ROUGE-L, vượt mốc `update_v2` và T5Gemma ở R2, cùng một
checkpoint validation và chênh lệch lặp lại trên tối thiểu ba seed. Nếu
independent simplex tốt hơn nhưng làm R2 giảm, giữ E. Nếu
`hidden-interpolation` ngang E, không claim probability fusion là nguyên nhân.

## Rủi ro và hướng sau

Rủi ro lớn nhất vẫn là shared trunk được cập nhật làm yếu đường copy dù output giữ `g`; cần log copy responsibility, copy precision và CE theo token copyable. `K=M` có thể double-focus với source prior; chỉ promotion khi R1/RL tăng mà R2 không giảm. Null slot có thể làm mất recall nếu evidence phân tán; chỉ thêm với mass `m=1-A_null` và value zero.

Nếu AFMR vượt ROUGE nhưng hallucination chưa giảm, không gán factuality từ kiến trúc mixture. Khi đó mới mở vòng riêng dùng SDACL/entailment/attribution, giữ nguyên checkpoint và protocol architecture để tách đóng góp.

## Nguồn chính

[^1]: Z. Yang et al., “Breaking the Softmax Bottleneck: A High-Rank RNN Language Model,” ICLR 2018, [arXiv](https://arxiv.org/html/1711.03953v3).
[^2]: A. See, P. Liu, C. Manning, “Get To The Point: Summarization with Pointer-Generator Networks,” ACL 2017, [ACL Anthology](https://aclanthology.org/P17-1099/).
[^3]: X. Shen et al., “Improving Latent Alignment in Text Summarization by Generalizing the Pointer Generator,” EMNLP-IJCNLP 2019, [ACL Anthology](https://aclanthology.org/D19-1390/).
[^4]: Z. Qiu et al., “Gated Attention for Large Language Models,” 2025, [arXiv](https://arxiv.org/html/2505.06708v1).
[^5]: B. Zhang et al., “T5Gemma 2: Seeing, Reading, and Understanding Longer,” 2025, [arXiv](https://arxiv.org/html/2512.14856v1).
[^6]: T. Bachlechner et al., “ReZero is All You Need,” 2020, [arXiv](https://arxiv.org/html/2003.04887).
[^7]: H. Wang et al., “DeepNet: Scaling Transformers to 1,000 Layers,” 2022, [arXiv](https://arxiv.org/html/2203.00555).
[^8]: M. Wilber, W. Timkey, M. van Schijndel, “To Point or Not to Point,” Findings ACL 2021, [ACL Anthology](https://aclanthology.org/2021.findings-acl.298/).
[^9]: Y. Liu et al., “BRIO: Bringing Order to Abstractive Summarization,” ACL 2022, [ACL Anthology](https://aclanthology.org/2022.acl-long.207/).
[^10]: Y. Liu and P. Liu, “SimCLS: A Simple Framework for Contrastive Learning of Abstractive Summarization,” ACL-IJCNLP 2021, [ACL Anthology](https://aclanthology.org/2021.acl-short.135/).
[^11]: Y. Huang et al., “Sentence salience contrastive learning for abstractive text summarization,” Neurocomputing 593 (2024) 127808, [DOI](https://doi.org/10.1016/j.neucom.2024.127808); local copy: [SDACL.pdf](/Users/kieugiangbien/Downloads/Paper/VAI/Summarization/SDACL.pdf).
