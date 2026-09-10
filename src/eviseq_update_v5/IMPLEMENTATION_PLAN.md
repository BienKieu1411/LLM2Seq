# Kế hoạch tích hợp EviSeq v5.1

## 1. Quyết định đã chốt

Đây là bản kế hoạch implementation sau deep research và hai vòng phản biện chéo của `critic_gradient`, `critic_architecture` và `critic_eval_novelty`. Package v5.1 đã được dựng trong folder này; checkpoint PubMed và kết quả ROUGE vẫn chưa có.

Mục tiêu ưu tiên là đạt đồng thời:

- R1 và ROUGE-L cao hơn mốc tốt nhất của `eviseq_new`: `49.626` và `45.895`.
- R2 cao hơn `update_v2`: `21.953`, đồng thời vượt mốc T5Gemma đang báo `21.990` nếu protocol được tái lập.

Main v5.1 dùng **copy-mass-preserving capped simplex**. Semantic branch có
mass riêng; giá trị forward của copy gate `g` của `new` được giữ nguyên để
không đánh mất tín hiệu R2. Đây là bảo toàn mass ở output, không phải đóng băng
tham số copy hay shared trunk.

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

Giá trị pilot: `alpha_max=0.20`, `generate_reserve=0.05`, router bias để
`alpha` ban đầu khoảng `0.05`. `generate_reserve` không phải floor tuyệt đối
cho `pi_base` khi `g` lớn. Nếu `Wo=0`, `delta=0`, `Ps=P0`; vì vậy output
khôi phục `new` ở cùng checkpoint, source và prefix. Đây là parity guarantee ở
forward, không phải guarantee rằng sau fine-tuning shared trunk vẫn giữ nguyên.

Main semantic reader giữ flat independent read của v2, rank `128`, một head.
Phải chép đúng graph reader: `H0n=RMS(H0)`, `K=RMS(Wk(H0n))` và
`V=Wv(H0n)`. Variant `K=M,V=H0` được chạy riêng, không âm thầm đưa vào main.

Thiết kế này thay nested gate cũ. Nó không dùng semantic coefficient `(1-g)beta`; semantic chỉ bị cap khi copy gate đã chiếm gần hết simplex. Không thêm semantic gate `.05` thứ hai; RMS cap `.10` là giới hạn residual duy nhất.

## 2. Phạm vi giữ và loại bỏ

Giữ nguyên từ `new`/`update_v2`:

- native encoder, AFMR, depth/feature memory và source prior;
- decoder cross-attention ở mọi layer với `K` từ `M`, `V` từ `H0`;
- character-overlap copy alignment, token-ID duplicate marginalization và copy mask;
- flat semantic source read rank128 của v2;
- một lượt backbone và teacher forcing CE.

Loại khỏi main:

- hierarchy/region softmax, planner usage, coverage/continuity;
- partition head, tangent projection và norm restore của v4;
- 4×128 semantic heads trước khi chứng minh rank128 thiếu capacity;
- contrastive learning, R-Drop, NEFTune, distillation, RL, candidate mining, reranking và self-improve;
- temperature/top-p trong benchmark greedy. Hai tham số chỉ có trong API sampled generation/candidate về sau.

## 3. Ranh giới module và file dự kiến

Các module bên dưới nằm trong `src/eviseq_update_v5`; acceptance evidence được ghi ở `IMPLEMENTATION_EVIDENCE.md`.

| File | Trách nhiệm | Điều kiện bắt buộc |
|---|---|---|
| `modeling/encoder.py`, `afmr.py`, `controller.py` | Native encoder/AFMR giữ graph của common control | Hash/state-output đối chiếu `new` và v2 |
| `modeling/grounded_copy.py` | Copy keys, alignment, masks, `g`, `Pcopy` | Semantic không được làm đầu vào copy |
| `modeling/semantic_read.py` | Flat rank128 reader, `K/V`, cap và evidence mask | Main `K=H0,V=H0`; `K=M` chỉ variant |
| `modeling/dual_readout.py` | `P0`, `Ps`, capped simplex, mixture NLL, gauge | Không weighted-sum CE riêng |
| `modeling/decoder.py`, `model.py`, `outputs.py` | Một backbone pass, nối source state và output | Không planner/coverage history |
| `training/engine.py`, `optimizer.py` | CE-only, DDP, accumulation, token normalization | Không gọi generation/miner |
| `training/checkpoint.py` | Strict architecture/config schema và resume | Từ chối mode/rank/gauge mismatch |
| `evaluation/generate.py` | Greedy và sampled API, cache lifecycle | Processor áp lên final mixture |
| `configs/afmr_v51.yaml` | Main và named controls | Không kế thừa silent defaults của v4 |
| `scripts/run_pubmed_pair.sh` | 1/2 GPU, resolved config, dry-run | In global batch/steps/dtype/seed |
| `tests/…` | Acceptance ở mục 7 | CPU test không thay CUDA/NCCL test |

Optimizer group phải nhận diện `grounded_copy`, `semantic_read` và `dual_readout` theo tên module rõ ràng. Không dựa vào `requires_grad=True` để suy ra parameter group.

## 4. API và config contract

Semantic API:

    semantic_read(h, semantic_state) -> hs, q, u, evidence, diagnostics

Readout API phải nhận một state duy nhất đã tính các branch cần thiết:

    ReadState = {
        h, hs, z0, zs, copy_log_attention, raw_copy_gate, g,
        semantic_evidence, alpha, pi_base, pi_sem, pi_copy,
        logP, gauge
    }
    readout(ReadInputs(h=h, hs=hs, z0=z0, zs=zs,
                       semantic_evidence=semantic_evidence,
                       copy_state=copy_state)) -> ReadState

`semantic_evidence` main có shape `[B,T,1]`, tạo bằng
`content_mask.any(dim=-1)[:,None,None].float()` rồi broadcast theo `T`, còn
`copy_state.mask` có shape `[B,W]`. Không được
dùng lại copy mask cho semantic. Một hàm readout duy nhất tạo `g`, `alpha`,
mixture và diagnostics để không gọi copy attention hai lần hoặc tạo nhánh
gradient khác nhau.

Dense logits và chunked-vocabulary loss phải dùng cùng route state và cùng
mixture-NLL kernel. Ở `return_logits=False`, branch vocabulary logits là lazy
và chỉ target logits/normalizers được stream để tránh materialize `[B,T,V]`;
đây là cùng `ReadState` contract về routes, mask và gauge chứ không phải một
công thức loss thứ hai. C18 phải so dense CE với chunked CE và nhóm gradient
trên cùng hidden/state ở FP32 trong tolerance đã đăng ký.

`semantic_state` chứa `H0=bridge.value_memory`, `M=bridge.memory`, native mask,
source prior và `semantic_prior_scale`. `bridge.value_memory` là bắt buộc:
`encode_source` phải raise lỗi khi nó là `None`, không fallback sang
`bridge.memory`. Hai đường copy và
semantic được phép chia sẻ encoder/AFMR gradient khi training, nhưng copy head
chỉ đọc original `h` và copy state. Router nhận `g_route=stop_gradient(g)` và
các copy diagnostics detached; `g` có gradient trực tiếp từ mixture, còn
`dL/dg` phải log riêng vì semantic branch thay đổi phân phối cuối.

Config main:

| Trường | Giá trị main | Control |
|---|---:|---|
| `semantic_read.rank` | `128` | `256` chỉ sau pilot |
| `semantic_read.num_heads` | `1` | Không dùng `4×128` ở vòng đầu |
| `semantic_read.key_source` | `H0` | `M` ở `v5.1-KM` |
| `semantic_read.value_source` | `H0` | Không dùng `M` làm value |
| `semantic_read.semantic_prior_scale` | `1.0` | `0.0` cho `v5.1-KM` mặc định |
| `semantic_read.inner_gate` | `false` | `true` + `gate_init=.05` chỉ cho `semantic_only_v2` |
| `semantic_read.max_relative_rms` | `0.10` | Giữ cố định trong ablation routing |
| `semantic_read.output_init` | `tiny_rms_1e-3` | `zero` chỉ parity/smoke |
| `readout.mode` | `copy_mass_preserving_capped_simplex` | `independent_capped_simplex`, `hidden_interpolation` |
| `readout.alpha_max` | `0.20` | `0.10`, `0.50` validation-only |
| `readout.generate_reserve` | `0.05` | `0.10`, `0.20`, `0.50` validation-only |
| `readout.base_floor` | — | `0.05` chỉ cho independent simplex |
| `readout.copy_gate` | `legacy_g` | `g=0` nếu copy mask rỗng |
| `readout.detach_copy_route_features` | `true` | `false` chỉ ở `router_sees_copy` |
| `readout.copy_entropy_feature` | `normalized_attention_entropy` | Không có `copy_support_mass` |
| `readout.hard_source_fallback` | `true` | Không được tắt main |
| `readout.logit_offset` | `z0_log_partition` | Interpolated offset chỉ diagnostic |
| `readout.hidden_lambda` | — | `0.05/0.10/0.20` chỉ cho `hidden-interpolation`; không dùng `alpha` mixture |
| `readout.null_slot` | `false` | Bật sau diagnostic evidence |
| `readout.ce_chunk_size` | đo bằng pilot | Dense FP32 oracle bắt buộc |

`z0_log_partition` có nghĩa:

    Z0 = logsumexp(z0)
    output_logits = log(P) + Z0

Đây là scalar offset theo timestep, không phải cộng vector `z0`. Nó giữ gauge legacy của `new` khi `P=P0` trước repetition penalty.

## 5. Thứ tự implementation

1. **Đóng băng protocol.** Ghi commit, tokenizer/alignment version, source length policy, split manifest, preprocessing, canonical global-batch manifest, optimizer updates, scheduler, clip, dtype và Perl ROUGE command. Tách dirty files khỏi artifact của run. verify: resolved config, manifest hash và command được lưu trước pilot.
2. **Dựng legacy endpoints.** `base_only` phải tương đương `new`; `semantic_only_v2` phải tương đương v2 flat. Dùng cùng pretrained base state theo từng seed.
   Khởi tạo module mới trong `torch.random.fork_rng` hoặc cơ chế tương đương để
   không làm lệch RNG của copy/AFMR; C1 phải so cả state và output. verify: state/output parity và optimizer parameter coverage trên cùng seed.
3. **Tách semantic reader.** Giữ chính xác thứ tự RMS, mask, cap và source prior
   của v2 (`K=RMS(Wk(RMS(H0)))`, `V=Wv(RMS(H0))`); kiểm tra cache native `K/V`.
   Main không có hierarchy hay planner. verify: shapes, mask, cap và cache K/V trên fixture không nhãn.
4. **Dựng dense probability oracle.** Tính `P0`, `Ps`, `Pcopy`, capped simplex và NLL ở FP32; test trước khi chunking. verify: tổng mass bằng một, duplicate IDs và target không copy được có oracle đúng.
5. **Thêm routing main.** Implement `copy_mass_preserving_capped_simplex`
   với `g_route=stop_gradient(g)` cho feature/cap, hard mask semantic/copy
   source-empty; log `alpha`, `g`, `pi_base`, `pi_sem`, `pi_copy` theo token và
   theo sample. verify: empty-source fallback, simplex mass và Jacobian `dP/dg` qua test oracle.
6. **Thêm controls.** Implement `independent_capped_simplex`, `constant-alpha`, `hidden-interpolation`, `K=M/V=H0` và `Wo=zero` parity với tên mode riêng. Main candidate luôn là `Wo=tiny` đã calibration deterministic. verify: mỗi control có config fingerprint riêng và không đổi main route.
7. **Nối engine.** Kiểm tra token-weighted loss, accumulation, DDP
   synchronization, optimizer groups, checkpoint và resume. Prediction resume
   phải kiểm tra checkpoint/config/split/decoder fingerprint trước khi trả metric. verify: resume test từ checkpoint và từ chối đổi protocol/world size.
8. **Nối cache/generation.** Source state chuẩn bị một lần; prefix-dependent `h/q/u/alpha` tính lại. Temperature/top-p chỉ áp lên `output_logits` cuối cùng trong sampled API. verify: incremental/reorder/source reset parity và seed sampling tái lập.
9. **Profile.** Đo dense/chunked vocabulary memory, throughput, peak VRAM, BF16 rounding và 1/2 GPU equivalence. verify: lưu metrics JSONL với dtype, peak memory, manifest và scheduler step.
10. **Smoke rồi pilot.** Chỉ train sau khi acceptance gates đạt; pilot dùng validation, không dùng test để chọn mode/epoch. verify: tiny CE giảm, test chỉ chạy sau khi checkpoint được khóa.

Checkpoint selection bắt buộc dùng `selection_metric=validation_ce` và
`save_best=true`, với cùng validation manifest/tie-break cho A/B/E/F. Evaluator
phải kiểm tra checkpoint/config/split/decoder fingerprint trước khi dùng
prediction cache; file cũ chỉ khớp ID/reference prefix vẫn bị từ chối.

Common recipe phải khóa AdamW, nhóm no-decay và một global cosine schedule trên
toàn bộ optimizer updates; không reset scheduler ở stage transition. Nếu engine
cũ còn stage-local scheduler, sửa trước khi rerun baseline.

## 6. Đường gradient và initialization

Main candidate dùng `Wo=tiny_rms_1e-3`; `Wo=0` chỉ để bảo toàn forward endpoint.
Ở backward đầu tiên của endpoint `Ps=P0`, nên
`∂L/∂alpha=0`, router không nhận gradient từ expert difference và Q/K/V chưa
nhận gradient qua `u` vì `∂delta/∂u=Wo=0`; `Wo` vẫn phải nhận gradient hữu
hạn trên batch nondegenerate qua `zs=W_lm(h+Wo u)`. Sau optimizer update mới
kiểm tra Q/K/V và router có gradient khác zero. V2 cũng có `W_out=0` cùng inner
gate khoảng `0.05`, nên không dùng tỷ lệ lý thuyết để kết luận v5 yếu hơn; đo
gradient theo step.

Variant tiny dùng `Wo_raw` seed cố định trên calibration batch không dùng label,
rescale để `RMS(Wo u)/RMS(h)=1e-3` với tolerance `[0.5e-3,2e-3]`; `h+delta`
được cộng ở FP32 trước khi cast. Variant này mất exact parity nhưng gần
endpoint. Zero chỉ là parity/smoke control; tiny là candidate training được
đăng ký trước, không chọn dựa trên test score.

Loss là mean NLL của **phân phối P hoàn chỉnh**:

    l0 = log P0(y)
    ls = log Ps(y)
    lc = log Pcopy(y), hoặc -inf nếu y không copyable
    lp = logsumexp(log(pi_base)+l0,
                   log(pi_sem)+ls,
                   log(pi_copy)+lc)
    loss_sum = -sum(lp trên label != -100)

Không dùng tổng `pi_k * CE_k`; đó là mục tiêu khác và có gradient khác. Trong
mỗi logical accumulation window, tính `N_global=all_reduce(sum_r local valid
target tokens_r)` rồi backward với `loss_r=world_size*local_loss_sum_r/N_global`;
DDP average đúng một lần, không chia thêm accumulation/world size. Clip sau
accumulate/synchronize, trước optimizer step. Log pre/post norm, coefficient và
tỷ lệ step bị clip. Giữ common clip `1.0` ở control; clipping không được tính là
đóng góp v5.1.

Independent control phải khởi tạo bằng `l_base=log(clamp(1-g_route,eps,1)) +
r_base`, `l_copy=log(clamp(g_route,eps,1)) + r_copy`, `l_sem=-20+r_sem`, với
`r_base/r_copy/r_sem=0` và expert rỗng mask `-inf`. Không dùng `logit(g)`; sau
softmax nó không khôi phục tỷ lệ copy legacy. `base_floor` được áp ở control
này và làm thay đổi prior copy khi `g` lớn.

## 7. Acceptance tests

| ID | Kiểm tra | Bằng chứng |
|---|---|---|
| C1 | Trunk/copy endpoint | Tắt reader, cùng weights/input, output trùng `new`; optimizer group không mất params |
| C2 | V2 endpoint | `semantic_only_v2` trùng probability/loss/gradient v2 flat; raw logits chỉ so sánh khi control dùng gauge `Zs`, vì main gauge là `Z0` |
| C3 | Mixture oracle | FP64/FP32, mass=1, duplicate IDs, copyable/non-copyable |
| C4 | Identity | Zero endpoint đối chiếu trực tiếp với kernel `_mix_logits` của `new`; `Ps=P0` không được cộng bằng một oracle khác |
| C5 | Gradient | Endpoint backward đầu: legacy base/copy/trunk vẫn nhận gradient, `Wo` hữu hạn; Q/K/V/router có thể zero ở endpoint nhưng candidate tiny phải khác zero ở các step đầu trên batch có signal |
| C6 | Source fallback | Semantic rỗng → `alpha=0`; copy rỗng → `g=0`; cả hai rỗng → `P0` |
| C7 | No leakage | Future target không đổi logits quá khứ; router không nhận gold/reference |
| C8 | Chunking | Dense/chunked loss và nhóm gradient tương đương |
| C9 | DDP | 1/2 GPU cùng canonical global examples, valid-token denominator và update schedule cho update gần nhau |
| C10 | Dtype | FP32/BF16 finite; gates/reductions/logsumexp ở FP32 |
| C11 | Gauge | `log(P)+Z0` giữ endpoint với repetition penalty 1.05 và probe 1.0 |
| C12 | Cache | Teacher-forced/incremental, compaction/reorder/source reset tương đương |
| C13 | Checkpoint | Save/load/resume giữ output, optimizer, RNG, scheduler; từ chối mismatch |
| C14 | Sampling | Temperature/top-p hợp lệ và áp sau final mixture; seed tái lập |
| C15 | Profile/runner | Batch, optimizer steps, scheduler, clip, dtype, source budget, manifest hash và evaluator được in từ resolved config |
| C16 | Fresh evaluation | Prediction JSONL từ chối nếu checkpoint/config/split/decoder fingerprint không khớp; không trả metric từ file cũ trước khi load checkpoint |
| C17 | Copy-gradient isolation | Kiểm tra Jacobian `dP/dg=Pcopy-P0` và alpha không tạo đường trực tiếp vào g; log `dL/dg` thực tế riêng, không yêu cầu nó khớp copy-only; shared-trunk drift được log riêng |
| C18 | Fixed-small-set | CE giảm rõ trên 16–32 examples cố định; mask semantic/copy và EOS hoạt động trước PubMed |
| C19 | Manifest replay | 1GPU/2GPU phát lại cùng update IDs, token denominator, tail policy và scheduler step |

Tolerance FP32 khởi điểm `rtol=1e-5, atol=1e-6`; BF16 phải dùng sai số đo từ baseline, không dùng tolerance để che NaN hoặc sai mask. CPU/Gloo chỉ là test logic; cần CUDA/NCCL cho C9/C10.

## 8. Ma trận đánh giá

Các run quyết định dùng cùng common CE recipe, source policy, tokenizer protocol, Perl ROUGE-1.5.5 và checkpoint selection bằng validation:

- A: `new/common-control`.
- B: `update_v2/common-control`.
- C: v4 negative control nếu tái lập được recipe.
- D: v5.1-zero-parity, `K=H0,V=H0,Wo=0`; chỉ dùng để kiểm tra endpoint/smoke.
- E: v5.1-tiny, chỉ đổi `Wo` init; candidate training chính sau pilot.
- F: independent capped simplex.
- G: `K=M,V=H0`.
- H: hidden interpolation.
- I: constant alpha.

Pilot một seed để loại implementation lỗi; A/B common-control phải chạy trước
để kiểm tra trade-off lịch sử. Nếu trade-off không tái lập, đó là cổng diễn giải
nhân quả, không tự động cấm một pilot D/E nhỏ để kiểm tra cải thiện trực tiếp.
Full candidate tối thiểu ba seed cố định `42/43/44`. Báo từng seed, mean, spread,
output length, copy rate, truncation, branch responsibility, CE từng branch,
delta norm, gradient theo step và throughput.

Primary generation giữ greedy và processors hiện hành. Thêm probe `repetition_penalty=1.0` và gauge `C=Z0`; không dùng probe thuận lợi để thay headline protocol.

Thắng kiến trúc được xét trước hết bằng D/E/F so với A/B dưới cùng recipe;
thắng hệ thống so với các mốc lịch sử và T5Gemma được báo riêng. Một candidate
chỉ được gọi là thắng hệ thống nếu vượt `new` ở R1/RL, vượt `update_v2` và
T5Gemma ở R2, cùng một checkpoint validation và lặp lại với paired bootstrap
qua seed. Nếu F tốt hơn E nhưng R2 giảm, giữ E. Nếu H ngang E, không claim
probability fusion là nguyên nhân. Nếu kết quả chỉ tăng R2 nhưng R1/RL tiếp tục
giảm, dừng thêm module và phân tích source visibility/length/copy trước.

## 9. Hướng mở sau khi v5.1 đã được cô lập

Chỉ thử null slot khi diagnostics cho thấy semantic reader đọc nhiễu:

    K = [K_source, k_null]
    V = [V_source, 0]
    evidence_mass = 1 - A_null
    delta = evidence_mass * bounded_residual(...)

Có thể thử copy-protected semantic residual hoặc entropy/evidence gating sau đó. Hai hướng này không nằm trong main vì có prior art và làm thay đổi thêm nhiều đường điều khiển.

SDACL/contrastive training được giữ ở nhánh nghiên cứu sau, không dùng để che failure của architecture CE-only. Nếu v5.1 tăng ROUGE nhưng chưa có bằng chứng giảm hallucination, phải báo riêng hai kết quả.

## 10. Artifact phải lưu

Mỗi run lưu:

- resolved config và architecture fingerprint;
- commit, pretrained revisions, tokenizer/alignment hashes, seed, world size và dtype;
- dataset split/source/reference manifest, preprocessing và truncation statistics;
- predictions có sample ID/checkpoint provenance;
- raw ROUGE-1.5.5 output, command, Perl/XML dependencies và detailed per-example scores;
- loss denominator, branch diagnostics, output length/copy rate và VRAM/throughput profile.

Không trộn predictions giữa checkpoint, seed hoặc evaluator. Không dùng test để chọn alpha, gauge, epoch, length hay mode.

## 11. Nguồn và giới hạn

Quyết định này dựa trên [bản nghiên cứu và phản biện chi tiết](DECISION_REPORT.md), [audit kiến trúc](ARCHITECTURE_AUDIT.md), [kế hoạch đánh giá](EVALUATION_PLAN.md) và các công trình gốc được dẫn trong đó. Mixture-of-Softmaxes cung cấp tiền lệ cho probability mixture nhưng cũng cảnh báo chi phí output và overfitting khi tăng số mixture.^1 Pointer-generator và generalized pointer hỗ trợ tách đường copy khỏi abstraction.^2,^3 Gated Attention cho thấy tác dụng phụ thuộc vị trí gate, không biện minh cho việc lặp lại planner/head gate của v4.^4 T5Gemma 2 cho thấy cross-attention ở mọi layer là một quyết định có thể quan trọng, nhưng kết quả của họ không dự báo ROUGE PubMed của EviSeq.^5

Không có checkpoint/predictions lịch sử đầy đủ để suy ra causal attribution từ các điểm ROUGE đã báo. Plan này đưa ra cách kiểm chứng và điều kiện bác bỏ; nó không phải cam kết v5.1 sẽ thắng.

[^1]: Z. Yang et al., “Breaking the Softmax Bottleneck,” ICLR 2018, [arXiv](https://arxiv.org/html/1711.03953v3).
[^2]: A. See, P. Liu, C. Manning, “Get To The Point,” ACL 2017, [ACL Anthology](https://aclanthology.org/P17-1099/).
[^3]: X. Shen et al., “Improving Latent Alignment in Text Summarization by Generalizing the Pointer Generator,” EMNLP-IJCNLP 2019, [ACL Anthology](https://aclanthology.org/D19-1390/).
[^4]: Z. Qiu et al., “Gated Attention for Large Language Models,” 2025, [arXiv](https://arxiv.org/html/2505.06708v1).
[^5]: B. Zhang et al., “T5Gemma 2: Seeing, Reading, and Understanding Longer,” 2025, [arXiv](https://arxiv.org/html/2512.14856v1).
