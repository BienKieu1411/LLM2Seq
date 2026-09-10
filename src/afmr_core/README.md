# AFMR core: copy-mass-preserving semantic mixture

AFMR core là implementation cục bộ sau deep research và phản biện chéo ba
`luna_worker`. Package, configs, runner, checkpoint provenance và acceptance
tests nằm cùng folder; chưa có checkpoint PubMed hoặc kết quả ROUGE AFMR.

Mục tiêu được tách thành architecture gate và system stretch: dưới common
recipe, vượt R1/RL của `eviseq_new` và R2 của `update_v2`; sau khi tái lập
T5Gemma công bằng, stretch là vượt thêm R2 của T5Gemma. Chưa có kết quả nào
được bảo đảm.
Main giữ AFMR, cross-attention, grounded copy của `new/v2` và flat semantic
read rank128 của v2. Reader phải giữ đúng thứ tự `H0n=RMS(H0)`,
`K=RMS(Wk(H0n))`, `V=Wv(H0n)`. V4 hierarchy/planner/coverage/4×128 bị loại
khỏi main.

## Công thức main

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

`g` là copy gate legacy; semantic không còn bị nhân trực tiếp với `(1-g)beta`.
`g_route` là bản detached chỉ dùng cho router/cap; `pi_copy` vẫn dùng `g` có
gradient. `generate_reserve=0.05` không phải floor tuyệt đối của `pi_base` khi
`g` lớn. Vì vậy tên chính xác là copy-mass-preserving capped simplex;
`independent_capped_simplex` bên dưới là control có trade-off khác. Khi `Wo=0`,
`delta=0`, `Ps=P0`, nên forward khôi phục phân phối của `new`. Main candidate
dùng `Wo=tiny_rms_1e-3` calibration deterministic; zero chỉ parity/smoke.
Main dùng `alpha_max=0.20`, `K=H0,V=H0`, không semantic gate `.05` kép và RMS
cap `.10`.

Source rỗng tắt semantic; copy source rỗng tắt copy; cả hai rỗng trả `P0`. Output gauge main:

    Z0 = logsumexp(z0)
    output_logits = log(P) + Z0

Temperature/top-k/top-p chỉ dùng cho sampled generation/candidates, không dùng benchmark greedy và không dùng trong training.

## Các variant phải chạy

- `AFMR-zero-parity`: endpoint/smoke của công thức main, `K=H0,V=H0`, `Wo=0`.
- `AFMR-tiny`: candidate training, `Wo` init correction khoảng `1e-3×RMS(h)` theo calibration batch cố định không dùng label.
- `legacy_semantic`: control legacy dùng `readout_mode=legacy_copy_mixture`, cap v2,
  inner gate `.05` và output projection zero để đối chiếu công bằng với v2.
- `AFMR-independent`: capped three-way simplex có base floor/copy feature.
- `AFMR-KM`: `K=M,V=H0`, source prior giảm/tắt để kiểm tra double-focus.
- `hidden-interpolation`: cùng reader nhưng fusion trong hidden.
- `constant-alpha`: kiểm tra learned router có hơn mixture cố định hay không.

Null slot, entropy/evidence gating và copy-protected residual chỉ thử sau khi các variant trên đã cô lập được nguyên nhân.

## Tài liệu

- [DECISION_REPORT.md](doc/DECISION_REPORT.md): deep-research report, phản biện chéo, quyết định và nguồn.
- [ARCHITECTURE_AUDIT.md](doc/ARCHITECTURE_AUDIT.md): đối chiếu code `new`, `update_v2`, v4.
- [RESEARCH.md](doc/RESEARCH.md): cơ sở paper/technical report và ranh giới novelty.
- [DESIGN.md](doc/DESIGN.md): shapes, gradient, loss, gauge, cache và dtype.
- [IMPLEMENTATION_PLAN.md](doc/IMPLEMENTATION_PLAN.md): thứ tự code, acceptance tests, ablation và điều kiện dừng.
- [EVALUATION_PLAN.md](doc/EVALUATION_PLAN.md): protocol fairness, checkpoint selection và statistical reporting.
- [ARCHITECTURE_TARGET_AUDIT.md](doc/ARCHITECTURE_TARGET_AUDIT.md): audit tổng hợp, hợp đồng tensor/xác suất/gradient/DDP và câu hỏi tự kiểm tra.
- [VERIFICATION_REPORT.md](doc/VERIFICATION_REPORT.md): xác minh độc lập điểm mạnh/điểm yếu và trạng thái đã chứng minh/chưa chứng minh.
- [IMPLEMENTATION_EVIDENCE.md](doc/IMPLEMENTATION_EVIDENCE.md): lệnh chạy, test result và gap analysis C1–C19.
- [KARPATHY_REVIEW.md](doc/KARPATHY_REVIEW.md): bounded code-quality review and closed findings.

## Chạy cục bộ

Smoke test dùng hai model `__tiny__` được tạo trong Transformers, không tải
model từ Hugging Face:

    PYTHON=/path/to/python ./run_afmr.sh smoke

Runner PubMed mặc định khóa hai GPU, batch 84 mỗi GPU và accumulation 1. Nó
chỉ kiểm tra model/dataset local rồi mới train; `DRY_RUN=true` chỉ sinh config
và in effective batch. Sau train, runner đánh giá checkpoint `best.pt` được
chọn bằng validation CE (có thể ghi đè bằng biến `CHECKPOINT`):

    PYTHON=/path/to/python DRY_RUN=true ./scripts/run_pubmed_pair.sh

Đổi `NPROC_PER_NODE`, `BATCH_SIZE` hoặc `GRADIENT_ACCUMULATION_STEPS` bằng biến
môi trường khi đã đăng ký protocol tương ứng. Benchmark vẫn greedy; `temperature`
và `top_p` chỉ được dùng bởi API sampled candidates.

## Mốc tham chiếu đã báo

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| `new` | 49.626 | 21.901 | 45.895 |
| `update_v2` | 49.488 | 21.953 | 45.776 |
| T5Gemma | 49.580 | 21.990 | 45.463 |

Đây là các điểm đã báo trong dự án, chưa được tái lập từ cùng predictions/resolved config trong folder này. Chỉ gọi AFMR thắng sau common recipe, Perl ROUGE-1.5.5, validation checkpoint selection và tối thiểu ba seed.
