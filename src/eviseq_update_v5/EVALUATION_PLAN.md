# Kế hoạch đánh giá EviSeq v5.1

## 1. Câu hỏi và tiêu chí

Câu hỏi chính: một semantic vocabulary branch có bị mất lợi ích khi đặt bên
trong copy gate hay không, và copy-mass-preserving capped simplex có thể cải
thiện R1/ROUGE-L mà vẫn giữ R2 không?

Mốc đã báo trong dự án:

- `new`: R1 49.626, R2 21.901, RL 45.895.
- `update_v2`: R1 49.488, R2 21.953, RL 45.776.
- T5Gemma: R1 49.580, R2 21.990, RL 45.463.

Tách hai gate để không trộn claim kiến trúc với claim hệ thống:

- **Architecture gate:** dưới cùng common EviSeq recipe, một checkpoint v5.1 có `R1>49.626`, `R2>21.953` và `RL>45.895`.
- **System stretch:** sau khi tái lập T5Gemma với split, selection rule và evaluator minh bạch, một checkpoint duy nhất có `R1>49.626`, `R2>21.990` và `RL>45.895`.

Các số này chỉ là target tham chiếu cho đến khi predictions, resolved config và evaluator được tái lập; nếu baseline rerun lệch mốc lịch sử thì common rerun là mốc causal chính.

## 2. Protocol chung

Tất cả EviSeq controls dùng cùng pretrained base state theo từng seed, dataset split, tokenizer/alignment version, source policy, optimizer/stages, learning rate, scheduler, token-normalized CE, dtype và checkpoint budget. Tạo `canonical_global_batch_manifest` để khóa thứ tự example IDs, valid target tokens, optimizer update IDs và tail policy. Common control nên dùng effective global batch 96 để nối mốc `new`: 1 GPU×48×accum2 hoặc 2 GPU×48×accum1. Nếu tài nguyên bắt buộc dùng global168, mọi EviSeq controls phải dùng global168 và tái train.

Trong mỗi logical accumulation window, tính `N_global=all_reduce(sum_r local_valid_target_tokens_r)` và backward `loss_r=world_size*local_loss_sum_r/N_global`; DDP average đúng một lần, không chia thêm theo world size/accumulation. Clip sau backward cuối và gradient synchronization.

T5Gemma được báo riêng theo recipe của nó; không ép dùng optimizer/backbone của EviSeq để tạo fairness giả. So sánh hệ thống phải cùng split, raw visible source policy, generation protocol và ngân sách chọn checkpoint. Config T5Gemma lịch sử có `eval_strategy: no`, nên phải rerun theo selection rule chung hoặc dùng final-checkpoint protocol cho tất cả. Báo source characters/tokens và truncation rate vì limit 4096 trên các tokenizer khác nhau không đảm bảo nhìn thấy cùng văn bản.

Canonical test dùng Perl ROUGE-1.5.5 với cùng script/data/options cho mọi model. Development chọn checkpoint bằng `selection_metric=validation_ce`, `save_best=true` và cùng tie-break; không dùng test để chọn alpha, cap, gauge, length hay seed. Evaluator kiểm tra checkpoint/config/split/decoder fingerprint trước khi đọc prediction cache; prediction chỉ khớp ID/reference prefix vẫn bị từ chối nếu stale.

Processed source/reference manifest phải có hash và hard-fail nếu phát hiện
cross-split content overlap. `ALLOW_CROSS_SPLIT_CONTENT` mặc định false, không
được bật trong runner để làm đẹp điểm. Optimizer groups, no-decay policy và
global cosine schedule phải giống nhau giữa A/B/E/F; không reset scheduler khi
đổi stage.

## 3. Ma trận thí nghiệm

| ID | Cấu hình | Mục đích |
|---|---|---|
| A | `new/common-control` | Mốc copy-only R1/RL |
| B | `update_v2/common-control` | Mốc flat semantic và R2 |
| C | v4 hiện tại | Negative control nếu recipe tái lập được |
| D | `v5.1-zero-parity` | Endpoint/smoke: copy-mass-preserving simplex, K=H0/V=H0, Wo=0 |
| E | `v5.1-tiny` | Candidate training: như D nhưng Wo calibration deterministic ~1e-3 RMS(h) |
| F | `v5.1-independent` | Capped three-way simplex có base floor, copy/semantic mass độc lập |
| G | `v5.1-KM` | K=M/V=H0, prior giảm/tắt để đo double-focus |
| H | `hidden-interpolation` | Cùng reader nhưng fusion trong hidden |
| I | `constant-alpha` | Kiểm tra learned router có hơn alpha cố định |
| J | `gauge-probe` | D/E với `C=Z0` và repetition penalty 1.0/1.05 |

A chỉ là endpoint copy-only, D là parity control; E là candidate training sau
pilot. F là ứng viên high-upside nhưng có rủi ro làm giảm copy mass; chỉ
promotion nếu R2 không giảm. G không được đưa vào main âm thầm. Không tăng lên
4×128 trước khi có evidence rank128 thiếu capacity.

## 4. Diagnostics cần ghi

Mỗi validation run ghi:

- `CE_base`, `CE_sem`, `CE_copy`, `CE_mixture` trên cùng prefix;
- `alpha`, `g`, `pi_base`, `pi_sem`, `pi_copy` theo token và theo sample;
- `log Ps(y)-log P0(y)` cho token copyable/non-copyable;
- RMS(delta)/RMS(h), logit-margin changes, tỷ lệ delta bị rounding mất;
- copy rate, copy precision, repeated n-grams, output/reference length, EOS và truncation;
- source position/evidence mass và source-visible vs source-invisible unigrams/bigrams;
- gradient norms của Wo, Q/K/V, router, AFMR, decoder và copy;
- gradient trực tiếp của copy gate so với A, tách khỏi gradient shared trunk;
- train/eval throughput, dense/chunked memory, peak VRAM.

Ghi thêm `alpha_cap_hit_rate`, `g` quantiles, `copy_entropy_norm`,
`grad_norm_pre_clip`, `clip_coefficient`, `grad_norm_post_clip` và
`fraction_steps_clipped`. `copy_support_mass` không phải feature main vì tổng
attention trên support hợp lệ thường bằng một.

Trước khi khóa `alpha_max=0.20`, đo `g` trên validation của A và chạy một pilot
validation với `alpha_max∈{0.10,0.20,0.50}`. `generate_reserve=0.05` chỉ là
phần mass ưu tiên giữ lại, không được diễn giải thành floor tuyệt đối khi `g`
cao. Independent simplex dùng `base_floor=0.05` riêng với main.

Các diagnostics dùng gold chỉ để đo sau forward, không đưa gold/reference vào
router hay loss mới. Main dùng `g_route=stop_gradient(g)` và detached copy
features; một variant `router_sees_copy` có thể bật đường này nhưng phải báo
tên riêng.

## 5. Gauge và decoding

Benchmark chính giữ greedy và processors hiện hành: `max_new_tokens`, `min_new_tokens`, repetition penalty và no-repeat n-gram như PubMed config. Main output:

    Z0 = logsumexp(z0)
    output_logits = log(P) + Z0

Chạy thêm probe repetition penalty 1.0 để tách probability fusion khỏi scalar offset. Temperature/top-p chỉ áp lên final mixed scores trong sampled generation/candidate API; không sample từng expert rồi mới mix và không dùng sampled output làm training target.

## 6. Seeds và bất định

Pilot một seed chỉ để bắt lỗi và loại ứng viên rõ ràng không học; không gọi
pilot là chiến thắng. A/B common-control phải chạy trước để kiểm tra trade-off
lịch sử. Nếu trade-off không tái lập, đó là cổng diễn giải nhân quả và phải hạ
claim mixture; vẫn có thể chạy một pilot E/F nhỏ để kiểm tra cải thiện trực tiếp,
nhưng không chạy sweep lớn theo giả thuyết cũ. Candidate E/F và baseline A/B
chạy tối thiểu ba seed định trước, ví dụ `42/43/44`. Mỗi seed chọn checkpoint
bằng cùng validation rule.

Trên test báo từng seed, mean và spread. Đăng ký trước paired percentile
bootstrap theo document (10.000 resamples, two-sided 95% CI) trên cùng sample
IDs; điều chỉnh family-wise bằng Holm ở mức `alpha=0.05` cho ba primary
comparisons. Báo riêng khoảng bất định giữa các seed; bootstrap theo document
không thay thế repeated training seeds và không được dùng để chọn checkpoint.

Thắng kiến trúc được xét bằng E/F so với A/B cùng recipe. Thắng hệ thống so
với `new` lịch sử và T5Gemma được báo riêng. Candidate chỉ được claim vượt
T5Gemma khi cả ba chênh lệch dương, lặp lại qua seeds và khoảng bất định không
bao phủ no-gain theo rule đã định trước. Không tạo một hàng kết quả bằng cách
lấy R1/R2/RL từ các checkpoint khác nhau.

## 7. Điều kiện diễn giải

- E thắng A/B và F: probability mixture với copy-mass-preserving routing có evidence riêng.
- F thắng E nhưng R2 giảm: giữ E; không hy sinh R2 để tăng một metric.
- H ngang E: không claim probability fusion hơn hidden fusion.
- I ngang E: learned router chưa chứng minh cần thiết; giữ alpha cố định nếu rẻ hơn.
- G tốt hơn E: xem xét promotion chỉ sau khi kiểm tra source prior double-count.
- Chỉ R2 tăng trong khi R1/RL giảm: phân tích length/copy/source visibility trước, không thêm planner hoặc contrastive loss để che regression.
- ROUGE tăng nhưng hallucination chưa được đo: không tuyên bố factuality gain. Factuality/SDACL là vòng riêng sau khi architecture đã được cô lập.

## 8. Artifact bắt buộc

Lưu run manifest, resolved config, commit, pretrained/tokenizer hashes, seed/world
size/dtype, dataset split/source/reference hashes, preprocessing và truncation
statistics, predictions có sample ID và fingerprint của checkpoint/config/split/
decoder revision, raw Perl ROUGE output/command/dependencies, per-document
scores, loss denominator, branch diagnostics và resource profile. Evaluator phải
từ chối prediction hoàn chỉnh nhưng stale trước khi trả metric.

Nếu ROUGE-1.5.5 không chạy vì thiếu Perl XML modules, phải lưu lỗi môi trường và không trộn kết quả Python ROUGE-1.0.0 vào bảng headline.
