# Kiểm định giả thuyết đóng góp của bridge AFMR

Ngày 2026-09-18. Đây là hồ sơ cơ sở và triển khai thử nghiệm, **chưa phải báo cáo rằng full vượt w/o bridge**. Các số AFMR dưới đây do người dùng cung cấp; không có checkpoint, source hoặc prediction cặp của ablation trên máy này để tính lại.

**Đọc thêm [kiểm định sâu lần hai](second_pass.md) và [kiểm định lần ba](third_pass_audit.md):** sau bản nháp đầu, đã sửa chuẩn hóa evidence loss trên valid target tokens (kể cả accumulation/DDP) và chặn n-gram băng qua ranh giới câu. Loss lexical vẫn có giới hạn quan trọng về paraphrase, dấu câu và các occurrence lặp; `λ=0.002` chưa được xác nhận trên validation. Kiểm định lần ba chuyển PubMed mặc định về `λ=0`: ablation full-vs-direct phải cùng CE; bật auxiliary bằng biến môi trường riêng để đo đóng góp training.

## Vấn đề đã quan sát

- Full value-anchor trước đây: R-1/2/L = **49.686/22.153/45.939**.
- `direct_projection` w/o bridge: **49.671/22.135/45.956**. Full hơn R-1 0.015 và R-2 0.018, kém R-L 0.017. Chênh lệch nhỏ, chưa chứng minh bridge hữu ích.
- Full với contextual value ở **run người dùng đã đo**: **49.657/22.098/45.920**. Kém cả ba điểm so với direct control. Code có thêm một biến thể contextual-value cục bộ tùy chọn, nhưng biến thể đó chưa có score riêng; recipe mới tắt nhánh này.

Code audit cho thấy AFMR có `depth_out`, `feature_up`, `focus_output` zero-init. Vì thế lúc khởi tạo full gần như cùng hàm với direct projection; CE không bắt buộc bridge học một prior phân biệt evidence. Đây là **chẩn đoán khả dĩ**, chưa xác định là nguyên nhân duy nhất từ ba điểm aggregate. Điểm R-L của full cũ còn thấp hơn direct.

## Cơ sở và giới hạn của suy luận

1. [TopDownFormer](sources/s01_topdownformer.md) dùng top-down token correction và báo cáo trên PubMed `48.34/21.40/44.22` so với no-top-down `46.97/20.23/42.88`. Nó xác nhận global-to-token read **có thể** hữu ích trong một kiến trúc khác, không dự báo điểm AFMR. [Paper](https://aclanthology.org/2023.findings-eacl.94.pdf).
2. [SEASON](sources/s02_season.md) kết hợp salience prediction với cross-attention keys, giữ original values; CNN/DailyMail predicted-salience đạt `46.27/22.64/43.08` so với BART reproduction `44.21/21.23/41.17`. **Chi tiết quan trọng:** paper dùng gold salience trong generation lúc train và predicted salience lúc test. Bản AFMR thử nghiệm này dùng **predicted source bias ở cả train và test**, còn gold-derived evidence chỉ làm auxiliary loss, tránh lệch input kiểu đó. Kết quả của SEASON không chứng minh biến thể AFMR sẽ tốt hơn. [Paper](https://aclanthology.org/2022.emnlp-main.409.pdf).
3. [HierGNN](sources/s03_hiergnn.md) và [QA salient spans](sources/s04_query_salience_deutsch_roth.md) hỗ trợ nguyên tắc phải đưa tín hiệu chọn nguồn tới decoder, và silver labels chỉ có ở train. Chúng dùng sentence graph / QA span selector riêng, không phải cùng implementation. [HierGNN](https://aclanthology.org/2022.emnlp-main.355.pdf), [QA spans](https://aclanthology.org/2023.eacl-main.42.pdf).
4. Phản chứng: [PROM](sources/s05_prom.md) cho arXiv chỉ tăng R-1/R-2 `+0.06/+0.08`, R-L giảm `0.04` so với BART row; [CODI](sources/s06_codi_spancopy.md) thêm global relevance trên PubMed tăng R-1 `0.06` nhưng R-2 giảm `0.04` và source precision giảm `68.91→66.91`. Vì vậy copy/lexical prior có thể chỉ tạo thay đổi nhỏ hoặc hại nội dung. [PROM](https://aclanthology.org/2024.lrec-main.1148.pdf), [CODI](https://aclanthology.org/2023.codi-1.9.pdf).

Các nguồn trên là nhiều nghiên cứu độc lập nhưng **đều là paper học thuật**; chưa có ba loại nguồn độc lập cùng chứng minh giả thuyết tăng ROUGE AFMR. Code audit là bằng chứng cơ chế, còn điểm người dùng cung cấp là bằng chứng hiệu năng hiện tại; cả hai không đủ để kết luận kết quả sau sửa. Theo quy tắc triangulation, claim “full mới > w/o bridge” hiện có **insufficient evidence**.

## Biến thể đã đặt trong `src/eviseq_new`

Giữ pretrained encoder, decoder, `H0` value anchor và grounded copy. Tắt **mặc định** nhánh contextual value đã giảm điểm, nhưng giữ code nhánh đó để có thể kiểm tra riêng. Bridge hiện có sinh `source_bias` từ nhiều cửa sổ nguồn; cùng prior dự đoán này đi vào semantic cross-attention và grounded-copy attention. Điểm khác so với full CE-only là thêm loss train-only cho chính prior này:

```text
L = L_CE + λ L_evidence
L_evidence = weighted_mean_rows softplus(margin - mean(bias on matched phrases)
                                      + mean(bias on other visible source tokens))
```

Silver positive lấy từ bigram/trigram exact match giữa **phần source encoder còn nhìn thấy** và **phần target decoder còn học sau truncation**; trigram được trọng số 2, bigram trọng số 1. Không có matching hợp lệ thì hàng đó không góp auxiliary loss. Các nhãn gold không đi vào encoder/decoder, copy state hay generation; inference vẫn dùng `source_bias` do bridge tự dự đoán. Không sinh candidates và không thêm lượt decoder. `λ=0.002` là **hyperparameter thử nghiệm cho arm C**, không phải mặc định PubMed hay giá trị tối ưu được chứng minh bởi paper; margin là `0.5`. Tiny-model tại khởi tạo đo gradient của `focus_output` từ CE khoảng `8.93e-6`; từ auxiliary sau nhân `0.2` khoảng `1.13e-2`, tức hơn ba bậc độ lớn. Hạ xuống `0.002` cho gradient auxiliary ngoại suy tuyến tính khoảng `1.13e-4`, vẫn khoảng **13 lần** CE ở tiny-model. Đây là cảnh báo phải theo dõi validation và tỷ lệ nhãn hợp lệ, không phải chứng cứ hệ số đã tối ưu cho PubMed. Focus strength và các thông số bridge khác giữ như base cũ để hạn chế biến gây nhiễu.

Khác biệt so với SEASON/PROM là prior của AFMR dùng chung cho hai đường semantic và copy, trong một interface ghép hai backbone với values giữ ở `H0`; nhưng salience prediction, auxiliary gold labels, key/value separation và phrase matching **đều có tiền lệ**. Novelty riêng chỉ có thể viết sau khi chứng minh hai đường phối hợp thực sự tạo lợi ích vượt các đối chứng thích hợp.

## Phản biện và tiêu chuẩn chấp nhận

- Weak labels dễ đánh dấu cụm chung và bỏ qua paraphrase/quan hệ định lượng; tăng match phrase có thể giảm tính tự nhiên hoặc faithfulness. Không gọi nhãn này là factuality ground truth.
- `source_bias` là prior tĩnh theo source/prompt; decoder QK vẫn query-dependent. Auxiliary có thể cải thiện salience loss nhưng không đổi output nếu cross/copy gate bỏ qua prior.
- Không được biến full-vs-direct thành kiểm tra “kiến trúc thuần”: full mới có thêm training loss. Phải báo cáo riêng **A:** direct CE, **B:** full CE (`λ=0`), **C:** full + evidence (`λ=0.002`). B/A đo đóng góp kiến trúc trước đây; C/B đo training signal; C/A đo hệ thống cuối. Nếu chỉ C>A mà B≈A, contribution là **bridge được giám sát evidence**, không phải cấu trúc bridge tự thân.
- Theo dõi `salience_coverage`: tỷ lệ ví dụ train/validation có cả token evidence dương và âm. Validation có tính auxiliary loss để chẩn đoán nhưng **không đưa nhãn vào generation**; recipe PubMed dùng `save_best: false` và chấm `last.pt`, nên nhãn validation không chọn checkpoint trong protocol hiện tại. Nếu coverage thấp, không thể diễn giải ROUGE của C là kiểm chứng mạnh cho loss mới; cần xem lại nhãn hoặc dừng hướng này.
- Cùng PPLX encoder, Qwen decoder, tập đã preprocess, prompt, seed, batch × accum, số update, optimizer, clip, max length, checkpoint rule (`last.pt`), decoding deterministic và ROUGE-1.5.5. Lặp nhiều seed hoặc paired bootstrap; không điều chỉnh hyperparameter trên test.
- Theo mục tiêu người dùng, mốc thành công thực dụng cho **đóng góp kiến trúc bridge** là **B−A ≥ 0.30 trên cả R-1, R-2 và R-L** ở cùng protocol, rồi kiểm tra độ bền qua seed/paired bootstrap. Nếu chỉ C đạt ngưỡng so với A, claim phải nói rõ đó là hệ thống gồm bridge **và** objective evidence, còn phần đóng góp kiến trúc riêng là B−A. Đây là ngưỡng quyết định thử nghiệm, không phải hiệu quả được paper hứa hẹn. Nếu một arm chỉ tăng R-2 mà R-1/L giảm, report đúng trade-off. Kiểm tra thêm output khi mask prior, salience positive-negative separation và source/copy precision.

## Verification cục bộ

Tiny-model tests xác nhận loss auxiliary có gradient trực tiếp vào `focus_output`, optimizer đổi weight, không truyền gold labels khi inference; so sánh accumulation và large batch cho cùng update. Full và direct khởi tạo cùng seed cho **logits và CE giống hệt** trên fixture nhỏ; width projection cũng khởi tạo giống khi kích thước encoder/decoder khác nhau. Không download model lớn. Các test này xác nhận code path, **không** xác nhận score tăng.

## Chạy đối chứng trên server

Từ `src/eviseq_new`, giữ một seed, một bộ dữ liệu và các tham số khác như `afmr_pubmed.yaml`; dùng ba output dir mới. `run_pubmed_pair.sh` đã train rồi eval `last.pt`; đặt `ROUGE155_SCRIPT` nếu cần chấm ROUGE-1.5.5 tự động.

```bash
AFMR_ENCODERS=pplx AFMR_BRIDGE_MODE=direct_projection AFMR_SALIENCE_WEIGHT=0 AFMR_OUTPUT_DIR="$PWD/runs/afmr/bridge_A_direct" CUDA_VISIBLE_DEVICES=0 bash scripts/run_pubmed_pair.sh
AFMR_ENCODERS=pplx AFMR_BRIDGE_MODE=afmr AFMR_SALIENCE_WEIGHT=0 AFMR_OUTPUT_DIR="$PWD/runs/afmr/bridge_B_ce" CUDA_VISIBLE_DEVICES=0 bash scripts/run_pubmed_pair.sh
AFMR_ENCODERS=pplx AFMR_BRIDGE_MODE=afmr AFMR_SALIENCE_WEIGHT=0.002 AFMR_OUTPUT_DIR="$PWD/runs/afmr/bridge_C_evidence" CUDA_VISIBLE_DEVICES=0 bash scripts/run_pubmed_pair.sh
```

Các lệnh trên chỉ là protocol thực nghiệm, không phải kết quả. Chọn hệ số/seed trên validation, khóa protocol rồi mới dùng test. Nếu nhiều seed, giữ bộ seed giống nhau cho cả ba arm và công bố độ biến thiên.
