# Kiểm định sâu giả thuyết bridge AFMR tăng ROUGE

Ngày 2026-09-18. Đọc cùng `final_report.md`. Tài liệu này phân biệt **kết quả đã đo**, **lỗi/cơ chế kiểm tra được trong code**, và **giả thuyết phải train/eval mới biết**. Không có checkpoint và PubMed JSONL trên máy hiện tại để chấm lại ROUGE hoặc đo tỷ lệ nhãn.

## Kết luận hiện tại

Chưa có cơ sở nói bridge AFMR hiện tại làm full tốt hơn `direct_projection`. Kết quả người dùng cung cấp: full 49.686/22.153/45.939, direct 49.671/22.135/45.956; chênh lệch **+0.015/+0.018/−0.017**. Biến thể contextual-value đã đo là 49.657/22.098/45.920, tức kém direct cả ba metric. Khoảng cách quan sát được không gần ngưỡng mong muốn **+0.30 ở cả R-1/R-2/R-L**, và không đủ để gán nguyên nhân cho một module từ điểm tổng hợp đơn lẻ. Việc chỉ tăng gate, rank, số head hoặc hệ số loss không có bảo đảm, dễ khuếch đại tín hiệu sai.

Nếu dùng đúng mốc direct trên, điều kiện số học cho gain ít nhất 0.30 ở cả ba metric là **R-1 ≥49.971, R-2 ≥22.435, R-L ≥46.256** trong cùng giao thức. Đây chỉ là ngưỡng quyết định do người dùng đặt, không phải dự đoán kết quả.

Direct vẫn giữ encoder, decoder cross-attention và grounded copy. Full chỉ bổ sung depth/feature residual cùng một `source_bias` do bridge dự đoán. Vì các đầu ra depth/feature/focus khởi tạo bằng 0, hai graph cho logits và CE bằng nhau ở bước đầu trên tiny fixture. Đây là khởi tạo bảo thủ nhưng cũng tạo đường học ưu tiên cho nền direct đã mạnh. Phải đo nhánh bridge sau train thay vì suy ra rằng tham số bổ sung được sử dụng.

## Những yếu tố có thể giải thích chênh lệch nhỏ

1. **Prior mới là tĩnh trong toàn bộ summary.** `source_bias[B,S]` được tính một lần từ source, prompt và budget, rồi cộng vào mọi timestep của cross-attention và copy. Attention QK/copy query của decoder **vẫn phụ thuộc timestep**; chỉ riêng tín hiệu mới của bridge không biết phần abstract nào đang viết. Điều này có thể khiến cùng một vùng source được ưu tiên khi viết objective, methods, results và conclusion. Đây là giả thuyết cơ chế, không phải nguyên nhân đã xác nhận bằng prediction.
2. **Hai đường tiêu thụ prior khác nhau.** Semantic attention dùng bias sau ép dtype sang BF16; grounded copy dùng FP32 và trước đó gộp các source occurrence của cùng decoder token ID thành một vị trí. Trong cấu hình value-anchor, copy keys còn được dựng từ `H0`, không trực tiếp từ depth/feature residual. Vì vậy nhánh depth/feature ảnh hưởng copy chủ yếu gián tiếp qua decoder hidden, còn `source_bias` là tác động bridge trực tiếp lên copy. Nếu từ xuất hiện ở nhiều nơi, một prior tốt cho occurrence cụ thể có thể bị trung bình hóa trước khi tính xác suất copy. Gate cross-attention và copy ban đầu cũng nhỏ. Cần đo can thiệp vào logits/predictions, không chỉ nhìn loss auxiliary.
3. **Loss evidence dùng weak labels rất nhiễu.** Nó coi mọi source token không thuộc bigram/trigram trùng target là negative, kể cả nội dung đúng được diễn đạt lại, dấu câu và định dạng. Tất cả occurrence của cụm lặp được đánh dấu dương. Ví dụ `drug A reduced risk. drug A increased risk.` với target `drug A reduced risk`: `drug A` ở câu thứ hai vẫn được dương. Phiên bản đầu còn nối các từ qua dấu chấm: source `alpha beta. gamma delta`, target `beta gamma` đánh dấu `beta` và `gamma` là evidence dù chúng thuộc hai câu khác nhau. **Code hiện tại đã chặn n-gram qua dấu kết thúc câu và newline**; ambiguity của occurrence, paraphrase và negative class vẫn chưa giải quyết.
4. **Thang của loss từng bị lệ thuộc số hàng invalid.** Phiên bản đầu lấy tổng loss của valid rows nhưng chia bằng tổng target tokens của *mọi* hàng, làm λ thực tế co theo tỷ lệ token của valid rows. **Code hiện tại đã sửa**: loss điều kiện trên valid rows, còn trainer chuẩn hóa đúng theo tổng valid target tokens trong toàn accumulation window và các DDP ranks. Log có cả `salience_coverage` theo hàng và `salience_token_coverage` theo token; hai thống kê này chỉ cho biết nhãn có/không, chưa chứng minh nhãn đúng.
5. **Gradient phụ có thể áp đảo đúng điểm bắt đầu.** Trên tiny fixture, agent audit đo gradient CE ở `focus_output` khoảng 2.4e−6, auxiliary chưa nhân λ khoảng 3.85e−2; λ=0.002 vẫn khoảng 32 lần CE. `focus_query`/`focus_key` ban đầu nhận gradient 0 do `focus_output` zero-init. Các tỷ lệ này phụ thuộc fixture và không suy rộng định lượng sang PubMed, nhưng chứng minh việc giảm λ từ 0.2 xuống 0.002 chưa bảo đảm CE chi phối scorer lúc đầu.
6. **Dữ liệu nhìn thấy có trần.** Recipe PubMed dùng `max_source_length=4096`. Evidence nằm ngoài đoạn tokenizer giữ lại không thể được chọn bởi bất kỳ bridge nào. Cần đo tỷ lệ n-gram reference có trong *visible source* trước khi kết luận module salience yếu.

## Cơ sở paper: điều học được và giới hạn chuyển giao

- [SEASON, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.409.pdf) đưa salience vào **key** của cross-attention và giữ source representation làm **value**. Điều này ủng hộ việc bảo toàn `H0` value anchor khi sửa bridge. Tuy nhiên SEASON dùng gold salience embedding khi train generator và predicted embedding khi test; nó không chứng minh loss n-gram hiện tại của AFMR sẽ tăng ROUGE trên PubMed.
- [TopDownFormer, Findings EACL 2023](https://aclanthology.org/2023.findings-eacl.94.pdf) báo cáo trên PubMed top-down cross-attention 48.34/21.40/44.22 so với no top-down 46.97/20.23/42.88 ở cấu hình tương ứng. Đây là chứng cứ vùng-to-token context có thể hữu ích, **không** phải dự báo +1 điểm cho AFMR/PPLX/Qwen.
- [DYLE, ACL 2022](sources/s10_dyle.md) dùng trọng số nguồn thay đổi theo decoder step. Nó tạo cơ sở cho hướng query-conditioned evidence nhưng cũng là phản chứng về tính phổ quát: trên arXiv DYLE 46.41/17.95/41.54, thấp hơn **hệ thống LSH khác** 48.24/20.26/41.78 trong bảng của paper. Đây không phải ablation trong cùng mô hình, nên không kết luận dynamic weighting gây giảm điểm.
- [LONGEVAL, EACL 2023](sources/s11_longeval.md) ghi nhận 54% bigram của human PubMed summaries trong mẫu của họ xuất hiện nguyên dạng trong **toàn bộ source**. Suy ra overlap là một **proxy không đầy đủ** cho nội dung quan trọng; không suy ra 46% phần còn lại là sai hay hoàn toàn không có bằng chứng, và cũng không suy ra mức coverage trong 4096 token AFMR nhìn thấy.
- [The Power of Summary-Source Alignments, Findings ACL 2024](sources/s12_summary_source_alignments.md) chỉ ra các heuristic alignment trước đó dựa trên sentence-level lexical matching có nhiễu; paper xây alignment ở mức proposition. Bài này dùng **Multi-News nhiều tài liệu**, nên chỉ hỗ trợ hướng kiểm tra nhãn tinh hơn, không cho phép chuyển thẳng một mức tăng ROUGE sang PubMed.
- [PROM, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1148.pdf) cho ví dụ phrase-copy chỉ đổi rất ít điểm và có thể đổi R-1/R-2 cùng chiều nhưng R-L ngược chiều trên arXiv. Đây là phản chứng cho giả định “copy mạnh hơn thì cả ba ROUGE cùng tăng”.

## Quyết định thiết kế sau khi phản biện

**Không dùng điểm test để tiếp tục tăng độ mạnh bridge một cách mù.** Ưu tiên kiểm định đường nhân quả rẻ hơn một lần train mới: trên validation, với cùng checkpoint full, đo (i) CE khi zero `source_bias`, (ii) CE khi tắt depth/feature residual, (iii) attention/copy gate sau train, (iv) prior positive-minus-negative theo vị trí, (v) reference n-gram coverage trong visible source và lỗi nhãn qua 100 mẫu đọc tay. Nếu can thiệp làm CE gần như không đổi, bridge đang bị mô hình bỏ qua; nếu CE xấu đi nhưng ROUGE không đổi, cần xem decoding/metric/output thay vì chỉ tăng gradient.

Đã thêm `src/eviseq_new/scripts/probe_bridge.py` cho phần (i) và (ii). Lệnh ví dụ trên server, dùng **resolved config của chính checkpoint full** và validation, không chấm test:

```bash
cd src/eviseq_new
RUN_DIR="$PWD/runs/afmr/pubmed_pair_afmr_value_anchor_copy/pplx"
python3 scripts/probe_bridge.py \
  --config "$RUN_DIR/resolved_config.yaml" \
  --checkpoint "$RUN_DIR/last.pt" \
  --device cuda:0 --batch-size 4 --max-examples 64 \
  --output "$RUN_DIR/bridge_probe.json"
```

`delta_ce_vs_full > 0` nghĩa là tắt đường đó làm teacher-forced CE trên subset tăng; `delta_ce_vs_full < 0` nghĩa là tắt nó làm CE giảm. Giá trị gần 0 chỉ gợi ý đường yếu trên subset này. Đây là **can thiệp trong cùng checkpoint**, không phải kết quả train w/o bridge và không suy ra trực tiếp ROUGE. Residual counterfactual chỉ tắt depth/feature; output báo `contextual_value_still_active` nếu nhánh tùy chọn đó vẫn bật. Script từ chối backbone không có thư mục local và đặt Transformers/HF ở offline mode; test dùng tiny backbone, không tải model lớn.

Đối chứng matched, cùng seed/protocol/checkpoint rule:

- **A**: `direct_projection` + CE.
- **B**: full bridge giữ `H0` value anchor + CE.
- **C**: full bridge + objective evidence, sau khi sửa chuẩn hóa và audit nhãn; chọn λ chỉ trên validation.

`B−A` là đóng góp kiến trúc thuần. `C−B` là đóng góp training objective. `C−A` là hiệu ứng hệ thống. Nếu chỉ C vượt A, claim phải là **bridge được huấn luyện bằng tín hiệu evidence**, không thể nói bridge đơn thuần tạo gain. Nếu B≈A và C vẫn không tăng, dừng hướng loss n-gram hiện tại.

Sau kiểm định lần ba, `afmr_pubmed.yaml` và `run_pubmed_pair.sh` mặc định `salience_loss_weight=0`. Arm C phải bật rõ `AFMR_SALIENCE_WEIGHT=0.002` với output directory mới; không dùng C làm full mặc định rồi so với A CE-only. Tên run cũ có thể chứa checkpoint thuộc protocol khác, nên chỉ tin `resolved_config.yaml` của từng checkpoint.

Trong trường hợp bridge thật sự bị bỏ qua, hướng kiến trúc đáng thử tiếp theo là **prior vùng phụ thuộc decoder query, giữ `H0` ở value và giữ grounded-copy route**. Để nó có vai trò riêng, decoder query ở timestep `t` chọn vùng source trước khi tính token-level semantic attention, thay vì cộng cùng một scalar prior lên mọi timestep. Thiết kế này nối nguyên lý key/value separation của SEASON, top-down region-to-token của TopDownFormer và dynamic weighting của DYLE, nhưng cách ghép với value anchor và copy của AFMR phải được thực nghiệm kiểm chứng. Trước tiên thử prior này trên **semantic attention**, không thay copy logits đồng thời; nếu thay cả hai sẽ không biết đường nào gây đổi điểm. Cần ablate lại A/B với CE-only để chứng minh *kiến trúc* mới giúp ích. Đây là **đề xuất**, chưa được implement hoặc chứng minh có +0.30 ROUGE.

Một dạng tối giản để kiểm tra giả thuyết là tính region key `R_j` từ các cửa sổ hiện có và, với mỗi hidden decoder `h_t`, tính `a_{t,j}=softmax_j(q(h_t)·k(R_j)/sqrt(d))`. Phân bố này được overlap-add về token thành bias nhỏ `b_{t,s}`; semantic attention dùng `softmax(Q_tK_s^T/sqrt(d)+g·b_{t,s})`, còn value vẫn là `V(H0_s)`. Đặt output của nhánh mới bằng 0 lúc khởi tạo, gate có giới hạn, và **không** tắt đường attention gốc. Đây là bổ sung có thể falsify: nếu can thiệp tắt `b_{t,s}` ở checkpoint mới không đổi CE/predictions, nhánh vẫn bị bỏ qua. Nếu nó đổi CE nhưng không tăng validation ROUGE, cơ chế không giải đúng bottleneck. Gradient từ CE đi qua query, region keys, overlap-add và encoder mà không cần sinh thêm bản tóm tắt trong train. Chi phí attention vùng là `O(B·T·R)` bên cạnh token cross-attention `O(B·T·S)` với `R` là số vùng và `S` là số source token; bộ nhớ/throughput thật phải đo trên B200. Nó không có bảo chứng novelty chỉ nhờ công thức; contribution phải nằm ở cách dùng region evidence bên cạnh `H0`-anchored values và copy, rồi được ablation chứng minh.

Tiêu chí dừng: nếu validation không có xu hướng tăng đồng thời cả R-1/R-2/R-L, hoặc gain chỉ có một seed, không đưa claim mạnh vào paper. Khi validation đạt mục tiêu, khóa config rồi chấm test, paired bootstrap theo ID và lặp seed; dùng cùng PPLX/Qwen, prompt, preprocessing, effective batch, update count, clip, decoding và ROUGE-1.5.5. Report cả trường hợp không vượt thay vì điều chỉnh sau khi xem test.
