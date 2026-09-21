# Shared Evidence Route: thử nghiệm bridge cho AFMR

Đây là bản sao độc lập của `eviseq_new` để thử **một giả thuyết kiến trúc**. Bản gốc không bị sửa. Encoder, value-anchored AFMR, Qwen decoder, grounded-copy mixture, prompt và CE loss giữ nguyên; chỉ thêm một router theo vùng nguồn ở các lớp cross-attention cuối và nối điểm router ấy với grounded copy.

## Giả thuyết

AFMR hiện tại tạo memory và source prior một lần trước decoder. Cross-attention sau đó học cách lấy thông tin ngữ nghĩa; grounded copy học một scorer khác để chọn vị trí từ nguồn. Hai đường có thể chọn **khác vùng sự kiện** trong cùng một bước sinh, nhất là khi tên hoặc con số xuất hiện nhiều lần. Một điểm bằng chứng chung phụ thuộc decoder prefix có thể giảm việc diễn đạt ý từ vùng A nhưng copy từ vùng B. Nếu lỗi này phổ biến, full router có thể cải thiện ROUGE và source support so với `AFMR + grounded copy` cùng ngân sách train. Đây chưa phải kết quả thực nghiệm.

Cơ sở và ranh giới prior art được đọc như sau:

- [Pang et al., Findings EACL 2023](https://aclanthology.org/2023.findings-eacl.94/) cập nhật token từ ngữ cảnh mức thô theo hướng top-down. **Source fact:** biểu diễn phân cấp giúp truyền ngữ cảnh dài về token. **Inference của bản này:** vùng ngắn có thể làm đơn vị thô cho router decoder; paper không chứng minh router này tăng ROUGE.
- [Qiu & Cohen, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.355/) dùng cấu trúc tài liệu tiềm ẩn và graph-level attention để tập trung decoder vào thông tin quan trọng. Bản thử này không học graph và vẫn giữ toàn bộ token nguồn.
- [See et al., ACL 2017](https://aclanthology.org/P17-1099/) có pointer-generator và coverage; [Li et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.336/) dùng lịch sử copy. Copy, attention chung và lịch sử copy đều là prior art; bản thử này chưa có state lịch sử.
- [Boutkan et al., 2019](https://arxiv.org/abs/1905.01975) khảo sát trade-off của pointer/generator attention. Vì vậy router chỉ ràng buộc **vùng bằng chứng**, không ép hai đường dùng cùng attention token.

## Cơ chế đã triển khai

`pool_source_regions` chia **chỉ token nội dung** thành vùng liên tiếp 32 encoder tokens mặc định, gần mức câu/cụm bằng chứng hơn 64 tokens nhưng không cần sentence parser. Prompt, padding và token nguồn không hợp lệ nhận ID `-1`. `R_k` là trung bình vector memory đã qua bridge; full token memory vẫn được giữ. Vùng nguồn không phải fact có nhãn, và một vùng vẫn có thể chứa nhiều facts.

Hai lớp cross-attention cuối thêm thành phần key theo vùng:

```text
K'_(layer,i) = K_(layer,i) + lambda_layer * U_layer(R_region(i))
attention = SDPA(Q_layer, K', V, source_bias, padding_mask)
```

`V` không đổi. Lớp cuối dùng chính query/key vùng để tính điểm vùng trước khi sinh token, trung bình trên heads, rồi truyền nó sang grounded copy:

```text
route_(t,k) = mean_heads(Q_(t,head) · U_last(R_k) / sqrt(head_dim))
copy_score'_(t,j) = copy_score_(t,j) + beta * route_(t,region(j))
```

`j` là **vị trí decoder-token trong nguồn**, không phải vocabulary ID. Ánh xạ hai tokenizer lấy từ offset overlap hiện có; nếu một decoder-token giao nhiều vùng, vùng có overlap lớn nhất được dùng. Copy vẫn softmax trên vị trí rồi cộng xác suất của các vị trí cùng token ID. `lambda` và `beta` học được, khởi tạo 0.10, giới hạn 0.50. Không tạo bias động `[batch,target,source]`: key vùng được cộng vào K trước SDPA.

`U_layer` được khởi tạo bằng chính `K` projection đã sao chép từ self-attention vào cross-attention. Do đó key vùng bắt đầu trong không gian truy xuất đã có của decoder; router không khởi đầu bằng một phép chiếu ngẫu nhiên mới. Mỗi lớp vẫn được phép cập nhật projection vùng của riêng nó trong training.

Trong train, router chỉ nhìn decoder prefix theo teacher forcing và nhận gradient từ **cùng CE mixture**. Không có oracle region label, candidate generation, contrastive loss, auxiliary loss hay decoder pass thứ hai. Khi `evidence_router.enabled=false`, code đi qua đường AFMR/copy cũ. `return_logits=true` và chunked CE dùng cùng route scores; inference cache và finished-row compaction mang theo region states/IDs.

## Chạy thử

Script mặc định chỉ chạy PPLX trên một GPU, dùng model local và data đã chuẩn bị trong `eviseq_new`:

```bash
cd /workspace/storage-shared/nlp/dungdx4/bien_projects/LLM2Seq-main
CUDA_VISIBLE_DEVICES=0 AFMR_ENCODERS=pplx \
PPLX_ENCODER=/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b \
DECODER_MODEL=/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B \
PROCESSED_DATA_DIR="$PWD/src/eviseq_new/datasets/pubmed" \
AFMR_EVIDENCE_ROUTER=true \
AFMR_OUTPUT_DIR="$PWD/runs/evidence_route_full" \
ROUGE155_SCRIPT="$PWD/src/rouge155/evaluate_rouge.py" \
PYROUGE_HOME_DIR=/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5 \
bash src/afmr_evidence_route/scripts/run_pubmed_pair.sh
```

Đối chứng trong cùng folder dùng lại lệnh với `AFMR_EVIDENCE_ROUTER=false` và output dir khác. Đối chứng `direct projection + copy` thêm `AFMR_BRIDGE_MODE=direct_projection`. Các run cần cùng data fingerprint, seed, batch, epochs, prompt, precision và decoding; chọn tham số trên **validation**, chỉ dùng test cho báo cáo cuối. Nếu gap nhỏ, chạy nhiều seeds và paired bootstrap theo example ID.

## Giới hạn

Router hiện không duy trì state vùng qua nhiều token; decoder self-attention chỉ có thể học điều đó ngầm. Nó không bảo đảm tránh nhảy giữa hai occurrence, không bảo đảm vùng 64 tokens là fact đúng, và không bảo đảm tăng ROUGE-2 hay giảm hallucination. Trước full run, kiểm tra validation source-linked xem lỗi ghép sai entity/number/relation giữa các vùng có phổ biến không. Nếu không có tín hiệu hoặc full không thắng đối chứng, không nên giữ claim bridge trong paper.

Kiểm tra local không tải model lớn:

```bash
PYTHONPATH=src/afmr_evidence_route /Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q src/afmr_evidence_route/tests
```
