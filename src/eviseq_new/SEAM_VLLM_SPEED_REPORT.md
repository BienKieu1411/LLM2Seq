# SEAM vLLM: tối ưu tốc độ mà giữ nguyên cấu hình sinh

Ngày kiểm tra: 2026-10-16  
Phạm vi: `src/eviseq_new/seam_vllm` và bộ đánh giá ROUGE tiếng Việt qua HTTP.

## Kết luận

Đã áp dụng một số thay đổi ít rủi ro để giảm cấp phát tensor tạm, đồng bộ GPU-CPU lặp lại và thời gian giữ đồng thời các tensor trung gian. Chúng không đổi trọng số, precision, prompt, giới hạn token hay thuật toán chọn token. Tuy nhiên, môi trường Mac không có CUDA/vLLM và không truy cập được GPU server, nên chưa có benchmark để khẳng định thời gian thực tế đã giảm hoặc đầu ra GPU giữ nguyên tuyệt đối.

## Thay đổi trong mã

- Copy-logit mixer dùng một buffer FP32 liên tục cho các phép tính trung gian thay vì tạo nhiều tensor lớn theo kích thước vocabulary. Thứ tự phép toán được giữ theo implementation tham chiếu.
- Không chuyển logits sang contiguous trước `logsumexp`: thử nghiệm cho thấy thao tác đó có thể đổi kết quả giảm khoảng `1.91e-6` trên một trường hợp FP32. Giữ layout gốc ưu tiên tương đương số học.
- Encoder đọc độ dài GPU thành danh sách CPU một lần cho cả batch, thay vì đồng bộ lặp lại ở mỗi microbatch.
- Giải phóng trạng thái encoder và prompt trung gian ngay sau bridge để giảm thời gian các tensor lớn cùng chiếm VRAM.
- Bộ đánh giá ghi thời gian tổng, examples/s và output tokens/s để đo cùng một cách giữa các lần chạy.
- Tùy chọn `encoder_max_batch_tokens` giới hạn tổng số source positions đã padding trong mỗi microbatch encoder/bridge. Mặc định `8192`; chỉ chia microbatch, không truncate dữ liệu. Đây chủ yếu là chốt kiểm soát VRAM, không nên coi là tối ưu tốc độ đã được chứng minh.

## Tính đúng và kiểm thử

- So sánh CPU với mixer tham chiếu trên các trường hợp FP32/BF16, tensor contiguous/transposed, vocabulary có padding, FP16, BF16, FP32, int64, NaN/Inf, toàn mask và vùng copy rỗng: **36 trường hợp chính xác** theo các kiểm tra đã chạy.
- Kiểm thử plugin và evaluator: **206 passed, 3 skipped**. Các bài GPU/vLLM bị skip trên Mac.
- Ruff, kiểm tra formatter và `git diff --check` đều qua.
- Một kiểm tra differential riêng trên CPU cho mixer cho `max_abs_diff=0`; kết quả này không thay thế việc kiểm tra logits trên GPU.

## Những gì chưa nên bật để đuổi tốc độ

- Không bật CUDA graphs chỉ bằng cờ cấu hình: adapter hiện lưu trạng thái copy theo request qua Python state; graph replay không chạy lại phần staging đó. Cần thiết kế lại staging và kiểm thử request đồng thời trước.
- Không bật quantization, giảm precision, speculative decoding hoặc cắt source/output token budget trong lần so sánh tốc độ giữ chất lượng.
- Không kỳ vọng tối ưu cross-attention theo từng token: vLLM giữ cross-attention K/V trong cache; decoder không cần dựng lại encoder representation ở mỗi bước.
- `max_num_seqs` cao hơn có thể tăng throughput nhưng cũng tăng KV-cache/VRAM. Log OOM đã thấy gần cạn VRAM, vì vậy cần thử theo nấc và theo dõi peak memory.
- Microbatch theo độ dài có thể giảm padding nhưng đổi shape/tổ chức batch, có thể chọn kernel khác và làm thay đổi số học GPU. Chỉ nên thử như một thí nghiệm riêng, rồi so sánh đầu ra và ROUGE.

## Quy trình benchmark trên server

Giữ nguyên checkpoint, config, prompt, `max_input_tokens`, `max_tokens`, temperature/top-p/top-k, dtype và dataset. Mỗi lượt dùng tên output mới để tránh evaluator tái sử dụng predictions cũ. Ghi examples/s, output tokens/s, thời gian tổng, peak VRAM, số prediction khác nhau và ROUGE-1/2/L.

Thử concurrency theo thứ tự `64 -> 96 -> 128`, dừng khi VRAM tiến sát ngưỡng OOM hoặc tốc độ không còn tăng. Giữ `encoder_max_batch_tokens=8192` trước; chỉ thử `16384` nếu microbatch hiện tại đang gây nhiều overhead và VRAM còn đủ. Không đổi concurrency và microbatch cùng lúc.

Ví dụ lệnh đánh giá qua host tạm:

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/evaluation/evaluate_rouge_vietnamese_vllm.py \
  --start-server \
  --afmr-config /path/to/resolved_config.yaml \
  --afmr-checkpoint /path/to/best.pt \
  --model-name seam_speed_64 \
  --batch-size 64 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --encoder-max-batch-tokens 8192
```

Kiểm tra các tên cờ với `python3 src/evaluation/evaluate_rouge_vietnamese_vllm.py --help` trên checkout đã cập nhật trước khi chạy trên server.

## Tham khảo kỹ thuật

- PyTorch [`torch.sub`](https://docs.pytorch.org/docs/2.10/generated/torch.sub.html), [`torch.where`](https://docs.pytorch.org/docs/2.10/generated/torch.where.html), và [`empty_like`](https://docs.pytorch.org/docs/2.10/generated/torch.empty_like.html) hỗ trợ ghi kết quả vào buffer có sẵn.
- PyTorch ghi rõ [`Tensor.tolist()` chuyển tensor sang CPU nếu cần](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.tolist.html), nên gộp lần gọi này tránh đồng bộ lặp lại.
- Trong vLLM 0.25.1, trạng thái encoder-decoder dùng cross-attention K/V cache ở decoder: [upstream model state](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu/model_states/encoder_decoder.py).
- Tài liệu vLLM 0.25.1 về [giới hạn bộ nhớ qua `max_num_seqs`/`max_model_len`](https://docs.vllm.ai/en/v0.25.1/configuration/conserving_memory/) và [CUDA graph modes](https://docs.vllm.ai/en/v0.25.1/design/cuda_graphs/) hữu ích khi thử trên server.

## Giới hạn kết luận

Đây là các thay đổi hướng tới giảm overhead và áp lực bộ nhớ, chưa phải kết quả benchmark end-to-end. Chỉ có thể kết luận nhanh hơn sau khi đo trên GPU server bằng cùng dữ liệu/cấu hình; chỉ có thể kết luận giữ nguyên chất lượng sau khi kiểm tra prediction và ROUGE trên chính checkpoint đó.
