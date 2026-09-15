# SDACL: trich xuat phuong phap va ke hoach ap dung cho EviSeq

Cap nhat: 2026-09-08

**Đính chính sau khi đối chiếu PDF và code EviSeq:** SDACL gốc đưa gold summary qua
một lượt encoder riêng (trang 4, Eq. 6). Mean-pool hidden decoder và dùng AFMR ở mục
11 là biến thể đề xuất trước đây, không phải reproduction và chưa đủ để claim
novelty. Thiết kế được phân tích lại trong
[EVISEQ_EVIDENCE_CONTRASTIVE_DESIGN.md](EVISEQ_EVIDENCE_CONTRASTIVE_DESIGN.md).
NEFTune đã được bỏ khỏi code; baseline cho thiết kế này là CE + cosine.

Nguon chinh:

- Ying Huang, Zhixin Li, Zhenbin Chen, Canlong Zhang, Huifang Ma.
  "Sentence salience contrastive learning for abstractive text summarization."
  Neurocomputing 593 (2024), 127808.
- DOI: https://doi.org/10.1016/j.neucom.2024.127808
- PDF da doc: /Users/kieugiangbien/Downloads/Paper/VAI/Summarization/SDACL.pdf
- Phan method: trang 3-6; setup: trang 7; ket qua/ablation: trang 8-10.

Day la tai lieu tra cuu phuong phap, khong phai ket qua cua mot run EviSeq moi. Cac
con so trong report la so paper bao cao voi PEGASUS/LLaMA va khong la cam ket EviSeq
se tang ROUGE tuong tu.

## 1. Y tuong cot loi

MLE/NLL chi toi uu du doan tung token cua target. No khong noi ro cau nao trong
document dong gop vao summary va cau nao la nhieu. SDACL dua contrastive learning
xuong muc cau source:

1. SSCL (Semantic Similarity Contrastive Learning) do do tuong dong giua moi cau
   source va target summary, sau do chon cau salient lam positive va cau it lien quan
   lam negative.
2. SDACL them salience distance-aware weighting de cap kho phan biet nhan penalty lon
   hon.

Objective cua paper:

    L = L_NLL + alpha * L_SDACL

SDACL tao pair tu cac cau co san trong source va target gold. Paper khong can sinh
candidate summary de tao positive/negative.

## 2. Seq2seq va representation

Voi document D = {s_1, ..., s_m} co m cau va summary Y co N token, model gom encoder
f va decoder g:

    H_enc = f(D_hat)
    h_dec_t = g(y_<t, H_enc)
    p(y_t | y_<t, D_hat) = softmax(W_dec h_dec_t + b)

Paper chen [CLS] vao dau va [SEP] vao cuoi moi cau source. Hidden cua [CLS] o layer
cuoi la sentence vector z_i. Target summary cung duoc chen [CLS]/[SEP] de tao z_Y.

Theo trang 4, Eq. (6), target gold được đưa **riêng vào cùng encoder f**:

    z_Y = f(Y_hat)

SDACL gốc cần xử lý thêm chuỗi target ở encoder. Không sinh candidate, nhưng không
thể gọi là chỉ thêm pooling vào lượt source-encoder đang có. Không nối target vào
source memory mà decoder đọc; việc đó sẽ làm lộ đáp án.

NLL baseline:

    L_NLL = -1/N * sum_t log p(y_t | y_<t, D_hat)

Cosine similarity chi la tin hieu semantic de xep hang salience; paper van dung
ROUGE de danh gia summary sinh ra. Paper khong toi uu truc tiep ROUGE.

## 3. Semantic similarity

Neu target la mot cau:

    sim(i, Y) = (z_i . z_Y) / (||z_i|| * ||z_Y||)

Neu target co n cau, paper tao vector z^Y_j cho tung cau summary:

    score_i_j = cosine(z_i, z^Y_j)
    sim(i, Y) = max_j score_i_j

Cau source duoc coi la quan trong neu gan it nhat mot cau target trong khong gian
semantic. Sau khi tinh, moi document co m diem sim(i,Y).

## 4. Tao positive va negative

Paper xep hang cau source trong cung document:

    S_p = arg TopK_i sim(i,Y)
    S_n = k_n cau co sim(i,Y) thap nhat

Trong do S_p la positive set, S_n la negative set. k_p va k_n thay doi theo so cau m;
viec chon co the dynamic trong training.

Ly do paper khong dung augmentation hoac random in-batch negative:

- chen/xoa/thay fragment co the pha cau truc tai lieu;
- in-batch negative voi pretrained model co the qua de phan biet;
- mot summary co dinh duoc chon lam positive co the lam giam tinh linh hoat cua
  model khi model co the tao mot summary tot khac.

## 5. SSCL

SSCL keo target gan positive va day xa negative. Voi tau la temperature:

    L_SSCL = -1/k_p * sum_i log(
        exp(sim(i,Y)/tau)
        /
        (sum_{j in S_n} exp(sim(j,Y)/tau) + exp(sim(i,Y)/tau))
    )

Loss tac dong chu yeu len encoder representation. Objective trung gian:

    L = L_NLL + alpha * L_SSCL

SSCL la contrastive learning co chon salience, nhung chua co distance-aware weight.

## 6. Salience distance-aware weighting

Paper cho rang moi document co phan bo salient khac nhau. Dac biet XSum co abstraction
cao, nhieu cau co similarity thap; cac cap positive/negative co the rat kho phan biet.

Khoang cach salience trung binh:

    dist_p-n =
        (1/k_p) * sum_{z_i in S_p} sim(i,Y)
      - (1/k_n) * sum_{z_j in S_n} sim(j,Y)
      + gamma

Paper dat gamma = 0.01.

Eq. (13) trong PDF in:

    w_i = - exp(1 - sim(i,Y)) / exp(dist_p-n)

Va Eq. (14) dua weight vao phan negative:

    L_SDACL = -1/k_p * sum_i log(
        exp(sim(i,Y)/tau)
        /
        (w_i * sum_{j in S_n} exp(sim(j,Y)/tau)
           + exp(sim(i,Y)/tau))
    )

Y nghia ma paper muon dat:

- positive co similarity thap duoc penalty lon hon;
- positive va negative co chenh lech nho duoc coi la cap kho;
- document co nhieu cau khong lien quan bi day ra xa.

### Canh bao ve dau tru trong Eq. (13)

Ban PDF that su in dau tru truoc phan so. Neu w_i < 0, mau so trong Eq. (14) co the
khong duong, khi do log khong hop le. Day la diem khong nhat quan toan hoc trong
ban paper/typeset. Khong copy cong thuc nay vao code ma khong kiem tra.

Gia thuyet engineering an toan can test la dung magnitude duong:

    w_i_positive = exp(1 - sim(i,Y)) / exp(dist_p-n)

Sau do kiem tra mau so, NaN/Inf va gradient tren fixture. Neu co code chinh thuc cua
tac gia, code do can duoc dung de xac minh lai quy uoc dau.

## 7. Cau hinh thuc nghiem paper

| Thanh phan | Cau hinh |
|---|---|
| Backbone | PEGASUS, encoder 16 layer, decoder 16 layer, 16 heads |
| Dropout | 0.1 |
| Weight decay | 1e-8 |
| Optimizer | Adafactor |
| Learning rate | 1e-4 |
| Epochs | 5 |
| Temperature tau | 1.0 |
| alpha | thu 0.5, 1.0, 1.5, 2.0, 3.0; chon 1.5 |
| GPUs | 4 x NVIDIA RTX 3090 |
| Batch CNNDM/XSum/PubMed | 8 / 16 / 4 |
| Source truncation | 1024 / 512 / 1024 tokens |
| Target truncation | 128 / 64 / 256 tokens |
| k_p, k_n | CNNDM m/6; XSum m/8; PubMed m/6 |
| ROUGE | rouge-score 0.1.2 |

Paper định nghĩa m là số câu document nhưng chưa mô tả đầy đủ quy tắc làm tròn,
xử lý câu cụt, hay m được tính trước/sau truncation. Batch 4 cho PubMed cũng không
nói rõ là per-GPU hay global. Khi triển khai cần ghi rõ các lựa chọn này; dùng số
câu còn nhìn thấy là đề xuất engineering, không phải chi tiết đã xác minh của paper.

## 8. Du lieu va ket qua paper

PubMed cua paper co 117,108 train, 6,631 validation, 6,658 test; trung binh 124 cau,
3,209 tu source va 9 cau, 208 tu summary. Tuy nhien paper cat source PubMed con 1,024
token trong thuc nghiem.

ROUGE-1 / ROUGE-2 / ROUGE-L:

| Dataset | PEGASUS | SDACL | Chenh lech |
|---|---|---|---|
| CNN/DailyMail | 44.17 / 21.47 / 41.11 | 46.24 / 22.52 / 43.57 | +2.07 / +1.05 / +2.46 |
| XSum | 47.21 / 24.56 / 39.25 | 47.83 / 24.90 / 39.81 | +0.62 / +0.34 / +0.56 |
| PubMed | 45.09 / 19.56 / 40.42 | 47.89 / 21.05 / 42.96 | +2.80 / +1.49 / +2.54 |

Tren PubMed, GRETEL trong cung bang cao hon SDACL:

    GRETEL: 48.20 / 21.20 / 43.16
    SDACL : 47.89 / 21.05 / 42.96

Vi vay ket luan dung la SDACL cai thien so voi PEGASUS trong protocol cua paper,
khong phai no la phuong phap tot nhat tuyet doi tren PubMed.

## 9. Ablation cua paper

Ablation tach tac dung cua contrastive learning va distance-aware weighting:

| Dataset | Model | CL | Distance-aware | R1 | R2 | RL |
|---|---|---:|---:|---:|---:|---:|
| CNNDM | PEGASUS implementation | No | No | 44.21 | 21.46 | 41.13 |
| CNNDM | SSCL | Yes | No | 45.13 | 22.10 | 42.09 |
| CNNDM | SDACL | Yes | Yes | 46.24 | 22.52 | 43.57 |
| XSum | PEGASUS implementation | No | No | 47.20 | 24.53 | 39.19 |
| XSum | SSCL | Yes | No | 47.64 | 24.74 | 39.56 |
| XSum | SDACL | Yes | Yes | 47.83 | 24.90 | 39.81 |

Paper cung bao cao tren LLaMA-7B (CNNDM):

    LLaMA baseline: 44.05 / 21.21 / 41.01
    LLaMA + SDACL: 45.43 / 21.94 / 42.68

Day la bang chung objective co the gan vao LLM backbone, nhung khong phai so sanh
decoder-only PubMed cong bang voi T5Gemma/EviSeq.

## 10. Phan tich va gioi han

Paper xem similarity tren 0.6 la salient. Trung binh khoang 10% cau CNNDM va 5% cau
XSum nam trong nhom salient; cau duoi 0.4 nhieu hon. Chon qua nhieu cau tao cap it
phan biet va co the overfit. Chon qua it cau co the bo sot thong tin.

Hai han che tac gia neu ro:

1. Lay ca cau lam salient span van co the gom thong tin khong can thiet.
2. Paper khong co objective rieng de dam bao factuality cua summary.

Do do khong duoc claim SDACL tu dong giam hallucination. FactCC chi la metric bao cao
them, khong phai factuality guarantee.

## 11. De xuat mapping vao EviSeq

Phan nay la de xuat cho repo, khong phai phuong phap nguyen ban cua paper.

### 11.1. Representation

EviSeq co AFMR bridge memory va decoder teacher forcing. Co the tao:

    z_i = masked_mean(AFMR_bridge_memory[token positions in sentence i])
    z_Y = masked_mean(decoder_hidden on valid gold target positions)

Sau do dung projection head ve cung dimension va L2-normalize truoc khi tinh cosine.
Lay source vector tu bridge memory de loss tac dong vao representation ma decoder dang
doc. Loai prompt, padding va token bi cat khoi source.

### 11.2. Sentence IDs

Collator can them source_sentence_ids hoac span list:

- tach cau truoc tokenization;
- dung offset mapping de gan token vao sentence id;
- chi giu sentence co token sau truncation;
- content mask van loai prompt/padding.

Khong bat buoc chen [CLS]/[SEP] vao tokenizer EviSeq neu span pooling da tao duoc
sentence representation. Chen special token se thay doi input format va embedding.

### 11.3. Xep hang va gradient

Co the xep hang dynamic bang cosine model:

- stop-gradient chi cho score dung de tao top/bottom index;
- khong detach source/target vector khi tinh loss, neu muon contrastive loss cap nhat
  encoder va decoder.

Neu ranking dao dong qua manh, dung gold overlap source-summary lam nhan salience co dinh
de lam ablation. Cach overlap on dinh hon nhung khong con hoan toan giong semantic
ranking cua SDACL.

### 11.4. Trong so

Khong copy alpha=1.5 cua paper vi thang do loss va backbone khac nhau. Bat dau:

    L_total = L_CE + lambda * L_SDACL
    lambda in {0.02, 0.05, 0.10}
    tau = 1.0

Khuyen nghi within-document negatives truoc; in-batch negatives co nguy co false
negative trong PubMed vi cac bai cung chu de chia se tu vung.

Log them:

- L_CE va L_SDACL;
- mean/max sim cua positive va negative;
- dist_p-n va w_i;
- mau so contrastive truoc log;
- gradient norm theo nhom encoder, bridge, decoder.

Mot forward teacher-forced da co hidden cho CE, nen source pooling, target pooling va
cosine khong can candidate generation hay decoder pass thu hai. Đây là biến thể đề xuất, không phải chi phí của SDACL nguyên bản.
Temperature `tau` trong contrastive loss khác với temperature/top-p của sampling:
SDACL có `tau=1.0` khi train, dù không sinh candidate.

## 12. Protocol thu nghiem

| Run | Objective | Muc dich |
|---|---|---|
| A | CE only | baseline EviSeq cung recipe |
| B | CE + SSCL | do tac dung contrastive khong weighting |
| C | CE + SDACL | do tac dung day du |

Tat ca run phai giu cung backbone PPLX, dataset/preprocessing, tokenizer, source limit,
global batch, accumulation, so optimizer updates, scheduler, clipping va seed. Chon
checkpoint bang validation; test chi dung sau khi chot recipe.

Dung cung Perl ROUGE-1.5.5, detokenization va space normalization cho EviSeq, T5Gemma
va decoder-only baselines. Current evaluator co config num_beams=1; can xac minh decode
thuc te truoc khi so sanh. Neu thay beam/length penalty, ap dung cung policy cho moi model.

## 13. Ket luan

Đề xuất sơ bộ trước đây (được đánh giá lại trong report thiết kế mới):

    CE + sentence-level salience contrastive tren AFMR bridge memory

Day la huong co bang chung PubMed truc tiep hon contrastive document-summary tong quat,
khong can candidate generation va bo sung cho diem yeu chon noi dung cua AFMR.

Can xu ly Eq. (13) nhu mot diem can xac minh truoc khi implement vi dau tru trong ban
PDF khong phu hop voi mau so Eq. (14). Bat dau voi lambda nho, within-document
negatives va mot forward teacher-forced duy nhat. Neu validation ROUGE khong tang, bo
objective nay thay vi chong them nhieu loss. Neu ROUGE tang, van can factuality
evaluation rieng; paper khong chung minh SDACL giai quyet hallucination.

## Tai lieu tham khao

1. Huang et al., "Sentence salience contrastive learning for abstractive text
   summarization", Neurocomputing 593 (2024) 127808.
   https://doi.org/10.1016/j.neucom.2024.127808
2. Ban PDF da doc:
   /Users/kieugiangbien/Downloads/Paper/VAI/Summarization/SDACL.pdf

## Các điểm tái lập cần giữ khi dùng report

- Dấu âm Eq. (13) được kiểm tra trên hình PDF; đổi sang dương là **giả thuyết sửa**,
  chưa được code tác giả xác nhận. Paper không nói rõ stop-gradient cho trọng số.
- Không gọi ROUGE trong paper là Perl155: tác giả ghi `rouge-score==0.1.2`.
  `rouge-score`, Python `rouge==1.0.0`, và Perl ROUGE-1.5.5 là các backend khác nhau.
- Table 4: FactCC PubMed 23.36 → 24.13, tức +0.77 điểm, khoảng +3.30% tương đối.
  Câu văn trang 8 nói +46.10% không khớp với bảng; không tái sử dụng con số đó.
- Table 5: CNNDM R2 của PEGASUS implementation 21.46 → SDACL 22.52 = +1.06;
  phần mũ tăng +0.96 in trong bảng không khớp. Report dùng giá trị score ở các cột.
- ROUGE/BERTScore cao hơn không tự chứng minh fluency hay factuality.
- Mean-pool toàn bộ gold summary có thể làm mất cấu trúc nhiều câu mà Eq. (8) của
  SDACL giữ bằng max trên từng câu summary; đây là thay đổi phương pháp đáng kể.
