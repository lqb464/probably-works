# Frozen CLIP layer probing — text và image

Hai notebook: `text.ipynb`, `image.ipynb`. Cell copy/paste: `KAGGLE_CELLS.md`.
Chạy trong repo gốc; chỉ import kiến trúc CLIP, tokenizer và dataset readers hiện có.
Không phụ thuộc code Q2 đang sửa trong `diagnostic/` hoặc `ex-of-ex/`.

## Câu hỏi và giới hạn kết luận

Experiment đo **thông tin phục vụ text-to-image person retrieval có thể đọc ra
bằng một linear adapter**, khi giữ encoder đối diện ở final global embedding.
Không đo tổng lượng thông tin, không phải classification/attribute/localization
probe và chưa chứng minh adapter dùng được với backbone khác. Hai encoder có
thể có layer tối ưu khác nhau. Checkpoint TBPS đã fine-tune không đồng nghĩa
với CLIP pretrained gốc; kết luận phải ghi đúng checkpoint.

- Quét output sau từng residual block, đánh số 1..L.
- `global`: CLS ảnh hoặc EOS text; `hidden`: mean patch không CLS hoặc mean
  content token giữa SOT và EOS. Token có ID 0 nằm trước EOS vẫn được giữ.
- `direct_*`: áp final LayerNorm/projection lên mỗi layer. Đây là phép đo
  alignment với head có sẵn, không dùng riêng nó để kết luận layer ít thông tin.
- `ridge_*`: non-learned LayerNorm trên từng token, pooling rồi L2-normalize;
  học ridge projection đến centroid global của modality đối diện cùng identity.
  Mỗi identity có tổng trọng số bằng nhau; penalty chọn trên selection.
  Đây là regression retrieval probe, không phải identity classifier.
- `average_*`: trung bình feature đã chuẩn hóa của các layer + cùng ridge head.
- `mix_*`: softmax scalar theo layer + một linear head chung, fit bằng weighted
  MSE có regularization. Chạy 3 initialization seeds; seed được chọn trên
  selection nên kết quả test không phải mean±std của 3 seeds.
- `fusion_hidden_*`: trộn baseline global với best hidden ridge embedding;
  beta thuộc 0, .25, .5, .75, 1 chọn trên selection, bao gồm no-op.
- `permuted_hidden_control`: null control cố định layer/penalty từ probe thật,
  nhưng hoán vị các hàng hidden ở FIT trước khi fit adapter. Nó không được phép
  thắng trong bước chọn model; xem `permutation_control.json` và dòng cùng tên
  trong `test_results.json`. Nếu probe thật không vượt rõ control này thì chưa
  có bằng chứng hidden feature mang tín hiệu retrieval/identity.

Single-layer, average và mix có cùng kích thước linear head trong mỗi encoder;
mix thêm L scalar. Text và image có hidden width khác nhau nên không so capacity
chéo encoder như thể bằng nhau. Mean pooling có thể mất thông tin cục bộ;
kết quả âm tính không phủ nhận thông tin còn nằm trong token sequence.

## Protocol tránh chọn layer trên test

RSTP/CUHK: chia official validation theo identity 50% fit / 50% selection;
test chính thức giữ nguyên. Fit adapter chỉ dùng fit; chọn layer, penalty,
mix seed và fusion beta bằng R@1 rồi mAP trên selection. Lưu `selection.json`
trước khi extract test. Không refit sau lựa chọn. Checkpoint phải được train
không dùng test; nếu checkpoint từng được chọn bằng test, code này không sửa
được leakage từ trước. Validation từng dùng chọn backbone cũng cần công bố.

ICFG: không có official validation. Bắt buộc `--allow-test-holdout`; chia test
theo identity 40% fit / 20% selection / 40% evaluation, gallery cũng theo nhóm.
Đây là protocol holdout riêng, **không so số tuyệt đối với official ICFG**.
Lưu IDs và hash annotations để tái lập. Không âm thầm lấy test làm validation.

Test chỉ đánh giá baseline và lựa chọn khóa trước của từng family, kể cả final
ridge baseline. Không xuất test curve tất cả layer để tránh tiếp tục chọn theo
test. Muốn đọc layer curve dùng `validation_grid.csv`.

## Output và cách đọc

- `manifest.json`: checkpoint SHA256, commit, args, phiên bản, split identities.
- `fit_features.pt`, `select_features.pt`, `test_features.pt`: pooled FP16 CPU
  features của từng layer; không lưu mọi patch/token (tránh tràn RAM/disk).
- `validation_grid.csv`, `selection.json`: grid và lựa chọn đã khóa.
- `adapters.pt`: fitted projection weights; alpha nằm trong selection specs.
- `test_results.json`: R@1/5/10, mAP, mINP trong [0,1], paired delta CI 95%
  so với global baseline (2.000 identity bootstrap).
- `queries_*.npy`: mỗi caption một hàng [R1,R5,R10,AP,INP], thứ tự annotation
  của split test. CI không hiệu chỉnh multiple comparisons; ưu tiên lựa chọn
  `overall` đã khóa, các family là phân tích phụ.

Retrieval luôn text→image và full gallery của split, tính theo query chunk 64;
không có oracle/candidate injection. Khi quét image, gallery feature thay đổi,
text query vẫn frozen. Khi quét text, image gallery vẫn frozen.

Để nói intermediate tốt hơn final, so `ridge_hidden` với `ridge_final_hidden`,
không chỉ với global baseline. CI xuất sẵn là vs global; muốn khẳng định chênh
lệch intermediate–final hãy bootstrap hiệu giữa hai file queries tương ứng.
Nếu fusion thắng baseline mới có bằng chứng hữu ích cho việc bổ sung global.
Đây vẫn là bằng chứng trong dataset/task/checkpoint và loại probe đã thử.

Permutation control chỉ kiểm tra **decodability** với adapter tuyến tính; nó
không chứng minh tính nhân quả của hidden state. Muốn nói causal cần thêm
activation ablation/patching hoặc intervention và một evaluation riêng.

## Kaggle T4 16GB × 2

Mỗi process dùng một T4; launcher chạy tối đa hai dataset/checkpoint độc lập
cùng lúc và truyền `CUDA_VISIBLE_DEVICES`. Hai GPU không cộng thành 32GB cho
một model. AMP FP16 cho encode, ridge CPU FP64, mix FP32 trên GPU.
Cell Kaggle dùng batch 64 làm điểm bắt đầu; nếu OOM giảm còn 32 hoặc 16. Chỉ chạy
một dataset thì một GPU được dùng. Notebook text và image nên là hai Kaggle
sessions riêng.

Checkpoint load strict, không download pretrained hoặc fallback weights.
Hỗ trợ ViT CLIP đúng architecture repo; image size/stride phải khớp checkpoint.
Không tự resize positional embedding vì có thể thay đổi experiment.
Output directory phải mới để tránh ghi đè; features được lưu để phân tích,
chưa có tự động resume khi process bị ngắt.

## Kiểm tra

`python layer_probe/tests.py`: numerical tests và tiny real CLIP CPU,
checkpoint roundtrip, hook fidelity cho cả hai modality, chạy full pipeline
bằng dữ liệu synthetic. Đây không thay thế chạy benchmark thật/T4.
