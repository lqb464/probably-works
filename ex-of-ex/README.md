# ex-of-ex — hidden-token scorer sweep

Thư mục này là một harness thí nghiệm **độc lập, evaluation-only** cho câu hỏi:

> Hidden information ở layer cuối có thể sửa các lỗi top-1 của global vector hay không, và lỗi nằm ở MaxSim hay ở chính biểu diễn hidden?

Nó không sửa training code, loss hoặc checkpoint. Mỗi ảnh/caption chỉ được encode một lần; cùng một tensor cosine token–patch được dùng cho tất cả scorer trong YAML. Vì vậy thêm nhiều pooling rule rẻ hơn nhiều so với chạy lại từng notebook từ đầu.

## Chạy nhanh một suite

Từ root của repository:

```bash
python ex-of-ex/run_experiments.py \
  --config ex-of-ex/configs/rstp_fast.yaml \
  --root-dir /kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark \
  --checkpoint rstp.pth \
  --model-preset clip \
  --run-name rstp-final-layer
```

Trong Kaggle notebook, thêm `!` ở đầu đúng lệnh trên. Suite `rstp_fast.yaml` chạy 9 scorer cùng lúc: MaxSim, top-m, SCAN-style soft pooling, Smooth-Chamfer, CFine cross-grained, TokenFlow-style stable flow, Sinkhorn OT và uniform-mean control. Mỗi scorer còn được thử local-only và fusion với global ở các trọng số đã khai báo.

Muốn quét nhiều hyperparameter hơn (và top-100):

```bash
python ex-of-ex/run_experiments.py \
  --config ex-of-ex/configs/rstp_full.yaml \
  --root-dir /kaggle/input/datasets/hoanggv/tbps-benchmark/benchmark \
  --checkpoint rstp.pth \
  --model-preset clip \
  --run-name rstp-full-sweep
```

Lần thứ hai sẽ dùng lại feature cache của lần đầu nếu checkpoint và cấu hình encoder giống hệt. `rstp_full` chậm hơn đáng kể vì có top-100 và hai cấu hình Sinkhorn.

Các dataset khác dùng `icfg_fast.yaml` và `cuhk_fast.yaml`. Có thể chạy nhiều config trong một process nếu mỗi YAML đã chứa checkpoint riêng:

```bash
python ex-of-ex/run_experiments.py \
  --config ex-of-ex/configs/rstp_fast.yaml ex-of-ex/configs/icfg_fast.yaml \
  --run-name cross-dataset
```

`--checkpoint` chỉ áp dụng khi chạy một config để tránh vô tình dùng một checkpoint cho sai dataset.

## Output được tách như thế nào

Mỗi lần chạy tạo một thư mục không ghi đè:

```text
ex-of-ex/outputs/<dataset>/<YYYYMMDD-HHMMSS>__<run-name>/
├── run.log
├── resolved_config.yaml
├── manifest.json
├── summary.csv
└── scorers/
    ├── maxsim/
    │   ├── summary.json
    │   └── query_results.csv
    ├── topm_m3/
    └── ...
```

- `run.log`: thời gian từng stage, cache hit/miss, metric tốt nhất của từng scorer và traceback nếu lỗi.
- `summary.csv`: một dòng cho mỗi `scorer × K × local/fusion`, gồm R@1/5/10, mAP, mINP, số query được cứu, bị phá và net rescue. Dòng đầu là global baseline.
- `query_results.csv`: quyết định theo từng caption để truy ngược case cụ thể; có PID dự đoán, global/local correctness, rescued/harmed và fixed-pair margins.
- `manifest.json`: SHA-256 checkpoint, git commit, phiên bản runtime, token policy, timing, config và citation.
- `resolved_config.yaml`: cấu hình thực tế sau khi áp dụng CLI override.

Feature cache nằm ở `ex-of-ex/cache/<dataset>/<preset>/<checkpoint-hash>/`. Không dùng cache với `--no-cache`; ép encode lại với `--force-recompute`. Nếu thiếu VRAM, giảm `--pair-batch-size 128`; việc này không đổi score.

## Scorer và ý nghĩa phản biện

| `kind` | Aggregation | Điều nó kiểm tra |
|---|---|---|
| `maxsim` | mean theo token của patch tốt nhất | Baseline hiện tại; rất nhạy với một patch match giả mạnh. |
| `topm` | mean của top-m patch rồi mean token | MaxSim có quá cực đoan không? |
| `softmax_t2i` | softmax attention trên patch | Cho nhiều patch đóng góp thay vì hard argmax. |
| `smooth_chamfer` | log-mean-exp hai chiều | Match token→patch và patch→token có nhất quán không? |
| `cfine_cross_grained` | global-image↔word và global-text↔patch | Global context có chọn đúng local evidence không? |
| `tokenflow_stable` | importance-aware bidirectional flow | Token/patch quan trọng có nên mang nhiều transport mass hơn không? |
| `sinkhorn_ot` | balanced optimal transport | Có cần matching phân bố thay vì mỗi token tự chọn patch? |
| `uniform_mean` | mean tất cả valid token–patch | Null/control: local signal có tốt hơn pooling không cấu trúc không? |

`smooth_chamfer` dùng normalized log-mean-exp để độ dài caption không tự cộng bias. `tokenflow_stable` là adaptation evaluation-only: cosine global–local được shift từ `[-1,1]` sang miền không âm trước khi tạo flow marginal. Vì vậy output ghi rõ tên `stable`, không tuyên bố là reproduction nguyên bản của TokenFlow.

Fusion được tính trong từng global top-K bằng z-score theo query:

```text
fusion = (1 - w) · z(global) + w · z(local)
```

Phần ngoài top-K giữ nguyên thứ tự global, nên R@K và mAP là metric của một ranking đầy đủ, không phải accuracy chỉ trên candidate set.

## Đọc kết quả

Ưu tiên nhìn đồng thời bốn cột trong `summary.csv`: `r1`, `map`, `rescued`, `harmed`. Chỉ nhìn `rescued` dễ dẫn tới chọn một scorer cứu nhiều failure nhưng phá còn nhiều global-correct hơn. `net_rescued = rescued - harmed` cho biết trade-off đó, còn `proper_oracle_r1` chỉ là upper bound nếu có oracle biết khi nào tin global/local — không phải metric triển khai thực tế.

Nếu mọi scorer đều có fixed-pair recovery nhưng local rerank/fusion không tăng R@1, hidden có tín hiệu nhưng thiếu cơ chế confidence/gating. Nếu top-m/softmax/OT thắng MaxSim ổn định, vấn đề có bằng chứng nằm ở aggregation. Nếu tất cả đều thất bại tương tự, giả thuyết “hidden cuối chứa đủ thông tin sửa lỗi” yếu hơn, thay vì chỉ đổ lỗi cho MaxSim.

## Bốn thí nghiệm validation-selected

Các file `*_validated.yaml` bật thêm khối `validated`. Một lệnh `run_experiments.py`
vẫn extract test features đúng một lần, sau đó extract/cache official validation split và
chạy bốn phép thử dưới `RUN_DIR/validated/`:

Các config này đặt `legacy_test_sweep: false`: grid scorer/K/fusion cũ không được
đánh giá bằng test labels. Grid chỉ chạy trên validation; test nhận cấu hình đã khóa.
Muốn tái tạo đúng thí nghiệm P–W1 cũ thì tiếp tục dùng các config `*_fast.yaml`.

1. `setwise_summary.csv`: oracle representation diagnostic. Mọi positive của query
   được so với top-1/5/10 hard negatives cố định từ global baseline. Báo setwise
   accuracy, pairwise AUC, MRR, recovery/harm, paired delta so với MaxSim và
   identity-bootstrap CI.
2. `probe_results.csv`: logistic probe có regularization, train trên validation
   identities và test trên test identities. So sánh `global_only`, `hidden_only`,
   `global_plus_hidden` và `permuted_hidden_control` với identity-balanced weights.
3. `reranking_validation_grid.csv` và `reranking_selected_test.csv`: chọn
   scorer/K/fusion trên validation, rồi khóa cấu hình để rerank global top-K ở test.
   Positive không được chèn vào candidate set.
4. `gating_summary.csv`: chọn uncertainty gate trên validation dưới các harm budget
   1%, 2% và 5%, rồi áp đúng threshold đã khóa lên test.

`coverage.csv` là ceiling quan trọng: nếu positive không nằm trong global top-K thì
mọi top-K reranker đều không thể cứu query đó. `summary.json` ghi protocol và đường
dẫn của toàn bộ output. Log của bốn phép thử dùng chung `run.log` của run để truy vết.

Các config `*_validated.yaml` mặc định dùng 300 identity-bootstrap repetitions để giữ
thời gian vừa phải. Khi làm bảng cuối cho paper, tăng
`validated.bootstrap_repetitions` lên 2000 và giữ nguyên mọi hyperparameter khác.

Các cell Kaggle hoàn chỉnh nằm trong `KAGGLE_CELLS.md`.

## Nguồn phương pháp

- FILIP late interaction / token-wise MaxSim: [Yao et al., ICLR 2022](https://openreview.net/forum?id=cpDhcsEDC2).
- SCAN soft cross-attention: [Lee et al., ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.html).
- CFine cross-grained alignment: [Yan et al., IEEE TIP 2023](https://doi.org/10.1109/TIP.2023.3327924).
- Smooth-Chamfer family: [Kim et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_Improving_Cross-Modal_Retrieval_With_Set_of_Diverse_Embeddings_CVPR_2023_paper.html).
- TokenFlow: [Zou et al., arXiv:2209.13822](https://arxiv.org/abs/2209.13822).
- Entropic optimal transport / Sinkhorn: [Cuturi, NeurIPS 2013](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html).

## Kiểm tra code không cần dataset

```bash
python ex-of-ex/tests.py
```
