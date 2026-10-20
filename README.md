# 🏸 BWF Match Predictor

Dự đoán xác suất thắng trận đơn MS/WS bằng Elo trước trận và machine learning,
với dashboard Streamlit để xem kết quả, giải thích đặc trưng và giả lập đối đầu.

**Dashboard hiện dùng 15.982 trận thi đấu thực tế** từ 10/01/2018 đến 05/07/2026:
8.429 MS và 7.553 WS. Nguồn là [kho cộng đồng tổng hợp kết quả BWF](https://github.com/SahilMotyar/bwf-match-data),
không phải bản phát hành chính thức của BWF. Xem [nguồn, bộ lọc và giới hạn](docs/real_data.md).
Dữ liệu giả lập cũ vẫn được giữ riêng tại `data/raw/matches.csv`.

## Cài đặt và chạy

Python 3.10 trở lên. Trên Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/fetch_real_data.py --offline
python scripts/run_experiment.py --data data/raw/bwf_real_matches.csv
python -m streamlit run app.py
```

Trên Linux/macOS, kích hoạt môi trường bằng `source .venv/bin/activate`.
Dependencies được khai báo trong `pyproject.toml`; `requirements.txt` dùng chung cấu hình này.
Chỉ cài ứng dụng, không cần công cụ phát triển: `python -m pip install .`.
Các khoảng phiên bản hỗ trợ resolver chọn bản phù hợp với Python; chưa có lockfile cố định môi trường.

## Nhập dữ liệu của bạn

```powershell
python scripts/prepare_kaggle_data.py --input path/to/kaggle.csv --output data/raw/real_matches.csv
python scripts/run_experiment.py --data data/raw/real_matches.csv
```

Hỗ trợ schema `player_a/player_b/a_won`, `winner/loser`, hoặc
`team_one_players/team_two_players` với nhãn người thắng được nhận diện rõ ràng.
Bộ chuyển đổi loại trận bỏ cuộc, lọc MS/WS, chuẩn hóa tên và kiểm tra dữ liệu.
Xem [schema và giới hạn dữ liệu](docs/data_schema.md).

Tải lại bản nguồn thực tế đã cố định commit và kiểm tra SHA-256:

```powershell
python scripts/fetch_real_data.py
```

Thêm `--offline` để chuẩn hóa lại từ bản nguồn đã lưu trong `data/external/bwf-match-data/`, không cần mạng.

Để tạo dữ liệu mẫu riêng hoặc chạy thử mà không ghi đè báo cáo hiện tại:

```powershell
python scripts/prepare_kaggle_data.py --generate-sample --output data/raw/sample.csv
python scripts/run_experiment.py --synthetic --output-dir .tmp/demo
$env:BWF_OUTPUT_DIR = "$PWD/.tmp/demo"
python -m streamlit run app.py
```

Xóa biến `BWF_OUTPUT_DIR` để dashboard đọc lại báo cáo mặc định ở thư mục dự án.
Cờ `--dynamic-k` buộc chọn K có điều chỉnh theo cấp giải; mặc định so sánh cả K động và K cố định.

## Thiết kế đánh giá

- Tính Elo, Elo theo cấp giải, phong độ 10 trận, đối đầu, kinh nghiệm, ngày nghỉ
  và chênh lệch thứ hạng chỉ từ **timestamp nhỏ hơn**. Nhóm trận cùng timestamp
  được dự đoán trước khi cập nhật bất kỳ kết quả nào của nhóm.
- Tách trạng thái vận động viên theo MS/WS. Hoán đổi A/B ngẫu nhiên với seed cố định
  để tránh thiên lệch vị trí ở nguồn luôn liệt kê người thắng trước.
- Chia Train/Validation/Test xấp xỉ 60/20/20 theo thời gian; các trận có cùng timestamp
  nằm cùng tập nên số lượng thực tế có thể lệch tỷ lệ.
- Chọn K và mô hình ML dùng cho giải thích trên Validation, rồi huấn luyện lại trên Train + Validation.
  Tập Test được dùng để báo cáo Log-loss, Brier, Accuracy, ROC-AUC và calibration.
- Đánh giá tuần tự theo nhóm timestamp: kết quả nhóm đã qua trong Validation/Test được phép cập nhật
  Elo và lịch sử cho nhóm sau; mô hình ML không được fit lại trên Test.

**Giới hạn thời gian:** nguồn hiện chỉ có ngày; mọi trận cùng ngày dùng trạng thái trước ngày đó.
Khi cập nhật cuối ngày, thứ tự các kết quả trong cửa sổ phong độ vẫn theo thứ tự nguồn.
Thứ hạng nếu bổ sung phải là snapshot đã công bố trước trận; nguồn thực tế hiện tại không có
`rank_a/rank_b`, vì vậy `rank_diff` bằng 0 và không đóng góp vào dự đoán.
Cấp giải suy luận từ tên là heuristic, chưa xét thay đổi theo từng mùa; nên cung cấp `tour_tier` chính xác.

## Kết quả trên dữ liệu thực tế

Lần chạy ngày 20/10/2026: Train **9.571**, Validation **3.199**, Test **3.212** trận.
Test từ 13/03/2025 đến 05/07/2026. Chọn **K = 48, cố định**;
mô hình giải thích được chọn trên Validation là **HistGradientBoosting**.

| Mô hình | Log-loss ↓ | Brier ↓ | Accuracy ↑ | AUC ↑ |
|---|---:|---:|---:|---:|
| Luôn đoán 0,5 | 0.6931 | 0.2500 | 0.4981 | 0.5000 |
| Elo | 0.5670 | 0.1941 | 0.6912 | 0.7714 |
| Elo + Tier Elo | 0.5672 | 0.1939 | 0.6955 | 0.7732 |
| Logistic Regression | 0.5655 | 0.1928 | 0.7052 | 0.7769 |
| HistGradientBoosting | 0.5642 | 0.1922 | 0.7045 | 0.7778 |

HistGradientBoosting có Log-loss thấp nhất trong lần chạy này. Đây là đánh giá trên phạm vi
trận có trong nguồn, chưa phải bằng chứng mô hình đạt hiệu quả tương tự cho toàn bộ lịch BWF.

## Báo cáo và dashboard

Pipeline xuất `data/processed/features_matches.csv` và các tệp trong `reports/`:

- `results.csv`, `discipline_comparison.csv`: kết quả toàn bộ và theo nội dung.
- `feature_importance.csv/.png`: permutation importance và đóng góp tuyến tính
  đã chuẩn hóa trên thang log-odds; không diễn giải là conditional SHAP cho đặc trưng tương quan.
- `calibration.png`: so sánh xác suất dự đoán và tỷ lệ thắng thực tế.
- `ratings.csv`: Elo toàn cục và theo cấp giải tại cuối lịch sử.
- `metadata.json`: nguồn, SHA-256, K, cấu hình, điểm Validation và các mốc chia dữ liệu.

File nguồn có sidecar `*.csv.metadata.json` để đánh dấu dữ liệu sinh thử; metadata chỉ được tin dùng
khi hash trùng CSV. CSV nhập ngoài chưa được xác minh nguồn được đánh dấu là chưa rõ nguồn gốc.

Dashboard cache CSV theo thời điểm sửa file, dùng đúng ratings từ pipeline, lọc VĐV cùng nội dung,
và hỗ trợ Elo hoặc Elo + Tier Elo (70/30). Giả lập dùng trạng thái cuối lịch sử, chưa kết nối dữ liệu trực tiếp.

## Kiểm tra chất lượng

```powershell
python -m pytest -q
python -m ruff check .
```

46 test kiểm tra cập nhật Elo, không dùng kết quả cùng ngày để tính đặc trưng,
độc lập MS/WS, validation, ranh giới thời gian, chuyển đổi CSV, lọc tỷ số và tương tác dashboard.
Đã chạy trên Python 3.14.7; workflow GitHub Actions được cấu hình cho Python 3.10 và 3.14.
Nếu môi trường hạn chế đa luồng, có thể đặt `$env:OMP_NUM_THREADS = "1"` trước khi chạy pipeline.

Mã chính nằm trong `src/bwf/`, CLI ở `scripts/`, dashboard ở `app.py`, kiểm thử ở `tests/`.
Xem [ghi nhận rà soát và nâng cấp](docs/review.md).
