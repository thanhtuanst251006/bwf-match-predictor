# Schema dữ liệu đầu vào & Hướng dẫn chuẩn hóa Kaggle

File CSV đầu vào (`data/raw/matches.csv`), mỗi dòng là một trận **ĐƠN (Singles: MS hoặc WS)**.

## 1. Các cột chuẩn

| Cột | Bắt buộc | Ý nghĩa |
|---|---|---|
| `date` | Có | Ngày thi đấu (chuỗi ISO `YYYY-MM-DD` hoặc định dạng `pd.to_datetime` đọc được) |
| `player_a`, `player_b` | Có | Tên vận động viên (đã chuẩn hóa khoảng trắng & chữ hoa/thường xuyên suốt các năm) |
| `a_won` | Có | `1` nếu `player_a` thắng, `0` nếu `player_b` thắng |
| `rank_a`, `rank_b` | Tùy chọn | Hạng BWF thế giới tại thời điểm thi đấu (số nhỏ = thứ hạng cao) |
| `discipline` | Tùy chọn | Nội dung thi đấu: `MS` (Đơn Nam) hoặc `WS` (Đơn Nữ) |
| `tournament` | Tùy chọn | Tên giải đấu (vd: `All England Open`, `Indonesia Open`, `Swiss Open`) |
| `tour_tier` | Tùy chọn | Cấp độ giải (`Super 1000`, `Super 750`, `Super 500`, `Super 300`, `Olympics`, ...) |
| `round` | Tùy chọn | Vòng đấu (`R32`, `R16`, `QF`, `SF`, `F`) |

## 2. Chuyển đổi tự động từ bộ dữ liệu Kaggle
Hầu hết các dataset cầu lông trên Kaggle (ví dụ: *BWF World Tour Matches*, *Badminton Tournament Software Dataset*) lưu dưới dạng `winner`/`loser` hoặc `team_one_players`/`team_two_players`.

Sử dụng script có sẵn để tự động lọc bỏ các trận bỏ cuộc (`Walkover`, `Retired`), chuẩn hóa tên VĐV, suy luận cấp độ giải đấu (`tour_tier`) và xuất ra `data/raw/matches.csv`:

```bash
# Chuyển đổi từ file CSV tải trên Kaggle:
python scripts/prepare_kaggle_data.py --input path/to/kaggle_bwf.csv --output data/raw/matches.csv

# Hoặc tạo bộ dữ liệu mẫu mô phỏng BWF World Tour (6,000 trận MS & WS) để chạy thử ngay:
python scripts/prepare_kaggle_data.py --generate-sample --output data/raw/matches.csv
```

> **Lưu ý chống rò rỉ nhãn (Positional Bias):** Khi chuyển từ định dạng `winner, loser` sang `player_a, player_b` (`a_won = 1`), hàm `randomize_sides()` trong pipeline sẽ tự động hoán đổi ngẫu nhiên vị trí A/B với xác suất 50% trước khi huấn luyện để mô hình không học vẹt vị trí cột.

## 3. Kiểm tra và giới hạn

Pipeline yêu cầu tên không rỗng, hai VĐV khác nhau, ngày hợp lệ và nhãn đúng 0/1.
Thứ hạng nếu có phải dương hữu hạn; thiếu hạng được xử lý bằng `rank_diff = 0`.
Nếu có cột `discipline`, chỉ nhận MS/WS. Nếu không có, coi là một nội dung đơn chung;
không thể tự tách nam/nữ chỉ từ tên VĐV.

Bộ chuyển đổi CSV loại dòng thiếu ngày/tên/nhãn và trận tự đối đầu; dừng nếu kết quả cuối rỗng.
Winner dạng đội chỉ hỗ trợ 1/2, team_one/team_two hoặc Player A/Player B.
Nhãn đội khác cần được ánh xạ rõ ràng trước khi nhập.

Nguồn phải cung cấp thứ hạng đã công bố trước trận và thứ tự thực tế của các dòng cùng ngày.
Các dòng trùng timestamp được giữ cùng tập Train/Validation/Test và dùng chung trạng thái trước nhóm;
kết quả trong nhóm chỉ cập nhật đặc trưng của timestamp tiếp theo. Khi chỉ có ngày, cách tính này
không sử dụng kết quả các trận khác trong cùng ngày.
Cấp giải suy luận theo tên không thay thế được lịch sử cấp giải từng mùa.

`data/raw/matches.csv` là mẫu giả lập seed 42; `data/raw/bwf_real_matches.csv` là dữ liệu thực tế
đã lọc, xem [chi tiết nguồn](real_data.md). Script chuẩn hóa xuất thêm
`*.csv.metadata.json` chứa hash và cờ `synthetic`; nguồn nhập ngoài mặc định chưa được xác minh.
