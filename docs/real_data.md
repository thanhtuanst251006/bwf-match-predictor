# Dữ liệu BWF thực tế

Thu thập và chạy pipeline ngày 20/10/2026.

## Nguồn và khả năng tái lập

- Kho dữ liệu: [SahilMotyar/bwf-match-data](https://github.com/SahilMotyar/bwf-match-data).
- Commit cố định: `0a6899162bf4b9639b76dcdbc0dca4c5d4800664`.
- Bản gốc: `data/external/bwf-match-data/ms.csv`, `ws.csv` và `README.md` của nguồn.
- Tập chuẩn hóa: `data/raw/bwf_real_matches.csv` và sidecar `.csv.metadata.json`.
- Chạy lại: `python scripts/fetch_real_data.py --offline` hoặc bỏ `--offline` để tải lại đúng commit.
- Script xác minh SHA-256 từng tệp nguồn trước khi xử lý; sidecar ghi hash đầu ra, bộ lọc và số lượng.

Theo tác giả, dữ liệu được tổng hợp từ kết quả giải đấu công khai do BWF phục vụ.
Đây là bản tổng hợp cộng đồng; dự án không nhận dữ liệu trực tiếp qua API chính thức BWF và
không khẳng định đã đối soát từng trận. README gốc được giữ nguyên để lưu thông tin tác giả,
cách thu thập và tuyên bố quyền sở hữu dữ liệu của BWF; nguồn không công bố giấy phép mở riêng cho dữ liệu.

Đã kiểm tra mẫu trận chung kết đơn nữ Malaysia Open ngày 11/01/2026:
AN Se Young thắng WANG Zhi Yi `21-15 24-22`, khớp
[bài tường thuật của Badminton Asia](https://badmintonasia.org/2026/01/11/petronas-malaysia-open-2026-an-se-young-wins-third-consecutive-malaysia-open-title/).

## Các bước lọc

| Bước | Số bản ghi |
|---|---:|
| Tải bản gốc MS + WS, 2007–2026 | 49.642 |
| Giữ từ 2018, Super 100/300/500/750/1000 và World Tour Finals | 19.421 |
| Loại vòng loại | 3.109 |
| Loại tỷ số chưa hoàn tất/không hợp lệ hoặc không khớp người thắng | 330 |
| Trùng bản ghi chính xác được loại | 0 |
| Còn lại | **15.982** |

Phạm vi đầu ra: **10/01/2018–05/07/2026**, gồm **8.429 MS**, **7.553 WS** và
**1.278 định danh tên + nội dung**. Không tự gán ID vận động viên khác với tên được nguồn cung cấp.
Giữ nguyên hậu tố phân biệt người trùng tên của nguồn.

Chỉ giữ vòng chính; `QF` là tứ kết và được giữ. Chỉ giữ trận thắng hai game hợp lệ theo hệ 21 điểm,
kiểm tra chênh lệch hai điểm và giới hạn 30. Nguồn không gắn cờ retired riêng; bộ lọc tỷ số
loại các trận chưa hoàn tất thay vì đoán từ nhãn winner. Những bản ghi khác thường được loại,
không tự sửa hoặc bịa tỷ số. Không gồm Olympic, World Championships hay giai đoạn trước World Tour 2018
trong tập dùng huấn luyện; các trận có trong hai CSV gốc vẫn được giữ ở thư mục external.

## Giới hạn ảnh hưởng mô hình

- Chỉ bao phủ các giải có trong nguồn. Không bao gồm kết quả mới hơn 05/07/2026 và không tự cập nhật trực tiếp.
- Không có thứ hạng lịch sử: `rank_diff = 0`, không lấy hạng hiện nay gán cho trận quá khứ.
- Không có giờ thi đấu: đặc trưng của toàn bộ trận cùng ngày dùng trạng thái trước ngày đó.
  Elo được cập nhật theo nhóm với xác suất trước nhóm; kết quả chỉ ảnh hưởng ngày sau.
- Cửa sổ phong độ 10 trận giữ thứ tự nguồn khi có nhiều trận cùng ngày; chưa xác minh thứ tự trong ngày.
- Không gộp giả lập vào tập thật. `data/raw/matches.csv` vẫn là mẫu giả lập cũ và có cờ riêng.
- Kết quả Test phản ánh tập lọc đã mô tả; không suy rộng thành tỷ lệ đúng cho mọi trận BWF.

## Kết quả lần chạy

Train 9.571 trận (đến 26/11/2023), Validation 3.199 trận (29/11/2023–12/03/2025),
Test 3.212 trận (13/03/2025–05/07/2026). K được chọn trên Validation là 48 cố định.
HistGradientBoosting được chọn trên Validation và đạt Test Log-loss 0.5642, Brier 0.1922,
Accuracy 0.7045, ROC-AUC 0.7778. Xem `reports/results.csv` và `reports/metadata.json`.

Kiểm chứng: 46 test đạt, gồm lọc tỷ số hoàn tất, phân biệt vòng loại và QF,
không dùng kết quả cùng ngày, và Streamlit AppTest với các báo cáo thực tế.
