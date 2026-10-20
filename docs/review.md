# Rà soát dự án ngày 20/10/2026

## Các vấn đề đã sửa

| Vấn đề phát hiện | Thay đổi |
|---|---|
| Simulator dùng K=24 dù pipeline chọn K khác | Pipeline xuất ratings cuối cùng; app đọc trực tiếp |
| Chọn cấp giải trong app không thay đổi xác suất | Thêm chế độ Elo + Tier Elo (70/30) |
| App cho phép chọn VĐV nam đối đầu VĐV nữ | Lọc theo nội dung, loại chính VĐV A khỏi lựa chọn B |
| Lưu trạng thái theo tên, trộn nội dung nếu tên trùng | Khóa trạng thái bằng `(discipline, player)` |
| Mô hình giải thích được chọn bằng điểm Test | Chọn trên Validation trước khi tính điểm Test |
| Cùng ngày bị chia sang các tập khác nhau | Dời ranh giới về đầu nhóm timestamp |
| Thiếu kiểm tra nhãn, ngày, tên và thứ hạng | Thêm `validate_matches` dùng chung |
| Tên thiếu trở thành chuỗi `nan` khi nhập CSV | Giữ missing để loại đúng dòng |
| Tier bị lệch sau khi lọc MD/WD/XD | Căn chỉnh index trước khi ghép tier |
| Winner không nhận diện được bị ngầm gán B thắng | Dừng với thông báo lỗi rõ ràng |
| Log-loss lỗi khi nhóm đánh giá chỉ có một nhãn | Chỉ định `labels=[0, 1]`; AUC thiếu trả NaN |
| Sinh mẫu lỗi với ít VĐV | Chia đúng số VĐV giữa hai pool, kiểm tra tham số |
| Requirements pin cũ không phù hợp Python đang dùng | Gom dependencies vào pyproject, thêm build backend và dev extras |
| App dùng đường dẫn phụ thuộc thư mục chạy | Đường dẫn theo vị trí app, hỗ trợ BWF_OUTPUT_DIR |
| Đọc lại CSV và tái dựng Elo mỗi lần chọn widget | Cache CSV có giới hạn, đọc ratings đã xuất |
| Kết quả giả lập bị diễn giải thành phát hiện thực tế | Đối chiếu dữ liệu với generator seed 42, bổ sung provenance và sửa tài liệu |
| Khẳng định SHAP chính xác với đặc trưng tương quan | Đặt tên đúng là đóng góp tuyến tính đã chuẩn hóa |

## Kiểm chứng

- Bộ gốc: 7 test đạt. Sau nâng cấp: 32 test đạt, gồm Streamlit AppTest.
- Chạy đầy đủ pipeline trên 6.000 dòng CSV đi kèm; đã xem hai biểu đồ xuất ra.
- Ruff lint đạt. Cấu hình CI thêm kiểm tra Python 3.10/3.14; chưa có lượt chạy CI từ GitHub trong phiên này.
- Ruff format đạt; đóng gói thành công wheel `bwf-0.2.0-py3-none-any.whl` bằng build cục bộ không mạng.
- Dashboard với báo cáo thực tế chạy qua AppTest từ thư mục khác; bộ lọc WS hiển thị đúng 30 VĐV.
- Môi trường kiểm chứng: Python 3.14.7, pandas 3.0.6, scikit-learn 1.8.0, Streamlit 1.65.0.
- Sandbox Windows hạn chế pipe và thư mục tạm: pipeline được kiểm chứng với một luồng;
  bộ AppTest chạy ngoài sandbox sau khi được cơ chế phê duyệt tự động cho phép.

## Hướng phát triển tiếp

Ưu tiên thu thập dữ liệu thật có nguồn, giờ thi đấu, ID vận động viên ổn định và snapshot thứ hạng
trước trận. Sau đó thực hiện đánh giá nhiều cửa sổ thời gian (walk-forward) và khoảng bất định.
Các test hiện tại xác minh logic tuần tự; chưa thể bảo đảm nguồn CSV không chứa thông tin công bố sau trận.
Chưa kết nối lịch BWF trực tiếp hoặc triển khai dashboard lên máy chủ.

Tài liệu API tham khảo khi kiểm tra:
[Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest),
[scikit-learn log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
