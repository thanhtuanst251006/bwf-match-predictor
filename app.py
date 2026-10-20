"""BWF experiment reports and Elo simulator. Run: python -m streamlit run app.py."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("BWF_OUTPUT_DIR", ROOT)).resolve()
sys.path.insert(0, str(ROOT / "src"))
from bwf.features import ELO_INIT, expected_score  # noqa: E402

st.set_page_config(page_title="BWF Match Predictor", page_icon="🏸", layout="wide")
st.title("🏸 BWF Match Predictor")
st.caption("Xác suất thắng trận đơn cầu lông · Elo trước trận · Đánh giá theo thời gian")


@st.cache_data(max_entries=16)
def load_csv(path: str, modified: int, index_col: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col)


def read_csv(path: Path, index_col: int | None = None) -> pd.DataFrame:
    return load_csv(str(path), path.stat().st_mtime_ns, index_col)


reports = OUTPUT_ROOT / "reports"
feats_path = OUTPUT_ROOT / "data/processed/features_matches.csv"
required = [reports / name for name in ("results.csv", "ratings.csv", "metadata.json")]
if not feats_path.exists() or any(not path.exists() for path in required):
    st.warning("Cần tạo báo cáo bằng pipeline hiện tại trước khi xem dashboard.")
    st.code("python scripts/run_experiment.py --data data/raw/bwf_real_matches.csv", language="bash")
    st.stop()

try:
    metadata = json.loads((reports / "metadata.json").read_text(encoding="utf-8"))
    res_df = read_csv(reports / "results.csv", index_col=0)
    feats_df = read_csv(feats_path)
    ratings = read_csv(reports / "ratings.csv")
except (ValueError, OSError) as exc:
    st.error(f"Không đọc được báo cáo: {exc}")
    st.stop()

if metadata.get("synthetic") is True:
    st.warning("Dữ liệu GIẢ LẬP: kết quả dùng kiểm thử pipeline, chưa đánh giá khả năng dự đoán BWF thực tế.")
elif metadata.get("synthetic") is None:
    st.info(
        "Nguồn CSV chưa được xác minh. Cần kiểm tra nguồn và thời điểm công bố thứ hạng trước khi sử dụng."
    )
else:
    st.info(
        "Dữ liệu kết quả thi đấu thực tế từ kho cộng đồng tổng hợp BWF. "
        "[Xem nguồn](https://github.com/SahilMotyar/bwf-match-data). "
        "Nguồn không có thứ hạng lịch sử hoặc giờ bắt đầu trận; phạm vi phụ thuộc kho dữ liệu."
    )

c1, c2, c3 = st.columns(3)
c1.metric("Số trận", f"{len(feats_df):,}")
c2.metric("Số trận Test", metadata["splits"]["test"]["n"])
c3.metric("Cấu hình Elo", f"K = {metadata['k']}" + (" · động" if metadata["dynamic_k"] else " · cố định"))
st.caption(
    f"Lịch sử: {feats_df['date'].min()[:10]} → {feats_df['date'].max()[:10]}. "
    "K và mô hình ML được chọn trên Validation; các trận cùng thời điểm được giữ cùng tập. "
    "Các trận cùng timestamp dùng chung trạng thái trước nhóm; chỉ cập nhật kết quả cho timestamp sau."
)

tab1, tab2, tab3, tab4 = st.tabs(["Hiệu năng", "Giải thích mô hình", "MS và WS", "Giả lập đối đầu"])
with tab1:
    st.subheader("Kết quả trên tập Test")
    st.dataframe(res_df, width="stretch")
    winner = res_df["log_loss"].idxmin()
    st.info(f"Log-loss thấp nhất trong lần chạy này: {winner} ({res_df.loc[winner, 'log_loss']:.4f}).")
    st.caption("Log-loss và Brier càng thấp càng tốt; Accuracy và AUC càng cao càng tốt.")
    if (reports / "calibration.png").exists():
        st.image(str(reports / "calibration.png"), caption="Độ tin cậy xác suất trên tập Test")
    st.download_button("Tải kết quả CSV", res_df.to_csv().encode("utf-8-sig"), "results.csv", "text/csv")

with tab2:
    st.subheader(f"Độ đóng góp đặc trưng · {metadata['selected_ml_model']}")
    st.caption("Mô hình được chọn bằng Log-loss trên Validation. Hoán vị được đo trên tập Test.")
    if (reports / "feature_importance.csv").exists():
        st.dataframe(read_csv(reports / "feature_importance.csv"), width="stretch")
    if (reports / "feature_importance.png").exists():
        st.image(str(reports / "feature_importance.png"), caption="Mức tăng Log-loss khi hoán vị đặc trưng")

with tab3:
    st.subheader("So sánh đơn nam (MS) và đơn nữ (WS)")
    if (reports / "discipline_comparison.csv").exists():
        comparison = read_csv(reports / "discipline_comparison.csv", index_col=0)
        if not comparison.empty:
            st.dataframe(comparison, width="stretch")
        else:
            st.info("Chưa đủ dữ liệu từng nội dung để so sánh (ít nhất 10 trận Test mỗi nội dung).")
    if metadata.get("synthetic") is True:
        st.caption("Dữ liệu giả lập: chênh lệch phản ánh quy luật sinh dữ liệu.")
    else:
        st.caption(
            "So sánh trong phạm vi tập Test hiện có; chưa kiểm soát khác biệt lịch đấu và số trận mỗi VĐV."
        )

with tab4:
    st.subheader("Giả lập xác suất thắng từ Elo cuối lịch sử")
    discipline = st.selectbox("Nội dung", sorted(ratings["discipline"].unique()), key="discipline")
    current = ratings[ratings["discipline"] == discipline]
    global_elo = current[current["tier"] == "global"].set_index("player")["elo"]
    players = sorted(global_elo.index)
    if len(players) < 2:
        st.info("Cần ít nhất hai vận động viên cùng nội dung.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    pa = c1.selectbox("Vận động viên A", players, key=f"player_a_{discipline}")
    pb = c2.selectbox("Vận động viên B", [p for p in players if p != pa], key=f"player_b_{discipline}")
    tiers = sorted(set(current["tier"]) - {"global"})
    tier = c3.selectbox("Cấp độ giải", tiers)
    method = st.selectbox("Phương pháp", ["Elo", "Elo + Tier Elo (70/30)"])
    ra, rb = global_elo[pa], global_elo[pb]
    if method != "Elo":
        tier_ratings = current[current["tier"] == tier].set_index("player")["elo"]
        ra = 0.7 * ra + 0.3 * tier_ratings.get(pa, ELO_INIT)
        rb = 0.7 * rb + 0.3 * tier_ratings.get(pb, ELO_INIT)
    prob = expected_score(ra, rb)
    history = feats_df
    if "discipline" in history:
        history = history[history["discipline"] == discipline]
    direct = history[(history["player_a"] == pa) & (history["player_b"] == pb)]
    reverse = history[(history["player_a"] == pb) & (history["player_b"] == pa)]
    wa = int(direct["y"].sum() + (1 - reverse["y"]).sum())
    wb = len(direct) + len(reverse) - wa
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Elo toàn cục · {pa}", f"{global_elo[pa]:.1f}")
    m2.metric(f"Elo toàn cục · {pb}", f"{global_elo[pb]:.1f}")
    m3.metric("Đối đầu A – B", f"{wa} – {wb}")
    st.success(f"{pa}: {prob:.1%} · {pb}: {1 - prob:.1%}")
    st.progress(float(prob), text=f"Xác suất thắng của {pa}")
    st.caption(
        "Sử dụng đúng Elo và K của lần thực nghiệm. Cấp giải ảnh hưởng dự đoán khi chọn Elo + Tier Elo; "
        "Elo thuần dùng điểm toàn cục. Đây là mô phỏng tại cuối lịch sử, không phải dự báo trực tiếp."
    )
