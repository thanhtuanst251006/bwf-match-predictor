"""python scripts/run_experiment.py --synthetic   |   --data data/raw/matches.csv"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bwf.features import FEATURES, build_features, randomize_sides  # noqa: E402
from bwf.models import (  # noqa: E402
    calibration_table,
    compute_feature_attribution,
    elo_only_proba,
    evaluate_all,
    evaluate_by_discipline,
    make_models,
    metrics,
    temporal_split,
)
from bwf.synthetic import make_synthetic_matches  # noqa: E402
from bwf.validation import validate_matches  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Đường dẫn file CSV chuẩn hóa (vd: data/raw/matches.csv)")
    source.add_argument("--synthetic", action="store_true", help="Chạy trên dữ liệu giả lập")
    ap.add_argument(
        "--dynamic-k", action="store_true", help="Điều chỉnh hệ số K theo cấp độ giải Super 1000/750/500/300"
    )
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Root directory for reports and processed data",
    )
    args = ap.parse_args()
    root = Path(args.output_dir).resolve()
    reports = root / "reports"
    processed = root / "data/processed"
    raw = make_synthetic_matches() if args.synthetic else pd.read_csv(args.data)
    source_hash = None if args.synthetic else hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    is_synthetic = True if args.synthetic else None
    data_provenance = None
    if args.data:
        provenance_path = Path(args.data).with_suffix(Path(args.data).suffix + ".metadata.json")
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("sha256") == source_hash:
                is_synthetic = provenance.get("synthetic")
                data_provenance = provenance
    raw = randomize_sides(validate_matches(raw), seed=0)

    # 1) Chọn hệ số K và chế độ Dynamic K của Elo trên tập validation (theo thời gian), KHÔNG nhìn tập test
    best_k, best_dyn, best_ll = 32, args.dynamic_k, 9.0
    dyn_options = [True, False] if not args.dynamic_k else [True]
    print("=== 1. Tinh chỉnh siêu tham số Elo trên tập Validation ===")
    for dyn in dyn_options:
        for k in [8, 16, 24, 32, 48, 64]:
            _, val, _ = temporal_split(build_features(raw, k=k, dynamic_k=dyn))
            ll = metrics(val["y"], elo_only_proba(val))["log_loss"]
            tag = "Dynamic-K" if dyn else "Static-K "
            print(f"[{tag}] K={k:>2}  val log-loss (Elo-only) = {ll:.4f}")
            if ll < best_ll:
                best_k, best_dyn, best_ll = k, dyn, ll
    print(f"-> Chọn trên Validation: K={best_k}, dynamic_k={best_dyn} (val log-loss={best_ll:.4f})\n")

    # 2) Huấn luyện và đánh giá một lần trên tập test
    full_feats = build_features(raw, k=best_k, dynamic_k=best_dyn)
    train, val, test = temporal_split(full_feats)
    train_val = pd.concat([train, val]).reset_index(drop=True)

    # Select ML using validation only, before looking at the test metrics.
    validation_scores = {}
    for name, model in make_models().items():
        model.fit(train[FEATURES], train["y"])
        validation_scores[name] = metrics(val["y"], model.predict_proba(val[FEATURES])[:, 1])["log_loss"]
    best_ml_name = min(validation_scores, key=validation_scores.get)

    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    full_feats.to_csv(processed / "features_matches.csv", index=False)
    pd.DataFrame(full_feats.attrs["ratings"]).to_csv(reports / "ratings.csv", index=False)

    print("=== 2. Kết quả trên tập Test (Out-of-time Evaluation) ===")
    res = evaluate_all(train_val, test)
    print(res.to_string())

    res.to_csv(reports / "results.csv")

    # 3) So sánh mức độ dự đoán được giữa Đơn Nam (MS) và Đơn Nữ (WS)
    disc_res = evaluate_by_discipline(train_val, test)
    disc_res.to_csv(reports / "discipline_comparison.csv")
    if not disc_res.empty:
        print("\n=== 3. So sánh Đơn Nam (MS) vs Đơn Nữ (WS) trên tập Test ===")
        print(disc_res.to_string())

    # 4) Phân tích độ đóng góp đặc trưng (Feature Attribution / SHAP Linear & Permutation)
    best_model = make_models()[best_ml_name].fit(train_val[FEATURES], train_val["y"])
    fi_df = compute_feature_attribution(best_model, test[FEATURES], test["y"])
    print(f"\n=== 4. Độ quan trọng đặc trưng ({best_ml_name}) ===")
    print(fi_df.to_string(index=False))
    fi_df.to_csv(reports / "feature_importance.csv", index=False)

    # Vẽ đồ thị Feature Importance
    plt.figure(figsize=(6.5, 4.2))
    fi_sorted = fi_df.sort_values("perm_importance_logloss", ascending=True)
    plt.barh(fi_sorted["feature"], fi_sorted["perm_importance_logloss"], color="#2b5c8f")
    plt.xlabel("Mức tăng Log-loss khi hoán vị đặc trưng (Permutation Importance)")
    plt.title(f"Feature Attribution trên tập Test ({best_ml_name})")
    plt.tight_layout()
    plt.savefig(reports / "feature_importance.png", dpi=150)
    plt.close()

    # 5) Đồ thị calibration so sánh Elo-only và mô hình ML tốt nhất
    p_ml = best_model.predict_proba(test[FEATURES])[:, 1]
    p_elo = elo_only_proba(test)
    cal_ml = calibration_table(test["y"], p_ml)
    cal_elo = calibration_table(test["y"], p_elo)

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Hiệu chỉnh hoàn hảo (y = x)")
    plt.plot(
        cal_elo["mean_predicted"], cal_elo["fraction_won"], "s--", color="#ff7f0e", label="Baseline: Chỉ Elo"
    )
    plt.plot(
        cal_ml["mean_predicted"], cal_ml["fraction_won"], "o-", color="#1f77b4", label=f"ML ({best_ml_name})"
    )
    plt.xlabel("Xác suất thắng dự đoán trung bình")
    plt.ylabel("Tỷ lệ thắng thực tế trên tập Test")
    plt.title("Biểu đồ Calibration (Độ tin cậy xác suất)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(reports / "calibration.png", dpi=150)
    plt.close()

    print(
        "\nĐã lưu toàn bộ báo cáo vào thư mục reports/:"
        "\n - reports/results.csv"
        "\n - reports/discipline_comparison.csv"
        "\n - reports/feature_importance.csv & reports/feature_importance.png"
        "\n - reports/calibration.png"
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "synthetic" if args.synthetic else str(Path(args.data).resolve()),
        "synthetic": is_synthetic,
        "data_provenance": data_provenance,
        "source_sha256": source_hash,
        "k": best_k,
        "dynamic_k": best_dyn,
        "validation_log_loss": best_ll,
        "selected_ml_model": best_ml_name,
        "ml_validation_log_loss": validation_scores,
        "evaluation": "timestamp batches: only strictly earlier timestamps contribute to features",
        "splits": {
            name: {"n": len(frame), "start": str(frame["date"].min()), "end": str(frame["date"].max())}
            for name, frame in [("train", train), ("validation", val), ("test", test)]
        },
    }
    (reports / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        sys.exit(str(exc))
