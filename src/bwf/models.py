from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def temporal_split(df: pd.DataFrame, fractions=(0.6, 0.2, 0.2)):
    """Chia theo THỜI GIAN (không xáo trộn): train -> validation -> test."""
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or (fractions <= 0).any():
        raise ValueError("Cần ba tỷ lệ chia dương hữu hạn.")
    if not np.isclose(fractions.sum(), 1):
        raise ValueError("Tổng tỷ lệ chia phải bằng 1.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].isna().any():
        raise ValueError("Ngày thi đấu bị thiếu.")
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    n = len(df)
    i, j = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    # Keep every equal timestamp on one side of a boundary.
    dates = df["date"].to_numpy()
    if i < n:
        i = int(dates.searchsorted(dates[i], side="left"))
    if j < n:
        j = int(dates.searchsorted(dates[j], side="left"))
    if not 0 < i < j < n:
        raise ValueError("Không đủ ngày thi đấu để chia train/validation/test không rỗng.")
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "accuracy": accuracy_score(y, p >= 0.5),
        "auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
    }


def elo_only_proba(df: pd.DataFrame) -> np.ndarray:
    return 1.0 / (1.0 + 10 ** (-df["elo_diff"].to_numpy() / 400.0))


def tier_elo_blend_proba(df: pd.DataFrame, w_global: float = 0.7) -> np.ndarray:
    if "tier_elo_diff" not in df.columns:
        return elo_only_proba(df)
    blended_diff = w_global * df["elo_diff"].to_numpy() + (1.0 - w_global) * df["tier_elo_diff"].to_numpy()
    return 1.0 / (1.0 + 10 ** (-blended_diff / 400.0))


def make_models() -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)),
        "hist_gb": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.04, max_iter=150, min_samples_leaf=30, random_state=0
        ),
    }


def evaluate_all(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str] | None = None
) -> pd.DataFrame:
    fcols = feature_cols or [c for c in FEATURES if c in train.columns]
    y_tr, y_te = train["y"], test["y"]
    rows = {
        "baseline: luôn đoán 0.5": metrics(y_te, np.full(len(test), 0.5)),
        "baseline: chỉ Elo": metrics(y_te, elo_only_proba(test)),
    }
    if "tier_elo_diff" in test.columns:
        rows["baseline: Elo + Tier Elo"] = metrics(y_te, tier_elo_blend_proba(test))
    for name, m in make_models().items():
        m.fit(train[fcols], y_tr)
        rows[name] = metrics(y_te, m.predict_proba(test[fcols])[:, 1])
    return pd.DataFrame(rows).T.round(4)


def evaluate_by_discipline(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """So sánh mức độ dự đoán được giữa Đơn Nam (MS) và Đơn Nữ (WS)."""
    if "discipline" not in test.columns:
        return pd.DataFrame()
    fcols = [c for c in FEATURES if c in train.columns]
    m_lr = make_models()["logreg"].fit(train[fcols], train["y"])
    out = {}
    for disc in sorted(test["discipline"].dropna().unique()):
        sub = test[test["discipline"] == disc]
        if len(sub) < 10:
            continue
        p_elo = elo_only_proba(sub)
        p_lr = m_lr.predict_proba(sub[fcols])[:, 1]
        m_e = metrics(sub["y"], p_elo)
        m_l = metrics(sub["y"], p_lr)
        out[disc] = {
            "n_matches": len(sub),
            "elo_log_loss": round(m_e["log_loss"], 4),
            "logreg_log_loss": round(m_l["log_loss"], 4),
            "logreg_brier": round(m_l["brier"], 4),
            "logreg_auc": round(m_l["auc"], 4),
            "logreg_accuracy": round(m_l["accuracy"], 4),
        }
    return pd.DataFrame(out).T


def compute_feature_attribution(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Permutation importance and centered linear contributions on the log-odds scale."""
    fcols = list(X_test.columns)
    perm = permutation_importance(model, X_test, y_test, scoring="neg_log_loss", n_repeats=10, random_state=0)
    rows = []
    # Centered linear contributions; no conditional SHAP claim for correlated features.
    has_linear = hasattr(model, "named_steps") and "logisticregression" in model.named_steps
    if has_linear:
        scaler = model.named_steps["standardscaler"]
        lr = model.named_steps["logisticregression"]
        x_std = scaler.transform(X_test)
        mean_abs_shap = np.mean(np.abs(x_std * lr.coef_[0]), axis=0)
        coefs = lr.coef_[0]
    else:
        mean_abs_shap = np.full(len(fcols), np.nan)
        coefs = np.full(len(fcols), np.nan)

    for idx, col in enumerate(fcols):
        rows.append(
            {
                "feature": col,
                "perm_importance_logloss": round(float(perm.importances_mean[idx]), 5),
                "perm_std": round(float(perm.importances_std[idx]), 5),
                "mean_abs_linear_contribution_logodds": round(float(mean_abs_shap[idx]), 4),
                "linear_coef": round(float(coefs[idx]), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("perm_importance_logloss", ascending=False).reset_index(drop=True)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    frac, mean_p = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_p, "fraction_won": frac})
