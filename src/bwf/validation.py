"""Validation shared by CSV ingestion and point-in-time feature generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_matches(matches: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "player_a", "player_b", "a_won"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(sorted(missing))}")
    if matches.empty:
        raise ValueError("Dữ liệu trận đấu rỗng.")
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Ngày thi đấu bị thiếu hoặc không hợp lệ.")
    for col in ("player_a", "player_b"):
        df[col] = df[col].astype("string").str.strip()
        if df[col].isna().any() or df[col].str.lower().isin(["", "nan", "none", "unknown"]).any():
            raise ValueError(f"Tên vận động viên bị thiếu ở cột {col}.")
    if df["player_a"].eq(df["player_b"]).any():
        raise ValueError("Một trận phải có hai vận động viên khác nhau.")
    df["a_won"] = pd.to_numeric(df["a_won"], errors="coerce")
    if not df["a_won"].isin([0, 1]).all():
        raise ValueError("a_won chỉ được nhận giá trị 0 hoặc 1.")
    df["a_won"] = df["a_won"].astype(int)
    for col in ("rank_a", "rank_b"):
        if col in df:
            values = pd.to_numeric(df[col], errors="coerce")
            invalid = df[col].notna() & (values.isna() | ~np.isfinite(values) | (values <= 0))
            if invalid.any():
                raise ValueError(f"{col} phải là số dương hoặc để trống.")
            df[col] = values
    if "discipline" in df:
        df["discipline"] = df["discipline"].astype("string").str.strip().str.upper()
        if not df["discipline"].isin(["MS", "WS"]).all():
            raise ValueError("discipline chỉ hỗ trợ MS và WS.")
    return df.sort_values("date", kind="stable").reset_index(drop=True)
