"""Chuẩn hóa dữ liệu trận đấu cầu lông từ Kaggle sang schema chuẩn (data/raw/matches.csv).

Cách dùng:
    python scripts/prepare_kaggle_data.py --input path/to/kaggle_bwf.csv --output data/raw/matches.csv
    python scripts/prepare_kaggle_data.py --generate-sample --output data/raw/matches.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bwf.features import infer_tier  # noqa: E402
from bwf.synthetic import make_synthetic_matches  # noqa: E402
from bwf.validation import validate_matches  # noqa: E402

PLAYER_ALIASES = {
    "viktor axelsen": "Viktor AXELSEN",
    "axelsen viktor": "Viktor AXELSEN",
    "an se-young": "AN Se Young",
    "an se young": "AN Se Young",
    "chen yufei": "CHEN Yu Fei",
    "chen yu fei": "CHEN Yu Fei",
    "shi yuqi": "SHI Yu Qi",
    "shi yu qi": "SHI Yu Qi",
    "tai tzu-ying": "TAI Tzu Ying",
    "tai tzu ying": "TAI Tzu Ying",
    "lee zii jia": "LEE Zii Jia",
    "kunlavut vitidsarn": "Kunlavut VITIDSARN",
    "anthony sinisuka ginting": "Anthony Sinisuka GINTING",
    "jonatan christie": "Jonatan CHRISTIE",
    "akane yamaguchi": "Akane YAMAGUCHI",
    "carolina marin": "Carolina MARIN",
    "kodai naraoka": "Kodai NARAOKA",
    "li shifeng": "LI Shi Feng",
    "li shi feng": "LI Shi Feng",
    "anders antonsen": "Anders ANTONSEN",
    "loh kean yew": "LOH Kean Yew",
    "chou tien-chen": "CHOU Tien Chen",
    "chou tien chen": "CHOU Tien Chen",
    "lakshya sen": "Lakshya SEN",
}


def normalize_name(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    cleaned = re.sub(r"\s+", " ", name.strip())
    key = cleaned.lower()
    return PLAYER_ALIASES.get(key, cleaned)


def is_walkover_or_retired(score_val: str | float | None) -> bool:
    if pd.isna(score_val):
        return False
    s = str(score_val).lower()
    return any(tok in s for tok in ("w/o", "walkover", "ret", "retired", "wdn", "dns"))


def convert_kaggle_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower().strip(): c for c in raw.columns}
    df = raw.copy()

    # 1. Lọc bỏ các trận Walkover / Retired nếu có cột score/status
    for scol in ("score", "scores", "result", "match_status"):
        if scol in cols_lower:
            c = cols_lower[scol]
            df = df[~df[c].apply(is_walkover_or_retired)].copy()

    # 2. Xác định cột ngày thi đấu
    date_col = None
    for cand in ("date", "match_date", "tournament_date", "start_date", "day"):
        if cand in cols_lower:
            date_col = cols_lower[cand]
            break
    if date_col is None:
        raise ValueError(f"Không tìm thấy cột ngày trong các cột: {list(raw.columns)}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")

    # 3. Xác định người chơi và kết quả a_won
    if "player_a" in cols_lower and "player_b" in cols_lower and "a_won" in cols_lower:
        out["player_a"] = df[cols_lower["player_a"]].map(normalize_name)
        out["player_b"] = df[cols_lower["player_b"]].map(normalize_name)
        out["a_won"] = pd.to_numeric(df[cols_lower["a_won"]], errors="coerce")
    elif "winner" in cols_lower and "loser" in cols_lower:
        out["player_a"] = df[cols_lower["winner"]].map(normalize_name)
        out["player_b"] = df[cols_lower["loser"]].map(normalize_name)
        out["a_won"] = 1
    elif "team_one_players" in cols_lower and "team_two_players" in cols_lower:
        out["player_a"] = df[cols_lower["team_one_players"]].map(normalize_name)
        out["player_b"] = df[cols_lower["team_two_players"]].map(normalize_name)
        w_col = cols_lower.get("winner", cols_lower.get("winning_team"))
        if w_col is None:
            raise ValueError("Thiếu cột winner cho định dạng team_one_players/team_two_players")
        winners = df[w_col].astype("string").str.strip().str.lower()
        out["a_won"] = winners.map(
            {
                "1": 1,
                "1.0": 1,
                "team_one": 1,
                "player a": 1,
                "2": 0,
                "2.0": 0,
                "team_two": 0,
                "player b": 0,
            }
        )
        if out["a_won"].isna().any():
            raise ValueError(
                "Không nhận diện được winner: cần 1/2, team_one/team_two hoặc Player A/Player B."
            )
    else:
        raise ValueError(
            "Không nhận diện được cột VĐV. Cần (player_a, player_b, a_won) hoặc (winner, loser)."
        )

    # 4. Cột tùy chọn: rank_a, rank_b, discipline, tournament, tour_tier
    for a_cand, b_cand in [
        ("rank_a", "rank_b"),
        ("winner_rank", "loser_rank"),
        ("player_a_rank", "player_b_rank"),
        ("team_one_rank", "team_two_rank"),
    ]:
        if a_cand in cols_lower and b_cand in cols_lower:
            out["rank_a"] = pd.to_numeric(df[cols_lower[a_cand]], errors="coerce")
            out["rank_b"] = pd.to_numeric(df[cols_lower[b_cand]], errors="coerce")
            break

    for d_cand in ("discipline", "category", "event", "match_type", "type"):
        if d_cand in cols_lower:
            out["discipline"] = df[cols_lower[d_cand]].astype(str).str.upper().str.strip()
            out = out[out["discipline"].isin(["MS", "WS", "MEN'S SINGLES", "WOMEN'S SINGLES"])].copy()
            out["discipline"] = out["discipline"].replace({"MEN'S SINGLES": "MS", "WOMEN'S SINGLES": "WS"})
            break

    for t_cand in ("tournament", "tournament_name", "tourney_name", "event_name"):
        if t_cand in cols_lower:
            out["tournament"] = df[cols_lower[t_cand]].astype(str).str.strip()
            break

    tier_col = cols_lower.get("tour_tier", cols_lower.get("tournament_type", cols_lower.get("level")))
    if "tournament" in out.columns or tier_col:
        t_series = out["tournament"] if "tournament" in out.columns else pd.Series(None, index=out.index)
        tr_series = df.loc[out.index, tier_col] if tier_col else pd.Series(None, index=out.index)
        out["tour_tier"] = [infer_tier(t, tr) for t, tr in zip(t_series, tr_series)]

    out = out.dropna(subset=["date", "player_a", "player_b", "a_won"])
    out = out[out["player_a"] != out["player_b"]]
    return validate_matches(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chuẩn hóa dataset BWF Kaggle sang chuẩn dự án.")
    ap.add_argument("--input", help="Đường dẫn tới file CSV tải từ Kaggle")
    ap.add_argument("--output", default="data/raw/matches.csv", help="Đường dẫn lưu file chuẩn hóa")
    ap.add_argument(
        "--generate-sample",
        action="store_true",
        help="Tạo file data/raw/matches.csv mô phỏng cấu trúc BWF World Tour (6000 trận MS & WS)",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.generate_sample:
        df = make_synthetic_matches(n_players=60, n_matches=6000, seed=42)
        df.to_csv(out_path, index=False)
        write_provenance(out_path, synthetic=True)
        print(f"Đã tạo dữ liệu mẫu chuẩn BWF ({len(df)} trận) tại: {out_path}")
        return

    if not args.input:
        sys.exit("Vui lòng truyền --input <kaggle.csv> hoặc --generate-sample")

    raw = pd.read_csv(args.input)
    clean = convert_kaggle_dataframe(raw)
    clean.to_csv(out_path, index=False)
    write_provenance(out_path, synthetic=None)
    print(f"Đã chuẩn hóa {len(clean)} trận đơn từ '{args.input}' -> '{out_path}'")


def write_provenance(path: Path, synthetic: bool | None) -> None:
    payload = {"synthetic": synthetic, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
