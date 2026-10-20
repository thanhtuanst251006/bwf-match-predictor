"""Xây đặc trưng TRƯỚC trận (point-in-time) cho từng trận đấu.

Mỗi nhóm timestamp chỉ dùng kết quả từ timestamp nhỏ hơn.
Nếu nguồn chỉ có ngày, tất cả trận cùng ngày dùng chung trạng thái trước ngày đó.

Schema đầu vào (một dòng = một trận đơn):
    date (parse được), player_a, player_b, a_won (1 nếu A thắng, 0 nếu B thắng)
    tùy chọn: rank_a, rank_b, tour_tier, tournament, discipline
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .validation import validate_matches

ELO_INIT = 1500.0
FORM_WINDOW = 10
FEATURES = [
    "elo_diff",
    "tier_elo_diff",
    "form_diff",
    "h2h_diff",
    "h2h_n",
    "experience_diff",
    "rest_diff",
    "rank_diff",
]

TIER_K_MULT = {
    "Olympics": 1.40,
    "World Championships": 1.35,
    "World Tour Finals": 1.30,
    "Super 1000": 1.25,
    "Super 750": 1.15,
    "Super 500": 1.00,
    "Super 300": 0.85,
    "Super 100": 0.75,
}


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def infer_tier(tournament: str | float | None, tier_col: str | float | None = None) -> str:
    if isinstance(tier_col, str) and tier_col.strip():
        return tier_col.strip()
    if not isinstance(tournament, str):
        return "Super 500"
    t = tournament.lower()
    if "olympic" in t:
        return "Olympics"
    if "world championship" in t:
        return "World Championships"
    if "finals" in t:
        return "World Tour Finals"
    for level in ("1000", "750", "500", "300", "100"):
        if level in t:
            return f"Super {level}"
    if any(k in t for k in ("all england", "indonesia open", "china open", "malaysia open")):
        return "Super 1000"
    if any(k in t for k in ("denmark open", "japan open", "french open", "singapore open", "india open")):
        return "Super 750"
    if any(k in t for k in ("swiss open", "german open", "taipei open", "spain masters", "orleans")):
        return "Super 300"
    return "Super 500"


def randomize_sides(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Nhiều dataset luôn liệt kê người THẮNG trước -> nhãn rò rỉ qua vị trí.
    Hoán đổi A/B ngẫu nhiên 50% để nhãn không phụ thuộc vị trí."""
    rng = np.random.default_rng(seed)
    swap = rng.random(len(df)) < 0.5
    out = df.copy()
    for a, b in [("player_a", "player_b"), ("rank_a", "rank_b")]:
        if a in out and b in out:
            out.loc[swap, [a, b]] = out.loc[swap, [b, a]].to_numpy()
    out.loc[swap, "a_won"] = 1 - out.loc[swap, "a_won"]
    return out


def build_features(matches: pd.DataFrame, k: float = 32.0, dynamic_k: bool = False) -> pd.DataFrame:
    if not np.isfinite(k) or k <= 0:
        raise ValueError("K phải là số dương hữu hạn.")
    df = validate_matches(matches)

    elo = defaultdict(lambda: ELO_INIT)
    tier_elo = defaultdict(lambda: ELO_INIT)
    form = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    played = defaultdict(int)
    last_date: dict = {}
    h2h = defaultdict(int)  # (x, y) -> số lần x thắng y

    rows = []
    cols = set(df.columns)
    has_rank = {"rank_a", "rank_b"} <= cols
    has_tour = "tournament" in cols
    has_tier = "tour_tier" in cols
    pending = []

    def update_completed_batch():
        # Every prediction at one timestamp uses the same pre-batch state.
        for a, b, d, tier, won, eff_k, e, e_tier in pending:
            elo[a] += eff_k * (won - e)
            elo[b] += eff_k * (e - won)
            tier_elo[(a, tier)] += eff_k * (won - e_tier)
            tier_elo[(b, tier)] += eff_k * (e_tier - won)
            form[a].append(won)
            form[b].append(1 - won)
            played[a] += 1
            played[b] += 1
            last_date[a] = last_date[b] = d
            h2h[(a, b) if won else (b, a)] += 1
        pending.clear()

    for r in df.itertuples():
        if pending and r.date != pending[0][2]:
            update_completed_batch()
        discipline = getattr(r, "discipline", "singles")
        a, b, d = (discipline, r.player_a), (discipline, r.player_b), r.date
        t_name = getattr(r, "tournament", None) if has_tour else None
        t_tier_raw = getattr(r, "tour_tier", None) if has_tier else None
        tier = infer_tier(t_name, t_tier_raw)

        def win_rate(p):
            return float(np.mean(form[p])) if form[p] else 0.5

        def rest(p):
            return min((d - last_date[p]).days, 60) if p in last_date else 30

        wa, wb = h2h[(a, b)], h2h[(b, a)]
        ra_tier, rb_tier = tier_elo[(a, tier)], tier_elo[(b, tier)]
        rows.append(
            {
                "elo_diff": elo[a] - elo[b],
                "tier_elo_diff": ra_tier - rb_tier,
                "form_diff": win_rate(a) - win_rate(b),
                "h2h_diff": (wa - wb) / (wa + wb) if wa + wb else 0.0,
                "h2h_n": wa + wb,
                "experience_diff": np.log1p(played[a]) - np.log1p(played[b]),
                "rest_diff": rest(a) - rest(b),
                "rank_diff": (r.rank_b - r.rank_a)
                if has_rank and pd.notna(r.rank_a) and pd.notna(r.rank_b)
                else 0.0,
            }
        )

        # ---- cập nhật trạng thái SAU khi đã ghi đặc trưng ----
        won = int(r.a_won)
        eff_k = k * TIER_K_MULT.get(tier, 1.0) if dynamic_k else k

        e = expected_score(elo[a], elo[b])
        e_tier = expected_score(ra_tier, rb_tier)
        pending.append((a, b, d, tier, won, eff_k, e, e_tier))

    update_completed_batch()
    feats = pd.DataFrame(rows)
    meta_cols = ["date", "player_a", "player_b"]
    for extra in ("discipline", "tournament", "tour_tier", "round"):
        if extra in cols:
            meta_cols.append(extra)
    out = pd.concat([df[meta_cols], feats], axis=1)
    out["y"] = df["a_won"].astype(int)
    out.attrs["ratings"] = [
        {"discipline": disc, "player": player, "tier": "global", "elo": rating}
        for (disc, player), rating in elo.items()
    ] + [
        {"discipline": disc, "player": player, "tier": tier, "elo": rating}
        for ((disc, player), tier), rating in tier_elo.items()
    ]
    return out
