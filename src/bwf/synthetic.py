"""Dữ liệu GIẢ LẬP chỉ để kiểm thử pipeline (không dùng để báo cáo kết quả)."""

from __future__ import annotations

import numpy as np
import pandas as pd

MS_PLAYERS = [
    "Viktor AXELSEN",
    "SHI Yu Qi",
    "Kunlavut VITIDSARN",
    "Anders ANTONSEN",
    "LI Shi Feng",
    "Jonatan CHRISTIE",
    "Kodai NARAOKA",
    "Anthony Sinisuka GINTING",
    "LEE Zii Jia",
    "LOH Kean Yew",
    "CHOU Tien Chen",
    "Lakshya SEN",
    "PRANNOY H. S.",
    "Kenta NISHIMOTO",
    "WENG Hong Yang",
    "LU Guang Zu",
    "LIN Chun-Yi",
    "Toma Junior POPOV",
    "Christo POPOV",
    "NG Tze Yong",
    "Brian YANG",
    "Koki WATANABE",
    "LEE Cheuk Yiu",
    "ANGUS Ng Ka Long",
    "LEONG Jun Hao",
    "Rasmus GEMKE",
    "Kento MOMOTA",
    "KIDAMBI Srikanth",
    "Su Li Yang",
    "Yushi TANAKA",
]

WS_PLAYERS = [
    "AN Se Young",
    "CHEN Yu Fei",
    "TAI Tzu Ying",
    "Akane YAMAGUCHI",
    "Carolina MARIN",
    "HE Bing Jiao",
    "Gregoria Mariska TUNJUNG",
    "WANG Zhi Yi",
    "HAN Yue",
    "Beiwen ZHANG",
    "PUSARLA V. Sindhu",
    "Ratchanok INTANON",
    "Supanida KATETHONG",
    "Aya OHORI",
    "Tomoka MIYAZAKI",
    "Pornpawee CHOCHUWONG",
    "Busanan ONGBAMRUNGPHAN",
    "Nozomi OKUHARA",
    "Mia BLICHFELDT",
    "Line Hojmark KJAERSFELDT",
    "Michelle LI",
    "YEO Jia Min",
    "Natsuki NIDAIRA",
    "Kirsty GILMOUR",
    "SIM Yu Jin",
    "KIM Ga Eun",
    "PAI Yu Po",
    "Hsu Wen Chi",
    "Putri Kusuma WARDANI",
    "Ester Nurumi Tri WARDOYO",
]

TOURNAMENTS = [
    ("All England Open", "Super 1000"),
    ("Indonesia Open", "Super 1000"),
    ("China Open", "Super 1000"),
    ("Malaysia Open", "Super 1000"),
    ("Denmark Open", "Super 750"),
    ("Japan Open", "Super 750"),
    ("French Open", "Super 750"),
    ("Singapore Open", "Super 750"),
    ("India Open", "Super 750"),
    ("China Masters", "Super 750"),
    ("Malaysia Masters", "Super 500"),
    ("Indonesia Masters", "Super 500"),
    ("Thailand Open", "Super 500"),
    ("Korea Open", "Super 500"),
    ("Hong Kong Open", "Super 500"),
    ("Arctic Open", "Super 500"),
    ("Kumamoto Masters Japan", "Super 500"),
    ("Swiss Open", "Super 300"),
    ("German Open", "Super 300"),
    ("Taipei Open", "Super 300"),
]


def make_synthetic_matches(n_players: int = 60, n_matches: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Sinh dữ liệu giả lập mô phỏng giải đấu BWF World Tour (MS & WS)."""
    if n_players < 4 or n_matches < 1:
        raise ValueError("Cần ít nhất 4 vận động viên và 1 trận đấu.")
    rng = np.random.default_rng(seed)
    half = n_players // 2
    ms_pool = [MS_PLAYERS[i % len(MS_PLAYERS)] if n_players <= 60 else f"MS_P{i:02d}" for i in range(half)]
    ws_pool = [
        WS_PLAYERS[i % len(WS_PLAYERS)] if n_players <= 60 else f"WS_P{i:02d}"
        for i in range(n_players - half)
    ]

    skill_ms = rng.normal(0, 1.05, len(ms_pool))
    skill_ws = rng.normal(0, 1.25, len(ws_pool))
    # Điểm xếp hạng BWF tích lũy 52 tuần có độ trễ và nhiễu so với thực lực tức thời
    rank_pts_ms = skill_ms.copy() + rng.normal(0, 0.45, len(ms_pool))
    rank_pts_ws = skill_ws.copy() + rng.normal(0, 0.45, len(ws_pool))

    dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 365 * 6, n_matches)), unit="D"
    )
    rows = []
    for d in dates:
        is_ms = rng.random() < 0.52
        pool = ms_pool if is_ms else ws_pool
        skill = skill_ms if is_ms else skill_ws
        r_pts = rank_pts_ms if is_ms else rank_pts_ws
        disc = "MS" if is_ms else "WS"

        a, b = rng.choice(len(pool), 2, replace=False)
        skill += rng.normal(0, 0.0035, len(pool))
        r_pts = 0.98 * r_pts + 0.02 * skill + rng.normal(0, 0.03, len(pool))
        t_name, t_tier = TOURNAMENTS[rng.integers(0, len(TOURNAMENTS))]

        scale = 1.35 if t_tier in ("Super 1000", "Super 750") else 1.10
        p = 1.0 / (1.0 + np.exp(-(skill[a] - skill[b]) * scale))
        rank = (-r_pts).argsort().argsort() + 1

        rows.append(
            {
                "date": d,
                "tournament": t_name,
                "tour_tier": t_tier,
                "discipline": disc,
                "round": rng.choice(["R32", "R16", "QF", "SF", "F"], p=[0.45, 0.25, 0.15, 0.10, 0.05]),
                "player_a": pool[a],
                "player_b": pool[b],
                "a_won": int(rng.random() < p),
                "rank_a": int(rank[a]),
                "rank_b": int(rank[b]),
            }
        )
    return pd.DataFrame(rows)
