import pandas as pd

from bwf.features import build_features, expected_score, infer_tier, randomize_sides
from bwf.models import temporal_split


def toy():
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="7D"),
            "tournament": [
                "All England Open",
                "Swiss Open",
                "Malaysia Open",
                "German Open",
                "China Open",
                "Taipei Open",
            ],
            "discipline": ["MS", "WS", "MS", "WS", "MS", "WS"],
            "player_a": list("ABABAB"),
            "player_b": list("BABABA"),
            "a_won": [1, 1, 0, 1, 1, 0],
        }
    )


def test_expected_score_symmetry():
    assert abs(expected_score(1600, 1500) + expected_score(1500, 1600) - 1) < 1e-12


def test_first_match_has_neutral_features():
    f = build_features(toy()).iloc[0]
    assert f["elo_diff"] == 0 and f["tier_elo_diff"] == 0 and f["form_diff"] == 0 and f["h2h_n"] == 0


def test_no_leakage_label_flip_does_not_change_own_features():
    """Đổi kết quả của trận i không được làm đổi đặc trưng của trận i (chỉ ảnh hưởng trận sau)."""
    base = toy()
    meta = {"y", "date", "player_a", "player_b", "discipline", "tournament", "tour_tier", "round"}
    for i in range(len(base)):
        flipped = base.copy()
        flipped.loc[i, "a_won"] = 1 - flipped.loc[i, "a_won"]
        f0, f1 = build_features(base, dynamic_k=True), build_features(flipped, dynamic_k=True)
        cols = [c for c in f0.columns if c not in meta]
        assert f0.loc[i, cols].equals(f1.loc[i, cols])
        assert f0.loc[: i - 1, cols].equals(f1.loc[: i - 1, cols])


def test_temporal_split_order():
    f = build_features(toy())
    tr, va, te = temporal_split(f, (0.5, 0.25, 0.25))
    assert tr["date"].max() <= va["date"].min() <= te["date"].min()


def test_randomize_sides_keeps_winner():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 200,
            "player_a": ["W"] * 200,
            "player_b": ["L"] * 200,
            "a_won": [1] * 200,
        }
    )
    out = randomize_sides(df)
    winners = out.apply(lambda r: r.player_a if r.a_won else r.player_b, axis=1)
    assert (winners == "W").all() and 0.3 < out["a_won"].mean() < 0.7


def test_infer_tier_detection():
    assert infer_tier("Yonex All England Open Badminton Championships") == "Super 1000"
    assert infer_tier("Yonex Swiss Open 2024") == "Super 300"
    assert infer_tier("Paris Olympics 2024") == "Olympics"


def test_dynamic_k_changes_elo_magnitude():
    matches = toy().assign(discipline="MS")
    f_static = build_features(matches, k=32.0, dynamic_k=False)
    f_dyn = build_features(matches, k=32.0, dynamic_k=True)
    # Trận 1 là All England (Super 1000 -> hệ số nhân 1.25), nên ở trận 2 elo_diff của dynamic_k phải lớn hơn
    assert abs(f_dyn.iloc[1]["elo_diff"]) > abs(f_static.iloc[1]["elo_diff"])
