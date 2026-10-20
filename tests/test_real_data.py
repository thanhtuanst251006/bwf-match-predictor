import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from bwf.features import FEATURES, build_features

spec = importlib.util.spec_from_file_location(
    "fetch_real", Path(__file__).resolve().parents[1] / "scripts/fetch_real_data.py"
)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


@pytest.mark.parametrize(
    "score,winner,valid",
    [
        ("21-15 24-22", 1, True),
        ("15-21 22-24", 2, True),
        ("21-19 19-21 30-29", 1, True),
        ("21-19 19-21 30-28", 1, True),
        ("21-19 19-21 30-27", 1, False),
        ("21-19 21-20", 1, False),
        ("21-19 21-19", 2, False),
        ("21-19 21-19 21-19", 1, False),
        ("21-19 3-8", 2, False),
        ("21-19", 1, False),
        ("w/o", 1, False),
        ("21-19 21-19", 0, False),
    ],
)
def test_completed_match_score_filter(score, winner, valid):
    assert fetch.completed_score(score, winner) is valid


def test_real_import_preserves_quarterfinals_and_filters_qualifying():
    raw = pd.DataFrame(
        {
            "date": ["2026-01-01"] * 4,
            "discipline": ["MS"] * 4,
            "tournament": ["Example Open"] * 4,
            "tier": ["HSBC BWF World Tour Super 300"] * 4,
            "round": ["QF", "Q Qual. R32", "SF", "Final"],
            "team1": ["A", "B", "C", "D"],
            "team2": ["E", "F", "G", "H"],
            "winner": [1, 1, 1, 2],
            "score": ["21-15 21-17", "21-10 21-12", "21-10 1-0", "17-21 19-21"],
        }
    )
    clean, stats = fetch.prepare_real_matches(raw)
    assert clean["round"].tolist() == ["QF", "Final"]
    assert clean["tour_tier"].tolist() == ["Super 300"] * 2
    assert clean["a_won"].tolist() == [1, 0]
    assert stats["excluded_qualifying"] == 1
    assert stats["excluded_incomplete_or_invalid_score"] == 1


def test_same_day_results_cannot_change_other_same_day_features():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "player_a": ["A"] * 3,
            "player_b": ["B"] * 3,
            "a_won": [1, 1, 1],
        }
    )
    original = build_features(raw)
    raw.loc[0, "a_won"] = 0
    flipped = build_features(raw)
    pd.testing.assert_frame_equal(original.loc[:1, FEATURES], flipped.loc[:1, FEATURES])
    assert original.loc[1, "h2h_n"] == 0
    assert original.loc[2, "h2h_n"] == 2
    assert original.loc[2, "elo_diff"] != flipped.loc[2, "elo_diff"]
