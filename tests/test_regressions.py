import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bwf.features import FEATURES, build_features
from bwf.models import metrics, temporal_split
from bwf.synthetic import make_synthetic_matches

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare", ROOT / "scripts/prepare_kaggle_data.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def matches():
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6),
            "player_a": ["A"] * 6,
            "player_b": ["B"] * 6,
            "a_won": [1, 0, 1, 0, 1, 0],
            "discipline": ["MS"] * 6,
        }
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("date", "bad"),
        ("date", None),
        ("player_a", None),
        ("player_a", " "),
        ("player_a", "B"),
        ("a_won", 2),
        ("a_won", 0.5),
        ("discipline", "MD"),
        ("rank_a", -1),
        ("rank_a", np.inf),
    ],
)
def test_invalid_matches_fail_with_clear_error(column, value):
    frame = matches().astype({"date": "object"})
    frame[column] = value
    with pytest.raises(ValueError):
        build_features(frame)


def test_discipline_state_is_independent():
    frame = matches()
    frame.loc[1, "discipline"] = "WS"
    features = build_features(frame)
    assert (features.loc[1, FEATURES] == 0).all()
    assert features.loc[2, "h2h_n"] == 1


def test_exported_ratings_match_last_update():
    result = build_features(matches(), k=24, dynamic_k=True)
    ratings = pd.DataFrame(result.attrs["ratings"])
    global_ratings = ratings[ratings["tier"] == "global"].set_index("player")["elo"]
    last_diff = result.iloc[-1]["elo_diff"]
    expected = 1 / (1 + 10 ** (-last_diff / 400))
    assert global_ratings["A"] - global_ratings["B"] == pytest.approx(last_diff + 48 * (0 - expected))
    assert global_ratings.sum() == pytest.approx(3000)


def test_temporal_split_keeps_dates_together():
    frame = pd.DataFrame({"date": np.repeat(pd.date_range("2024-01-01", periods=7), 3)})
    train, val, test = temporal_split(frame)
    assert train.date.max() < val.date.min()
    assert val.date.max() < test.date.min()
    assert len(train) + len(val) + len(test) == len(frame)


@pytest.mark.parametrize("fractions", [(0.6, 0.3, 0.3), (0, 0.5, 0.5), (1,), (np.nan, 0.2, 0.2)])
def test_temporal_split_rejects_invalid_fractions(fractions):
    with pytest.raises(ValueError):
        temporal_split(matches(), fractions)


def test_temporal_split_rejects_insufficient_dates():
    with pytest.raises(ValueError):
        temporal_split(matches().assign(date="2024-01-01"))


def test_metrics_handles_single_class():
    result = metrics([1, 1], [0.6, 0.7])
    assert np.isfinite(result["log_loss"])
    assert np.isnan(result["auc"])


def test_csv_tiers_remain_aligned_after_filtering():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 3,
            "winner": ["A", "C", "E"],
            "loser": ["B", "D", "F"],
            "discipline": ["MS", "MD", "WS"],
            "tour_tier": ["Super 1000", "Super 750", "Super 300"],
        },
        index=[10, 20, 30],
    )
    result = prepare.convert_kaggle_dataframe(raw)
    assert result["tour_tier"].tolist() == ["Super 1000", "Super 300"]


def test_csv_missing_name_is_dropped():
    raw = matches()
    raw.loc[0, "player_a"] = None
    result = prepare.convert_kaggle_dataframe(raw)
    assert len(result) == 5
    assert "nan" not in result.player_a.tolist()


def test_csv_rejects_unknown_winner_instead_of_assigning_loss():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "team_one_players": ["A"],
            "team_two_players": ["B"],
            "winning_team": ["unknown"],
        }
    )
    with pytest.raises(ValueError, match="winner"):
        prepare.convert_kaggle_dataframe(raw)


def test_small_synthetic_pool_works_and_is_reproducible():
    first = make_synthetic_matches(n_players=4, n_matches=30, seed=7)
    assert first.equals(make_synthetic_matches(n_players=4, n_matches=30, seed=7))
    assert len(set(first.player_a) | set(first.player_b)) == 4
