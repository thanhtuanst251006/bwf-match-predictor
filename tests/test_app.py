import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def report_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BWF_OUTPUT_DIR", str(tmp_path))
    return tmp_path


def test_missing_reports_show_instructions(report_root):
    app = AppTest.from_file(APP, default_timeout=15).run()
    assert not app.exception
    assert app.warning
    assert "run_experiment.py" in app.code[0].value


def test_simulator_filters_discipline_and_uses_tier_ratings(report_root):
    reports = report_root / "reports"
    reports.mkdir()
    processed = report_root / "data/processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"log_loss": [0.5], "brier": [0.2], "accuracy": [0.7], "auc": [0.8]}, index=["Elo"]).to_csv(
        reports / "results.csv"
    )
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "discipline": ["MS", "WS"],
            "player_a": ["A", "C"],
            "player_b": ["B", "D"],
            "y": [1, 0],
        }
    ).to_csv(processed / "features_matches.csv", index=False)
    rows = []
    for discipline, players in [("MS", ["A", "B"]), ("WS", ["C", "D"])]:
        for player, rating in zip(players, [1700, 1300]):
            rows.extend(
                [
                    {"discipline": discipline, "player": player, "tier": "global", "elo": rating},
                    {"discipline": discipline, "player": player, "tier": "Super 300", "elo": 3000 - rating},
                ]
            )
    pd.DataFrame(rows).to_csv(reports / "ratings.csv", index=False)
    (reports / "metadata.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "k": 16,
                "dynamic_k": False,
                "selected_ml_model": "logreg",
                "splits": {"test": {"n": 2}},
            }
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(APP, default_timeout=15).run()
    assert not app.exception
    assert app.warning
    assert app.selectbox(key="player_a_MS").options == ["A", "B"]
    assert app.selectbox(key="player_b_MS").options == ["B"]
    initial = app.success[0].value
    app.selectbox[4].select("Elo + Tier Elo (70/30)").run()
    assert not app.exception
    assert app.success[0].value != initial
    app.selectbox(key="discipline").select("WS").run()
    assert not app.exception
    assert app.selectbox(key="player_a_WS").options == ["C", "D"]
