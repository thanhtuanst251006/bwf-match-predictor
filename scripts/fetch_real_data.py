"""Download a pinned public BWF-results archive and prepare completed World Tour singles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bwf.validation import validate_matches  # noqa: E402

REPOSITORY = "https://github.com/SahilMotyar/bwf-match-data"
COMMIT = "0a6899162bf4b9639b76dcdbc0dca4c5d4800664"
CHECKSUMS = {
    "ms.csv": "983d6127026819fa89cb463da7ac45b88d153199fdbff2533c19b5cbc3122000",
    "ws.csv": "e29f01b5e8f21c645f8377e28441c6135c1fb7dfb094fe281f1953ab9bd6a9cf",
    "README.md": "1cf170bb7c26009aee5b66e79a5f22230e0a4134a2d969de7e1da203a1d9ee3e",
}


def completed_score(score: str, winner: int) -> bool:
    """Require a complete best-of-three match under the 21-point scoring system."""
    if not isinstance(score, str) or not re.fullmatch(r"\d+-\d+(?:\s+\d+-\d+){1,2}", score.strip()):
        return False
    wins = [0, 0]
    for game in score.split():
        if max(wins) == 2:
            return False
        a, b = map(int, game.split("-"))
        high, low = max(a, b), min(a, b)
        valid = (high == 21 and low <= 19) or (22 <= high <= 29 and high - low == 2)
        valid = valid or (high == 30 and low in (28, 29))
        if not valid:
            return False
        wins[int(b > a)] += 1
    return winner in (1, 2) and wins[int(winner) - 1] == 2 and max(wins) == 2


def prepare_real_matches(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"date", "discipline", "tournament", "tier", "round", "team1", "team2", "winner", "score"}
    if required - set(raw):
        raise ValueError(f"Missing archive columns: {sorted(required - set(raw))}")
    dates = pd.to_datetime(raw["date"], errors="raise")
    # Retain the explicit modern tiers; do not guess historical tournament levels from names.
    tier_map = {f"HSBC BWF World Tour Super {level}": f"Super {level}" for level in (300, 500, 750, 1000)}
    tier_map["BWF Tour Super 100"] = "Super 100"
    tier_map["HSBC BWF World Tour Finals"] = "World Tour Finals"
    scope = raw[
        raw["discipline"].isin(["MS", "WS"]) & (dates >= "2018-01-01") & raw["tier"].isin(tier_map)
    ].copy()
    # QF means quarter-final, not qualification.
    qualifying = scope["round"].fillna("").str.match(r"(?i)^Q(?:\s|ual)")
    main = scope[~qualifying].copy()
    complete = main.apply(lambda row: completed_score(row["score"], row["winner"]), axis=1)
    clean = main[complete].copy()
    duplicate_columns = ["date", "discipline", "tournament", "round", "team1", "team2", "winner", "score"]
    before_dedup = len(clean)
    clean = clean.drop_duplicates(subset=duplicate_columns)
    out = clean.rename(columns={"team1": "player_a", "team2": "player_b"})
    out["a_won"] = (out["winner"] == 1).astype(int)
    out["tour_tier"] = out["tier"].map(tier_map)
    out = validate_matches(
        out[
            [
                "date",
                "discipline",
                "tournament",
                "tour_tier",
                "round",
                "player_a",
                "player_b",
                "a_won",
                "score",
            ]
        ]
    )
    stats = {
        "downloaded_singles": len(raw),
        "modern_world_tour": len(scope),
        "excluded_qualifying": int(qualifying.sum()),
        "excluded_incomplete_or_invalid_score": int((~complete).sum()),
        "excluded_exact_duplicates": before_dedup - len(clean),
        "retained": len(out),
        "by_discipline": out["discipline"].value_counts().to_dict(),
        "date_start": str(out["date"].min().date()),
        "date_end": str(out["date"].max().date()),
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use the existing, hash-verified archive")
    parser.add_argument("--output", type=Path, default=Path("data/raw/bwf_real_matches.csv"))
    parser.add_argument("--archive-dir", type=Path, default=Path("data/external/bwf-match-data"))
    args = parser.parse_args()
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for filename, expected_hash in CHECKSUMS.items():
        url = f"https://raw.githubusercontent.com/SahilMotyar/bwf-match-data/{COMMIT}/{filename}"
        path = args.archive_dir / filename
        if not args.offline:
            request = Request(url, headers={"User-Agent": "bwf-match-predictor/0.2"})
            with urlopen(request, timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ValueError(f"Checksum mismatch: {filename}")
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Archive checksum mismatch: {filename}")
        sources.append({"url": url, "sha256": expected_hash})
    raw = pd.concat(
        [pd.read_csv(args.archive_dir / name) for name in ("ms.csv", "ws.csv")], ignore_index=True
    )
    clean, stats = prepare_real_matches(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    metadata = {
        "synthetic": False,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "repository": REPOSITORY,
        "commit": COMMIT,
        "sources": sources,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "statistics": stats,
        "source_type": "community archive of public BWF results; not an official BWF release",
        "scope": "2018+, World Tour Super 100-1000 and Finals, main draws, completed 21-point matches",
        "limitations": [
            "No historical ranking snapshots",
            "Dates only; no match start times",
            "Coverage is limited to tournaments present in the source archive",
        ],
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
