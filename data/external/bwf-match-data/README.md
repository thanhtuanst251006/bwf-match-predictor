# BWF Badminton Match Results (2007–2026)

Every elite-tier badminton match sanctioned by the Badminton World Federation
from January 2007 to July 2026, compiled into a single clean CSV (plus
per-discipline splits).

**105,147 matches · 608 tournaments · 5 disciplines · ~19.5 years**

| | |
|---|---|
| Coverage | 2007-01-16 → 2026-07-05 |
| Disciplines | Men's singles, women's singles, men's doubles, women's doubles, mixed doubles |
| Competition tiers | World Superseries, Grand Prix, HSBC BWF World Tour, World Championships, Olympic Games |
| Format | CSV, one row per match |
| License | See [Licensing & attribution](#licensing--attribution) |

## Contents

| File | Description | Size |
|---|---|---:|
| `matches.csv` | All 105,147 matches, every discipline, sorted by date | 17.5 MB |
| `ms.csv` | Men's singles only | 4.2 MB |
| `ws.csv` | Women's singles only | 3.2 MB |
| `md.csv` | Men's doubles only | 3.6 MB |
| `wd.csv` | Women's doubles only | 3.0 MB |
| `xd.csv` | Mixed doubles only | 3.6 MB |
| `index.json` | Tournament index: id, name, dates, location, tier for every tournament since 2007 | 5.0 MB |

## Discipline breakdown

| Discipline | Matches |
|---|---:|
| Men's singles | 28,270 |
| Women's singles | 21,372 |
| Men's doubles | 19,693 |
| Women's doubles | 16,376 |
| Mixed doubles | 19,436 |

## Schema

One row per completed match. UTF-8, comma-delimited, header row included.

| Column | Type | Description |
|---|---|---|
| `date` | date | Match date, `YYYY-MM-DD` |
| `discipline` | string | `MS`, `WS`, `MD`, `WD`, or `XD` |
| `tournament` | string | Tournament name |
| `tier` | string | Competition category (e.g. `HSBC BWF World Tour Super 750`, `World Superseries`, `Grade 1 – Individual Tournaments` for Worlds/Olympics) |
| `round` | string | Round name; a leading `Q` marks qualification rounds |
| `host_location` | string | Tournament host country |
| `team1`, `team2` | string | Player name(s); doubles partners joined with ` / ` |
| `winner` | int | `1` or `2` — which side won |
| `score` | string | Game scores in order, e.g. `21-15 17-21 21-16` (`team1`'s points first) |
| `team1_at_home`, `team2_at_home` | bool | `True` when that side's player is competing in their own country |

### Tournament index (`index.json`)

A JSON array, one object per tournament since 2007 (4,029 entries — a
superset of the 608 elite tournaments behind the match files, including lower
grades not otherwise included here). Key fields: `id`, `name`, `start_date`,
`end_date`, `location`, `country`, `category` (maps to `tier` above), and
`url`. Use it to look up tournament metadata or to re-scope the dataset to a
different tier cutoff.

## Competition tiers

| Tier | Matches | Era |
|---|---:|---|
| Grand Prix Gold | 20,440 | 2007–2017 |
| World Superseries | 19,253 | 2007–2017 |
| HSBC BWF World Tour Super 300 | 12,213 | 2018– |
| Grand Prix | 11,673 | 2007–2017 |
| BWF Tour Super 100 | 11,476 | 2018– |
| HSBC BWF World Tour Super 500 | 9,166 | 2018– |
| World Superseries Premier | 7,541 | 2007–2017 |
| HSBC BWF World Tour Super 750 | 5,613 | 2018– |
| HSBC BWF World Tour Super 1000 | 4,420 | 2018– |
| Grade 1 – Individual Tournaments | 2,765 | World Championships & Olympic Games, all years |
| HSBC BWF World Tour Finals | 587 | 2018– |

The tour was restructured in 2018, replacing the Superseries/Grand Prix
tier system with the HSBC BWF World Tour; both eras are represented here
under their original tier labels rather than normalized to one scheme.

## Data collection

Compiled from BWF's own publicly served tournament results, tournament by
tournament, then cleaned and flattened into the tables above.

Processing notes:

- Duplicate match records (some tournaments publish results under more than
  one grouping) were de-duplicated on match id.
- Walkovers and matches with no recorded winner are excluded.
- Player names are normalized to a consistent spelling across the full time
  span (surname in caps, e.g. `SHI Yu Qi`). Where two players share a display
  name, one carries a `[player-id]` disambiguation suffix.
- Scores are the raw per-game point totals as recorded; matches ending in
  retirement or default are represented by whatever games were completed,
  with no separate annotation.
- Qualification rounds are included and prefixed with `Q` in `round` — filter
  them out if you want main-draw results only.

## Known limitations

- `round` naming isn't fully standardized across eras — most rounds use
  abbreviations (`R32`, `QF`, `SF`) but a small number of older or
  team-format entries use spelled-out or group-stage labels (`Round of 16`,
  `Group C`), and a handful are blank.
- Coverage begins in 2007; earlier results (e.g. Lin Dan's 2004–06 run)
  aren't included.
- This is singles/doubles individual-event data only — team competitions
  (Thomas Cup, Uber Cup, Sudirman Cup) aren't broken out into individual
  rubbers here.

## Quick start

```python
import pandas as pd

df = pd.read_csv("matches.csv", parse_dates=["date"])
ms = df[df.discipline == "MS"]

# main-draw matches only
main_draw = df[~df["round"].str.startswith("Q", na=False)]

# a player's career record
player = "SHI Yu Qi"
matches = df[(df.team1 == player) | (df.team2 == player)]
wins = ((matches.team1 == player) & (matches.winner == 1)) | \
       ((matches.team2 == player) & (matches.winner == 2))
print(f"{player}: {wins.sum()}-{(~wins).sum()}")
```

## Licensing & attribution

Match results and tournament data are the property of the Badminton World
Federation. This is a research/hobby redistribution of results BWF already
serves publicly to its own website; it is not an official BWF product and
carries no affiliation or endorsement. If you use it for something
interesting, I'd like to hear about it.
