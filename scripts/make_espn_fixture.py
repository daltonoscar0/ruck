"""Build the offline ESPN test fixture from the live cache.

Run once after `ruck fetch espn ...` has populated `data/raw/`. Trims the
summary payload down to the parts the loader reads (header and rosters), keeps
the plays payload whole, and writes both into `tests/fixtures/espn_cache/`
under the exact filenames :class:`ruck.ingest.http.Fetcher` would look for, so
the test suite exercises the real cache path with real payload shapes and no
network.

    python scripts/make_espn_fixture.py 597378 six_nations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ruck.ingest.espn import CORE_API, SITE_API, resolve_league
from ruck.ingest.http import Fetcher

FIXTURE_DIR = Path("tests/fixtures/espn_cache")


def trim_summary(summary: dict) -> dict:
    """Keep only what :func:`ruck.ingest.espn._match_context` reads."""
    header = summary.get("header", {})
    competition = (header.get("competitions") or [{}])[0]
    return {
        "header": {
            "id": header.get("id"),
            "season": header.get("season"),
            "league": {
                "id": (header.get("league") or {}).get("id"),
                "name": (header.get("league") or {}).get("name"),
            },
            "competitions": [
                {
                    "id": competition.get("id"),
                    "date": competition.get("date"),
                    "competitors": [
                        {
                            "id": c.get("id"),
                            "homeAway": c.get("homeAway"),
                            "score": c.get("score"),
                            "team": {
                                "id": (c.get("team") or {}).get("id"),
                                "displayName": (c.get("team") or {}).get("displayName"),
                                "abbreviation": (c.get("team") or {}).get("abbreviation"),
                            },
                        }
                        for c in competition.get("competitors", [])
                    ],
                }
            ],
        },
        "rosters": [
            {
                "team": {"id": (r.get("team") or {}).get("id")},
                "roster": [
                    {
                        "jersey": entry.get("jersey"),
                        "athlete": {
                            "id": (entry.get("athlete") or {}).get("id"),
                            "displayName": (entry.get("athlete") or {}).get("displayName"),
                        },
                    }
                    for entry in r.get("roster", [])
                ],
            }
            for r in summary.get("rosters", [])
        ],
    }


def main(event_id: str, league: str = "six_nations") -> int:
    league_id = resolve_league(league)
    live = Fetcher(cache_dir="data/raw")
    fixture = Fetcher(cache_dir=FIXTURE_DIR)

    summary_url = f"{SITE_API}/{league_id}/summary"
    summary_params = {"event": event_id}
    plays_url = f"{CORE_API}/{league_id}/events/{event_id}/competitions/{event_id}/plays"
    plays_params = {"limit": 1000, "page": 1}

    for url, params, transform in (
        (summary_url, summary_params, trim_summary),
        (plays_url, plays_params, lambda payload: payload),
    ):
        source = live.cache_path(url, params)
        if not source.exists():
            print(f"missing from live cache: {source}", file=sys.stderr)
            return 1
        payload = transform(json.loads(source.read_text()))
        target = fixture.cache_path(url, params)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        print(f"{target}  {target.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
