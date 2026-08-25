# TODO

Things that are known-incomplete, blocked on an external party, or need a
manual step. Anything here is deliberate; nothing here is a silent gap.

## Data sources

- **No open rugby event data with coordinates exists.** Checked before writing
  the ingest layer:
  - `rugbypy` publishes team and player summary aggregates only (59 and 54
    columns of match totals). No events, no `x`/`y`, no phase counts.
  - ESPN's rugby play feed
    (`sports.core.api.espn.com/v2/sports/rugby/leagues/{league}/events/{id}/competitions/{id}/plays`)
    returns scoring and personnel events only. Italy v England, Six Nations
    2024 (event `597378`) yields 49 plays: 16 substitutions, 16 substitute-on,
    6 penalty goals, 5 tries, 4 conversions, 2 yellow cards. No coordinates, no
    phase counts. The `commentary` endpoint 404s for rugby.
  - GitHub has no maintained rugby event-data repository with phase-level
    events; the search hits are image-annotation datasets and Qlik dashboards.
  Consequence: the ESPN mapper is real and tested but produces a scoring
  timeline, not a phase-play corpus. The generic analyst CSV is the only route
  to real phase data, which is why it is a first-class ingest path.

- **ESPN rejects a descriptive User-Agent.** Requests carrying
  `ruck/0.1 (...)` get an Akamai `Access Denied`; the same request with a
  browser-style UA succeeds. `ruck.ingest.http` therefore sends a
  browser-style UA with the project name and contact appended, and the UA is
  overridable with `RUCK_USER_AGENT`. Rate limiting (1 req/sec, backoff,
  on-disk cache) is unaffected.

- **The demo LM corpus is synthetic.** Because of the above there is no real
  phase-play corpus to train on. `ruck demo` trains on a seeded generator
  (`ruck.demo_data.generate`) and every artefact it touches — PDF cover, CSV
  header, README metrics table — says so. Replace it with your own exports and
  the same pipeline produces meaningful numbers.

## Audit

- The spec calls the local Hindsight CLI as `hindsight run <repo>`; the
  installed CLI's subcommand is `hindsight audit <repo>`. The wrapper probes
  `audit` first, then `run`, then falls back to native checks. Remove the
  probe once the CLI settles.
