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

- **The demo LM corpus is synthetic.** Because of the above there is no real
  phase-play corpus to train on. `ruck demo` trains on a seeded generator
  (`ruck.demo_data.generate`) and every artefact it touches — PDF cover banner,
  competition name, README metrics table — says so. Replace it with your own
  exports (`ruck train`) and the same pipeline produces meaningful numbers.

  The generator is tuned to plausible per-match rates (about 265 carries, 260
  passes, 210 rucks, 48 kicks, 6 tries) but it is not a rugby simulator, and
  nothing about the *model's* accuracy on it transfers to real matches.

- **ESPN's edge is picky about User-Agent.** Measured against
  `site.api.espn.com`: a bare product token (`ruck/0.1`) returns 403, and so
  does a browser impersonation string; a product token with a contact comment
  (`ruck/0.1 (+https://github.com/daltonoscar0/ruck)`) returns 200 reliably.
  That is the default in `ruck.ingest.http`, overridable with
  `RUCK_USER_AGENT`. No browser impersonation is used anywhere.

- **SoccerNet, StatsBomb and the rest of the sportio sources are not wired up
  here.** `ruck` reads rugby; the shared loaders belong in `sportio`, which
  does not exist yet. `ruck.schema.load_sportio_events` hands off to it if it
  ever becomes importable.

## Model

- The transformer is trained on possessions, so the first event of each
  possession is conditioned on `<bos>` alone. Its surprisal measures a
  corpus-wide prior over set pieces rather than anything about the match, so it
  is left unscored. Giving the model cross-possession context — encoding a
  match as one stream with possession boundaries as tokens — would recover
  those events and is the obvious next experiment.
- `ruck compare` scores both matches with the same model but does not test
  whether the difference between two surprisal curves is larger than run-to-run
  noise. Treat the overlay as descriptive.

## Audit

- The spec calls the local Hindsight CLI as `hindsight run <repo>`; the
  installed CLI's subcommand is `hindsight audit <repo>`. The wrapper probes
  `audit` first, then `run`, then falls back to native checks. Remove the probe
  once the CLI settles.
- The checks read Python only. A leak expressed in SQL, in a warehouse view, or
  in how the injury records were entered is outside what a static reader can
  see; the report says so in its closing section.
- Notebooks are not parsed. `hindsight` handles `.ipynb`; ruck does not, so
  point it at the exported `.py` or run both.
- `CV004` (no athlete grouping) is a whole-file heuristic and fires at low
  confidence. It is deliberately noisy-but-cheap; suppress it with `--only` if
  it does not apply to your layout.
