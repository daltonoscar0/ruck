# ruck

Rugby phase-play analyzer and injury-model auditor. Two subcommands, two users.

`ruck analyze` takes a match event log and produces a PDF with a per-phase
surprisal timeline, a timestamped list of the plays a language model found most
unexpected, and an attack-shape summary against competition average. The
timestamp list also comes out as CSV so an analyst can drive video clipping
from it.

`ruck audit` points at a workload/injury-model repository and produces a
leakage report: lookahead in feature windows, injury dates inside the workload
window, survivorship, unpurged cross-validation, and the mathematical coupling
that makes acute:chronic workload ratios look predictive when they are not.
Every finding carries a `file:line` citation.

```sh
ruck analyze match.csv --team Munster --out report.pdf
ruck compare munster.csv leinster.csv --out overlay.pdf
ruck audit path/to/repo --out audit.md
ruck demo --out-dir runs/demo        # offline, no network
```

## Install

Python 3.11 or newer.

```sh
git clone <this repo> && cd ruck
make install     # uv venv + editable install with dev extras
make test
make demo        # writes runs/demo/report.pdf and runs/demo/audit.md
```

## Data

Rugby event data is the hard part of this project, and the situation is worth
stating plainly before the metrics section.

| Source | What it actually provides | Usable for phase play? |
| --- | --- | --- |
| Analyst CSV export | `t, team, action, x, y, phase, outcome` | Yes — this is the real path |
| ESPN rugby play feed | ~50 scoring/card/substitution events per match, no coordinates, no phase counts | Partially: timeline only |
| RugbyPy | team and player summary aggregates | No |

There is no open, free rugby dataset with phase-level events and pitch
coordinates. Every mapper here is real and tested against real payloads, but
the model's training corpus in `ruck demo` is **synthetic and labelled as
such** everywhere it appears. See [Metrics](#metrics) for what that does and
does not license you to conclude.

The generic CSV schema is documented in [`docs/csv_format.md`](docs/csv_format.md).
If you have an Opta, Stats Perform or SportsCode export, that is the format to
target, and the numbers below become meaningful on your data.

## Metrics

Filled in by `scripts/report_metrics.py` at release; see that section of this
file after running `make demo`.

## Licence

MIT. Vendored audit fixtures under `demo/audit_fixtures/` keep their own
upstream licences; see the `SOURCE.md` next to each.
