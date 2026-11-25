# ruck

[![tests](https://github.com/daltonoscar0/ruck/actions/workflows/tests.yml/badge.svg)](https://github.com/daltonoscar0/ruck/actions/workflows/tests.yml)

Rugby phase-play analyzer and injury-model auditor. Two subcommands, two users.

**`ruck analyze`** takes a match event log and produces a PDF: a per-phase
surprisal timeline, a timestamped list of the plays a language model found most
unexpected, and an attack-shape summary against a competition average. The
timestamp list also comes out as CSV, because the analyst's next step is cutting
video from it.

**`ruck audit`** points at a workload or injury-model repository and produces a
leakage report: lookahead in feature windows, injury dates inside the workload
window, survivorship, unpurged cross-validation, and the mathematical coupling
that makes acute:chronic workload ratios look predictive when they are not.
Every finding carries a `file:line`, and the report ends with a purge list
written for GPS data.

```sh
ruck analyze match.csv --team Munster --out report.pdf --csv clips.csv
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

`make demo` runs offline on bundled data. Nothing in this repository needs the
network unless you ask it to fetch.

## Data

Rugby event data is the hard part of this project, and the situation is worth
stating before the metrics.

| Source | What it actually provides | Phase play? |
| --- | --- | --- |
| Analyst CSV export | `t, team, action, x, y, phase, outcome` | **Yes**, this is the real path |
| ESPN rugby play feed | ~50 scoring, card and substitution events per match with a match clock; no coordinates, no phase counts | Timeline only |
| RugbyPy | team and player summary aggregates (59 and 54 columns of match totals) | No |

There is no open dataset with phase-level rugby events and pitch coordinates.
All three loaders here are real and tested against real payloads, the ESPN one
against a cached Italy v England, Six Nations 2024, which is bundled and shows
up as an appendix in the demo PDF, but a phase-play corpus has to come from
somewhere else.

So the model that ships with `ruck demo` is trained on a **seeded synthetic
corpus** (`ruck.demo_data.generate`), and every artefact that touches it says so:
the PDF cover carries a banner, the competition is named "Synthetic Phase League
(generated)", and the metrics below are labelled. The numbers measure the
pipeline, not rugby.

Point it at your own Opta, Stats Perform or SportsCode export and the numbers
start meaning something. The format is documented in
[`docs/csv_format.md`](docs/csv_format.md); `ruck train` fits a model on it.

## Metrics

<!-- metrics:start -->

Measured on a **synthetic** corpus (see [Data](#data)): 80 generated matches, split by match into 64 train / 8 validation / 8 test. These numbers measure the pipeline, not rugby. Reproduce with `python scripts/train_demo_model.py`.

### Next-event perplexity

| Model | Test perplexity |
| --- | ---: |
| Causal transformer (1,080,320 params) | **3.74** |
| Add-k smoothed bigram | 3.94 |
| Unigram | 478.36 |

Over 9,584 test tokens, vocabulary 1986. The count baselines choose their smoothing constant on the same validation split the transformer early-stops against (bigram k = 0.0003); a fixed k would hand the transformer a win it has not earned.

**The transformer beats the bigram by only 1.05x.** Almost all the predictable structure in this corpus is in the previous event alone, which is what you would expect from a generator whose next step depends mostly on the last one, a real corpus may or may not behave the same way. On this evidence the transformer is not earning its million parameters, and the honest reading is that the bigram is the model to beat.

### Downstream: points within 3 phases

| Predictor | ROC AUC | Average precision | Brier |
| --- | ---: | ---: | ---: |
| GBM on hand features | 0.866 | 0.419 | 0.0722 |
| LM probe (frozen transformer) | 0.846 | 0.375 | 0.0739 |
| Base rate | 0.500 | 0.102 | 0.0914 |

8,455 in-play events, base rate 0.102. Both predictors are fitted on the same match-disjoint split, and both are strictly causal: the probe reads the frozen model's hidden state at that event, which is computed from that event and everything before it and nothing after. Events that *are* a score are excluded from the evaluated rows, predicting "will they score?" at the moment the points go up is not a prediction, and leaving those rows in inflated the reported average precision.

**The GBM on hand features wins.** A hand-built feature set, field position, phase count, what has happened in the possession so far, carries more of this signal than a linear read-out of the language model's state. The LM probe is behind it, and saying so is the point of running the comparison.

Trained in 726s, best validation perplexity 3.75 at epoch 33 of 38.

<!-- metrics:end -->

## How `analyze` works

**Tokenizer.** Each event becomes `action|zone|phase|outcome`, one of 25
actions, an 18-cell pitch zone (six bands from own 22 to the opposition
five-metre line, three channels split on the 15-metre lines), a phase-count
bucket, and an outcome. The full cross-product is about 40,000 combinations, far
past the 2,000-token budget, so the vocabulary is fitted to the corpus under a
cap and anything outside it backs off down a cascade:

```
carry|opp22:M|2|retained     full
carry|opp22:M|*|retained     drop the phase bucket
carry|*|*|retained           drop the zone
carry|*|*|*                  action only
```

Pure table lookup on a normalized string: no fuzzy matching, no learned merges,
and the same event always produces the same id.

**Model.** A small causal transformer over possessions, 4 layers, 128
dimensions, about 1.1M parameters against the spec's 10M ceiling, because a
season of tagged matches is tens of thousands of events and more capacity would
just memorise them. Splits are by match, never by possession.

**Surprisal.** `-log2 P(event | everything before it)`, in bits. The timeline
plots the mean per phase; the clip list takes the highest-scoring three-event
windows, thins them so ten clips are not ten cuts of the same try, and puts the
cut point eight seconds before each passage starts. Descriptions are generated
from the token's own fields through a fixed phrase table, the model produces a
number, never prose.

Surprisal is not quality. An unexpected event may be brilliant or a mistake; the
model has no view on that.

## How `audit` works

Static, read-only, and offline. Files are parsed with `ast` and never imported,
which matters when the target's top-level code trains a model on import.

If the [Hindsight](https://github.com/daltonoscar0/hindsight) CLI is on PATH it
is wrapped and its findings merged in, marked by source. Its subcommand is
`hindsight audit`; the brief for this tool says `hindsight run`, so the wrapper
probes both and falls back to the native checks.

| Check | What it looks for |
| --- | --- |
| `AC001` `AC002` | Coupled acute:chronic ratio, the acute window inside the chronic one, in pandas or in list slices |
| `AC003` | EWMA ACWR, where every session is in both terms |
| `AC004` | A ratio with no floor on the denominator |
| `AC005` | Sweet-spot binning at 0.8 / 1.3 / 1.5 |
| `AC006` | An acute window that includes the day the outcome is recorded |
| `LA001`-`LA007` | Negative shifts, centred windows, backfill, interpolation, negative diffs, transformers fitted before the split, windows ending on the row they describe |
| `LB001`-`LB005` | The outcome in the feature matrix, features derived from it, windows over it, inclusive injury-date filters, outcome-defined features |
| `SV001`-`SV003` | Population filtered by the outcome, unrestricted `dropna`, cohorts chosen on season totals |
| `CV001`-`CV005` | Shuffled splits, default K-fold, no athlete grouping, `TimeSeriesSplit` with no embargo |

### The ACWR section

The one worth reading first. An ACWR divides a recent window of load by a longer
one over the same series. In the standard formulation the acute window sits
inside the chronic window, so writing the chronic block as the acute week plus
the three before it:

```
ACWR = A / ((A + B) / 4) = 4A / (A + B)
```

That is a bounded, increasing function of acute load on its own. It goes up
after a hard week whether or not hard weeks cause injuries, and a model fitted
on it inherits the relationship from the algebra rather than from the athletes.

The report always prints this section, findings or none, because "we looked and
the ratio is uncoupled" is an answer a performance department needs in writing.

`demo/audit_fixtures/` holds snapshots of two public MIT-licensed repositories -
[`AbdullahKaddoura/acwr`](https://github.com/AbdullahKaddoura/acwr) and
[`mdana5/injury-prediction`](https://github.com/mdana5/injury-prediction), so
`ruck audit demo/audit_fixtures` runs on real third-party code offline. The
findings describe patterns in a code snapshot, not the competence of its
authors; the same patterns are in most published workload-modelling code, which
is rather the point.

## Sample output

`demo/expected/` holds checked-in output so you can read it without running
anything:

| File | What it is |
| --- | --- |
| `report.pdf` | The demo analysis, six pages, on the bundled synthetic match |
| `clips.csv` | The matching timestamp list |
| `audit.md` | ruck's report on `tests/fixtures/leaky_workload/pipeline.py`, a file with every checked leak in it on purpose |
| `fixtures_audit.md` | ruck's report on the two vendored real repositories |

Regenerate with `make expected`.

## Commands

```
ruck analyze TARGET  --team --out --csv --model --reference --top --window --device
ruck compare A B     --out --teams --model --device
ruck audit PATH      --out --no-hindsight --hindsight-bin --only --fail-on
ruck demo            --out-dir --device
ruck fetch espn      --league --dates --cache-dir
ruck train CORPUS    --out --epochs --max-vocab --device
```

`TARGET` is a CSV, a RugbyPy file, or `espn:<league>:<event_id>`. `ruck audit`
exits 1 when `--fail-on` is given and a finding meets it, so it drops into CI.

Fetching is rate-limited to one request per second per host with exponential
backoff, and everything is cached under `data/raw/`. ESPN's edge wants a product
token with a contact comment in the User-Agent, a bare token and a browser
impersonation string both get a 403, so that is what is sent; override with
`RUCK_USER_AGENT`.

## Layout

```
src/ruck/
  schema.py        canonical rugby events, the sportio contract
  tokenizer.py     the four-field token and its backoff cascade
  ingest/          generic CSV, ESPN, RugbyPy, and a polite cached fetcher
  lm/              model, training, perplexity baselines, downstream comparison
  report/          surprisal, attack shape, charts, PDF, timestamp CSV
  audit/           checks, the hindsight wrapper, the markdown report
  demo_data/       the seeded generator
demo/              bundled model, corpus, ESPN cache, audit fixtures
docs/csv_format.md the export format to target
TODO.md            what is known-incomplete and why
```

## Licence

MIT, see [`LICENSE`](LICENSE). Vendored audit fixtures under
`demo/audit_fixtures/` keep their upstream licences; see the `SOURCE.md` beside
each.
