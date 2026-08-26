# ruck audit

**Target:** `/Users/oscardalton/ruck/demo/audit_fixtures`  
**Scanned:** 25 Python files · 25 checks · ruck native checks  
**Generated:** 2026-08-26 00:34 UTC

The target directory is only ever read. Files are parsed with `ast`; nothing in them is imported or executed.

## Summary

**7 findings.**

| Severity | Count |
| --- | ---: |
| critical | 1 |
| high | 4 |
| low | 2 |

| Category | Count |
| --- | ---: |
| Acute:chronic ratio | 4 |
| Unpurged cross-validation | 3 |

## Acute:chronic workload ratio

The rugby-specific section. An ACWR divides a recent window of load by a longer one over the same series. In the standard formulation the acute window sits inside the chronic window, so writing the chronic block as the acute week plus the three before it, `ACWR = A / ((A + B) / 4) = 4A / (A + B)`. That is a bounded, increasing function of acute load on its own: it goes up after a hard week whether or not hard weeks cause injuries. A model fitted on it inherits the relationship from the algebra.

**4 findings in this section.** Each one is written up in full below.

| Check | Where | Severity | What |
| --- | --- | --- | --- |
| AC002 | `acwr/src/acwr/core.py:134` | critical | Coupled acute:chronic ratio — both windows end on the same day |
| AC003 | `acwr/src/acwr/core.py:187` | high | Exponentially weighted ACWR shares every observation between numerator and denominator |
| AC005 | `acwr/src/acwr/core.py:71` | low | ACWR compared against an imported sweet-spot threshold (3 thresholds in this file) |
| AC005 | `acwr/tests/test_core.py:34` | low | ACWR compared against an imported sweet-spot threshold |

The single most useful thing to do with a coupled ratio is to recompute it uncoupled and refit. If the association with injury survives, it was real; if it disappears, the model was reading its own arithmetic.

## Findings

### 1. AC002 — Coupled acute:chronic ratio — both windows end on the same day

**critical** · Acute:chronic ratio · confidence high · detected by ruck

`acwr/src/acwr/core.py:134`

```python
acute_mean / chronic_mean
```

**Why this matters**

The two windows are slices of `series` that both end at the same index, with lengths `acute` and `chronic`. The shorter window is therefore entirely inside the longer one, and every day in the numerator is counted again in the denominator. With the chronic block written as the acute window plus the rest, the ratio reduces to a bounded increasing function of acute load: it rises after a hard week as a matter of arithmetic, before any athlete is involved.

**Fix**

End the chronic slice where the acute one begins, so the two share no observations: `series[max(0, i - chronic + 1) : max(0, i - acute + 1)]`. Divide by its own length rather than the acute one's. Compute both the coupled and uncoupled ratios for one season and compare before choosing.

### 2. AC003 — Exponentially weighted ACWR shares every observation between numerator and denominator

**high** · Acute:chronic ratio · confidence high · detected by ruck

`acwr/src/acwr/core.py:187`

```python
ewma_a / ewma_c
```

**Why this matters**

`ewma_a` and `ewma_c` are two exponentially weighted means of the same load series with different smoothing constants, so every session contributes to both. Removing the hard window edge removes the discontinuity, not the coupling: the ratio is still a function of the series divided by a smoother function of the same series.

**Fix**

Seed the chronic EWMA from load up to the start of the acute period, or report acute and chronic load as two features and let the model combine them.

### 3. CV002 — train_test_split shuffles by default

**high** · Unpurged cross-validation · confidence medium · detected by ruck

`injury_prediction/src/train_injury_risk_lstm.py:10`

```python
train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True )
```

**Why this matters**

`shuffle` defaults to True, so the test set is drawn uniformly from the whole season rather than from its end. The reported score is an average over a task the model will never face: it never gets to interpolate between weeks it has already seen.

**Fix**

Pass `shuffle=False` and split on a date, or use a grouped, time-ordered splitter.

### 4. CV002 — train_test_split shuffles by default

**high** · Unpurged cross-validation · confidence medium · detected by ruck

`injury_prediction/src/train_lstm_volleyball.py:80`

```python
train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
```

**Why this matters**

`shuffle` defaults to True, so the test set is drawn uniformly from the whole season rather than from its end. The reported score is an average over a task the model will never face: it never gets to interpolate between weeks it has already seen.

**Fix**

Pass `shuffle=False` and split on a date, or use a grouped, time-ordered splitter.

### 5. CV002 — train_test_split shuffles by default

**high** · Unpurged cross-validation · confidence medium · detected by ruck

`injury_prediction/src/train_with_synth.py:48`

```python
train_test_split(X_all, y_all, test_size=0.2, random_state=42, shuffle=True)
```

**Why this matters**

`shuffle` defaults to True, so the test set is drawn uniformly from the whole season rather than from its end. The reported score is an average over a task the model will never face: it never gets to interpolate between weeks it has already seen.

**Fix**

Pass `shuffle=False` and split on a date, or use a grouped, time-ordered splitter.

### 6. AC005 — ACWR compared against an imported sweet-spot threshold (3 thresholds in this file)

**low** · Acute:chronic ratio · confidence medium · detected by ruck

`acwr/src/acwr/core.py:71`

```python
ratio < 0.8
```

**Why this matters**

The threshold comes from studies whose ratio was coupled the same way this one is. Applying it here reproduces the artefact rather than testing it.

**Fix**

Fit any threshold on this squad's own uncoupled ratio, or drop it.

### 7. AC005 — ACWR compared against an imported sweet-spot threshold

**low** · Acute:chronic ratio · confidence medium · detected by ruck

`acwr/tests/test_core.py:34`

```python
points[-1].ratio < 0.8
```

**Why this matters**

The threshold comes from studies whose ratio was coupled the same way this one is. Applying it here reproduces the artefact rather than testing it.

**Fix**

Fit any threshold on this squad's own uncoupled ratio, or drop it.

## Purge list — GPS workload data

Written for a squad's GPS and session-load export: one row per athlete per session or per day, columns for distance, high-speed running, player load, accelerations and session RPE, and an availability or injury record alongside it. Work down the list; each item is a thing to remove or change before refitting.

### 1. Purge the current observation from every feature window

A session's own load must not appear in a feature predicting that session's outcome. Shift by one observation before every window:

```python
loads = panel.sort_values(['player_id', 'time'])
past = loads.groupby('player_id')['load'].shift(1)
acute = past.rolling(7, min_periods=7).sum()
```

Columns this applies to in this repository: `duration`, `load`.

### 2. Purge the acute window out of the chronic window

Compute chronic load from the period *before* the acute one so the two share no sessions:

```python
acute   = past.rolling(7,  min_periods=7).sum()
chronic = past.shift(7).rolling(21, min_periods=21).sum() / 3   # weekly equivalent
acwr    = acute / chronic
```

Report the coupled and uncoupled ratios side by side for one season before choosing. A relationship that only exists in the coupled version is arithmetic.

### 3. Purge rows whose windows are not yet full

The first four weeks of every athlete's record have an incomplete chronic window. Passing `min_periods` and then dropping those rows is honest; letting pandas emit a partial mean is not, and clipping the resulting extreme ratios keeps a fabricated number in the sample. Expect to lose the opening 27 athlete-days of each record and one window after every gap in collection.

### 4. Purge the outcome and everything built from it out of the feature matrix

Outcome columns found in this repository: `label`, `y`.

Keep out of `X`: the outcome itself, anything derived from it (`days_since_injury`, `injury_count`, `availability`, `games_missed`) unless recomputed as of each prediction date, and any column whose value is only known once the season is over (appearances, total minutes, sessions completed). Build the feature list explicitly and assert the outcome is absent, rather than dropping by name afterwards.

### 5. Purge the fold boundary

Split forward in time with an embargo of at least 28 days — the longest feature window in this repository — so no test row shares data with a training row, and group by athlete so the same player is not on both sides:

```python
splitter = TimeSeriesSplit(n_splits=5, gap=28)
#   plus: hold out whole athletes as well as later dates, and pass
#   groups=panel['player_id']
```

### 6. Purge non-random missingness rather than filling it

Missing GPS clusters around injury, illness and rehab, which is exactly the period being modelled. Do not backfill and do not interpolate across a gap. Carry the last value forward only if that is defensible for the metric, otherwise leave the gap, add a missing-indicator column, and report how many athlete-days were affected and how close they sat to recorded injuries.

## Notes

Hindsight was not available, so only ruck's native checks ran. Install it and re-run for the wider trading-research taxonomy on top of these.

### What this audit cannot see

- Leakage inside a SQL query or a data warehouse view that arrives as a finished table.
- A window that is correct in code but wrong because the export's timestamps are session start rather than session end.
- Anything in a notebook whose cells were run out of order.
- Whether the injury records themselves were entered retrospectively, which makes even a correct pipeline unusable.
