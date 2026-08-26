# ruck audit

**Target:** `/Users/oscardalton/ruck/tests/fixtures/leaky_workload`  
**Scanned:** 1 Python file · 25 checks · ruck native checks  
**Generated:** (regenerate with make expected)

The target directory is only ever read. Files are parsed with `ast`; nothing in them is imported or executed.

## Summary

**28 findings.**

| Severity | Count |
| --- | ---: |
| critical | 7 |
| high | 12 |
| medium | 9 |

| Category | Count |
| --- | ---: |
| Lookahead | 9 |
| Label bleed | 6 |
| Unpurged cross-validation | 5 |
| Acute:chronic ratio | 4 |
| Survivorship | 3 |
| Preprocessing leakage | 1 |

## Acute:chronic workload ratio

The rugby-specific section. An ACWR divides a recent window of load by a longer one over the same series. In the standard formulation the acute window sits inside the chronic window, so writing the chronic block as the acute week plus the three before it, `ACWR = A / ((A + B) / 4) = 4A / (A + B)`. That is a bounded, increasing function of acute load on its own: it goes up after a hard week whether or not hard weeks cause injuries. A model fitted on it inherits the relationship from the algebra.

**4 findings in this section.** Each one is written up in full below.

| Check | Where | Severity | What |
| --- | --- | --- | --- |
| AC001 | `pipeline.py:48` | critical | Coupled acute:chronic ratio — the acute window is inside the chronic window |
| AC003 | `pipeline.py:53` | high | Exponentially weighted ACWR shares every observation between numerator and denominator |
| AC004 | `pipeline.py:48` | medium | ACWR divides by chronic load with no floor on the denominator |
| AC005 | `pipeline.py:56` | medium | ACWR binned into sweet-spot categories |

The single most useful thing to do with a coupled ratio is to recompute it uncoupled and refit. If the association with injury survives, it was real; if it disappears, the model was reading its own arithmetic.

## Findings

### 1. LA002 — Centred rolling window averages across the current observation

**critical** · Lookahead · confidence high · detected by ruck

`pipeline.py:39`

```python
panel["player_load"].rolling(14, center=True)
```

**Why this matters**

`center=True` places the window symmetrically, so roughly 7 observations after the current row are inside it. A centred seven-day load average on Tuesday contains Thursday and Friday. Smoothing this way is fine for a chart and never fine for a feature.

**Fix**

Drop `center=True`; a trailing window is the causal one.

### 2. LA001 — Negative shift pulls 1 future observation(s) backwards

**critical** · Lookahead · confidence high · detected by ruck

`pipeline.py:41`

```python
panel["session_rpe"].shift(-1)
```

**Why this matters**

`shift(-1)` moves values from 1 row(s) in the future onto the current row. If the result is a feature, the model is reading sessions that have not happened. If it is a label, that is correct and intended — but then nothing built from the same shifted frame may be a feature.

**Fix**

Use a negative shift only to construct the label, in its own statement, and build features from positive shifts of the unshifted frame.

### 3. AC001 — Coupled acute:chronic ratio — the acute window is inside the chronic window

**critical** · Acute:chronic ratio · confidence high · detected by ruck

`pipeline.py:48`

```python
features["acwr"] = acute / chronic
```

**Why this matters**

The acute window is the last 7 observations and the chronic window is the last 28, so 7 observations are counted in both the numerator and the denominator. Writing the chronic block as acute plus the rest, the ratio reduces to 4A / (A + B): a bounded, increasing function of acute load whatever B does. It rises after a hard week because of the algebra, not because of the athlete. An association between this quantity and injury is therefore not evidence of one between load and injury.

**Fix**

Take the chronic window from the period before the acute one so the two share no sessions: `chronic = load.shift(7).rolling(21).sum()`, then divide by the number of weeks it spans. Report the uncoupled ratio next to the coupled one; if the effect only exists in the coupled version, it was the algebra.

### 4. LB002 — Feature `risk_score` is computed from the outcome

**critical** · Label bleed · confidence medium · detected by ruck

`pipeline.py:61`

```python
features["risk_score"] = panel["injured"] * 0.9 + np.random.rand(len(panel)) * 0.1
```

**Why this matters**

`risk_score` is derived from 'injured'. Even a heavily transformed function of the label is still the label; the model recovers it and the score is meaningless.

**Fix**

Compute the feature from workload and availability data recorded before the outcome window opens.

### 5. SV001 — Population is filtered by the outcome being predicted

**critical** · Survivorship · confidence medium · detected by ruck

`pipeline.py:73`

```python
panel["injured"] == 0
```

**Why this matters**

The rows kept are chosen by the injury column, so the sample is conditioned on the outcome. If uninjured rows are kept for feature building, the features describe uninjured weeks and the contrast the model needs has been removed; if injured players are dropped entirely, the cohort is the players who survived the season.

**Fix**

Keep every athlete-week in the panel and let the label mark which ones ended in an injury. If athlete-weeks must be excluded — a player on loan, a mid-season signing — exclude them on a rule that does not mention the outcome, and record the count.

### 6. LB001 — Feature set contains the outcome column (injured)

**critical** · Label bleed · confidence high · detected by ruck

`pipeline.py:81`

```python
X = features[["acute_load", "acwr", "injured"]]
```

**Why this matters**

`X` selects 'injured', which is the thing being predicted. Any model fitted on this reports near-perfect skill and has learned nothing.

**Fix**

Build the feature list explicitly and assert the label is not in it, rather than dropping columns by name after the fact.

### 7. CV001 — KFold(shuffle=True) on a time-ordered panel

**critical** · Unpurged cross-validation · confidence high · detected by ruck

`pipeline.py:95`

```python
KFold(n_splits=5, shuffle=True, random_state=0)
```

**Why this matters**

Shuffling breaks the time order, so a fold can train on next week and test on this one. Worse here: rolling windows mean adjacent rows share most of their input, so a test row's 28-day window overlaps a training row's by 27 days. The fold boundary is not a boundary.

**Fix**

Split forward in time (`TimeSeriesSplit`, or explicit date cut-offs), and purge a gap at least as long as the longest feature window so no test row shares data with a training row.

### 8. LA003 — Backward fill copies later values onto earlier rows

**high** · Lookahead · confidence high · detected by ruck

`pipeline.py:32`

```python
panel["total_distance"].bfill()
```

**Why this matters**

Backfilling a gap in a workload series fills it with the next recorded session. Missing GPS is not missing at random — it clusters around injuries, illness and rest — so the filled value is drawn from exactly the period the model is trying to predict.

**Fix**

Forward-fill (`ffill`) if carrying the last known value is defensible, or leave the gap and let the model see a missing indicator. Say in the write-up which one you chose.

### 9. LA005 — `diff(-3)` differences against a future observation

**high** · Lookahead · confidence high · detected by ruck

`pipeline.py:43`

```python
panel["high_speed_running"].diff(-3)
```

**Why this matters**

A negative period differences the current row against a later one, so the result encodes what happens next.

**Fix**

Use a positive period to difference against the past.

### 10. AC006 — Acute window includes the day the outcome is recorded

**high** · Label bleed · confidence medium · detected by ruck

`pipeline.py:48`

```python
features["acwr"] = acute / chronic
```

**Why this matters**

The acute window ends on the current row, so a session's own load is a feature for that session's outcome. When an injury occurs during a session, the truncated or elevated load of that session is in the window — the feature partly encodes the label. Models trained this way look predictive and cannot be used prospectively, because on the morning of a session that load does not exist yet.

**Fix**

Shift the whole feature block by one observation before the ratio: `load.shift(1).rolling(7).sum()`. Every feature should be computable the morning before the session it describes.

### 11. AC003 — Exponentially weighted ACWR shares every observation between numerator and denominator

**high** · Acute:chronic ratio · confidence high · detected by ruck

`pipeline.py:53`

```python
features["ewma_acwr"] = ewma_acute / ewma_chronic
```

**Why this matters**

Both the acute and the chronic term are exponentially weighted means of the same series, so every session contributes to both. The ratio is a smoothed function of the series divided by a more smoothed function of the same series; it moves with recent load by construction. EWMA removes the hard window edge, not the coupling.

**Fix**

Compute the chronic term from load up to the start of the acute period: `chronic = load.shift(acute_span).ewm(span=chronic_span).mean()`. Better, drop the ratio and give the model acute and chronic load as two features, letting it find its own combination.

### 12. LB003 — Rolling window is computed over the outcome column

**high** · Label bleed · confidence high · detected by ruck

`pipeline.py:59`

```python
features["recent_history"] = panel[["total_distance", "injured"]].rolling(28).mean()
```

**Why this matters**

The window covers 'injured' alongside the workload columns, so the aggregate carries the outcome into a feature. A 28-day mean that includes an injury flag is a smoothed injury flag.

**Fix**

Select only workload columns before the window, by explicit list.

### 13. LB005 — `days_since_injury` is defined by the outcome it is used to predict

**high** · Label bleed · confidence medium · detected by ruck

`pipeline.py:63`

```python
features["days_since_injury"] = (panel["date"] - panel["injury_date"]).dt.days
```

**Why this matters**

Injury-history features are legitimate only when they are strictly backward-looking from the prediction point. Computed over the whole record — which is the default when the column is built once from the full frame — they encode injuries that have not happened yet. `days_since_injury` computed this way is negative before an injury and the model reads the sign.

**Fix**

Recompute the feature as of each prediction date, using only injuries recorded strictly before it, and unit-test the boundary.

### 14. LB004 — Feature window is inclusive of the injury date

**high** · Label bleed · confidence medium · detected by ruck

`pipeline.py:66`

```python
panel["date"] <= injury_date
```

**Why this matters**

`<=` keeps the injury day itself inside the feature window. That day's session is the one that was cut short, so its load is unusually low and unusually informative. The comparison should be strict.

**Fix**

Use `<` against the injury date, or better, end the window a full day before it so a mid-session injury cannot bleed in at all.

### 15. SV003 — Cohort selected on a season total (`games_played`)

**high** · Survivorship · confidence medium · detected by ruck

`pipeline.py:75`

```python
fit_players["games_played"] > 12
```

**Why this matters**

`games_played` is only known once the season is over, and it is lower for exactly the players who got injured. Requiring a minimum therefore removes the injured players from the sample before the model sees them. In September, the squad this model will be applied to includes everyone.

**Fix**

Select the cohort on information available at the start of the window (contracted, in the squad, has n weeks of prior data) rather than on totals accumulated across it.

### 16. LA006 — StandardScaler fitted on the full dataset before the split

**high** · Preprocessing leakage · confidence medium · detected by ruck

`pipeline.py:86`

```python
scaler.fit_transform(X)
```

**Why this matters**

`scaler.fit_transform` is fitted on `X`, which is the same data handed to the splitter at line 89. The transformer's statistics — means, quantiles, imputation values, selected features — are therefore computed over rows that end up in the test set, so the test set is no longer held out. On a workload panel this is usually worth a few points of AUC that do not survive deployment.

**Fix**

Split first, then fit on the training rows only, or wrap the transformer and the estimator in a `Pipeline` and fit the pipeline inside the cross-validation loop.

### 17. CV002 — train_test_split shuffles by default

**high** · Unpurged cross-validation · confidence medium · detected by ruck

`pipeline.py:89`

```python
train_test_split(X_scaled, y, test_size=0.2)
```

**Why this matters**

`shuffle` defaults to True, so the test set is drawn uniformly from the whole season rather than from its end. The reported score is an average over a task the model will never face: it never gets to interpolate between weeks it has already seen.

**Fix**

Pass `shuffle=False` and split on a date, or use a grouped, time-ordered splitter.

### 18. CV003 — cross_val_score uses the default K-fold split

**high** · Unpurged cross-validation · confidence medium · detected by ruck

`pipeline.py:97`

```python
cross_val_score(model, X_scaled, y, cv=5)
```

**Why this matters**

With `cv` unset or an integer, scikit-learn builds a plain (stratified) K-fold. On an athlete-week panel that mixes athletes and weeks across the fold boundary in both directions.

**Fix**

Pass an explicit splitter that is both time-ordered and grouped by athlete, and pass `groups=` so it is honoured.

### 19. CV005 — TimeSeriesSplit with no gap, but features use rolling windows

**high** · Unpurged cross-validation · confidence high · detected by ruck

`pipeline.py:100`

```python
TimeSeriesSplit(n_splits=5)
```

**Why this matters**

The split respects time order but leaves no embargo. The first test row sits one day after the last training row, and its 28-day chronic window contains 27 days that are in the training set. The fold boundary leaks for as many rows as the window is long.

**Fix**

Set `gap` to at least the longest feature window — 28 for a four-week chronic load, more if any feature looks further back.

### 20. SV002 — Unrestricted dropna() removes athlete-weeks non-randomly

**medium** · Survivorship · confidence medium · detected by ruck

`pipeline.py:24`

```python
panel.dropna()
```

**Why this matters**

Missing GPS is not missing at random. It is concentrated in the weeks around injury, illness and rehab — the weeks the model most needs. Dropping every row with any null quietly removes them, and also removes the first 27 days of every athlete's record, where the 28-day chronic window has not filled.

**Fix**

Drop on an explicit subset, count what was dropped and report it, and check whether the dropped rows are disproportionately close to injuries.

### 21. LA004 — Interpolation fills gaps from both sides

**medium** · Lookahead · confidence medium · detected by ruck

`pipeline.py:34`

```python
panel["player_load"].interpolate()
```

**Why this matters**

pandas interpolates between the values on either side of a gap, so a filled Wednesday is partly made of Friday. Over a missed week around an injury, that is the outcome leaking into the features.

**Fix**

Pass `limit_direction='forward'`, or leave the gap and flag it.

### 22. LA007 — Rolling feature window ends on the observation it is a feature for

**medium** · Lookahead · confidence medium · detected by ruck

`pipeline.py:37`

```python
features["acute_load"] = panel["total_distance"].rolling(7).sum()
```

**Why this matters**

`acute_load` is a 7-observation window with no shift, so the current session's own load is inside it. For a same-day outcome that is a feature containing part of the label, and prospectively the value does not exist until the session is over.

**Fix**

Shift before the window: `.shift(1).rolling(7)`. If the outcome is genuinely next-day, say so explicitly and keep the shift anyway, so the feature is computable in the morning.

### 23. AC004 — ACWR divides by chronic load with no floor on the denominator

**medium** · Acute:chronic ratio · confidence medium · detected by ruck

`pipeline.py:48`

```python
features["acwr"] = acute / chronic
```

**Why this matters**

Chronic load is near zero for a returning player, pre-season, or the first four weeks of any athlete's record. Dividing by it produces ratios in the hundreds or an infinity, and those rows then dominate whatever is fitted on them. The extreme values are an artefact of the denominator, not a training spike.

**Fix**

Require a minimum chronic period before emitting a ratio (`rolling(28, min_periods=28)`), and drop rather than clip the rows that do not have one. Clipping keeps a fabricated value in the sample.

### 24. LA007 — Rolling feature window ends on the observation it is a feature for

**medium** · Lookahead · confidence medium · detected by ruck

`pipeline.py:51`

```python
ewma_acute = panel["total_distance"].ewm(span=7).mean()
```

**Why this matters**

`ewma_acute` is a 7-observation window with no shift, so the current session's own load is inside it. For a same-day outcome that is a feature containing part of the label, and prospectively the value does not exist until the session is over.

**Fix**

Shift before the window: `.shift(1).rolling(7)`. If the outcome is genuinely next-day, say so explicitly and keep the shift anyway, so the feature is computable in the morning.

### 25. LA007 — Rolling feature window ends on the observation it is a feature for

**medium** · Lookahead · confidence medium · detected by ruck

`pipeline.py:52`

```python
ewma_chronic = panel["total_distance"].ewm(span=28).mean()
```

**Why this matters**

`ewma_chronic` is a 28-observation window with no shift, so the current session's own load is inside it. For a same-day outcome that is a feature containing part of the label, and prospectively the value does not exist until the session is over.

**Fix**

Shift before the window: `.shift(1).rolling(28)`. If the outcome is genuinely next-day, say so explicitly and keep the shift anyway, so the feature is computable in the morning.

### 26. AC005 — ACWR binned into sweet-spot categories

**medium** · Acute:chronic ratio · confidence medium · detected by ruck

`pipeline.py:56`

```python
pd.cut(features["acwr"], bins=[0, 0.8, 1.3, 1.5, 99])
```

**Why this matters**

Cutting the ratio at 0.8 / 1.3 / 1.5 turns a mechanically coupled quantity into a risk category, which hides the coupling behind a label. The bin edges are inherited from published work whose ratios were built the same coupled way, so agreement with it is not corroboration.

**Fix**

Keep the ratio continuous, report the uncoupled version alongside it, and let any thresholds be fitted on this squad rather than imported.

### 27. LA007 — Rolling feature window ends on the observation it is a feature for

**medium** · Lookahead · confidence medium · detected by ruck

`pipeline.py:59`

```python
features["recent_history"] = panel[["total_distance", "injured"]].rolling(28).mean()
```

**Why this matters**

`recent_history` is a 28-observation window with no shift, so the current session's own load is inside it. For a same-day outcome that is a feature containing part of the label, and prospectively the value does not exist until the session is over.

**Fix**

Shift before the window: `.shift(1).rolling(28)`. If the outcome is genuinely next-day, say so explicitly and keep the shift anyway, so the feature is computable in the morning.

### 28. CV004 — Athlete-level panel split without grouping by athlete

**medium** · Unpurged cross-validation · confidence low · detected by ruck

`pipeline.py:89`

```python
train_test_split(X_scaled, y, test_size=0.2)
```

**Why this matters**

The file works with athlete identifiers (player_id) but no splitter is grouped by them and no `groups=` is passed. The same player appears in training and test, so part of what the model learns is that player's baseline. Squad-level scores are then optimistic for anyone new.

**Fix**

Use `StratifiedGroupKFold` or `GroupKFold` with `groups=df['player_id']`, or hold out whole athletes as well as later dates.

## Purge list — GPS workload data

Written for a squad's GPS and session-load export: one row per athlete per session or per day, columns for distance, high-speed running, player load, accelerations and session RPE, and an availability or injury record alongside it. Work down the list; each item is a thing to remove or change before refitting.

### 1. Purge the current observation from every feature window

A session's own load must not appear in a feature predicting that session's outcome. Shift by one observation before every window:

```python
loads = panel.sort_values(['player_id', 'date'])
past = loads.groupby('player_id')['total_distance'].shift(1)
acute = past.rolling(7, min_periods=7).sum()
```

Columns this applies to in this repository: `acute_load`, `high_speed_running`, `load_delta`, `next_session_load`, `player_load`, `session_rpe`, `smoothed_load`, `total_distance`.

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

Outcome columns found in this repository: `days_since_injury`, `injured`, `injury_date`.

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
