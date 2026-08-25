<div align="center">

# acwr

**Acute:chronic workload ratio and training-load metrics — no dependencies, no spreadsheet.**

<p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT" /></a>
  <img src="https://img.shields.io/badge/dependencies-none-30a46c?style=flat-square" alt="No dependencies" />
  <img src="https://img.shields.io/badge/tests-34%20passing-30a46c?style=flat-square" alt="34 tests" />
</p>

<img src="docs/demo.svg" width="620" alt="acwr CLI output showing a training block pushing the ratio into the high-risk band" />

</div>

---

## What It Does

The acute:chronic workload ratio compares how hard you have trained **recently**
(the acute window, typically 7 days) against what your body is **used to** (the
chronic window, typically 28 days). A ratio near 1.0 means today looks like your
normal. Ramp up too fast and the ratio climbs — which is associated with a higher
injury rate across a lot of team-sport literature.

Everyone recomputes this in a spreadsheet and gets it subtly wrong. This is the
library version, with the edge cases handled and the failure modes named.

## Install

```bash
pip install acwr        # or: pip install git+https://github.com/AbdullahKaddoura/acwr
```

## Use It

```python
from acwr import rolling_acwr, ewma_acwr

loads = [300] * 21 + [900] * 7      # three steady weeks, then a hard block
points = rolling_acwr(loads)

latest = points[-1]
latest.ratio      # 2.0
latest.zone       # <Zone.HIGH_RISK: 'high_risk'>
latest.acute      # 900.0
latest.chronic    # 450.0
```

Loads are unitless — session-RPE × minutes, TRIMP, WHOOP strain, distance,
anything monotonic in *how hard was that day*. **Rest days are `0.0` and they
must be included**; dropping them silently inflates every window that follows.

### From the command line

```bash
acwr example.csv                       # rolling, last 14 days
acwr example.csv --method ewma -n 30   # exponentially weighted
acwr data.csv --column strain -a 5 -C 21
```

## The Two Methods

| | Rolling average | EWMA |
| :--- | :--- | :--- |
| Source | Hulin et al. (2014) | Williams et al. (2017) |
| Weighting | Every day in the window counts equally | Recent days count more |
| Day 29 | Day 1 drops out entirely, which can jolt the ratio | Decays smoothly, nothing falls off a cliff |
| Reacts to a spike | Slower | Faster |
| Function | `rolling_acwr()` | `ewma_acwr()` |

Use `ewma_acwr` if you want the ratio to respond quickly; use `rolling_acwr` if
you want the version most papers and most coaches will recognise.

## Zones

| Ratio | `Zone` | Reading |
| :--- | :--- | :--- |
| `< 0.8` | `UNDERLOADED` | Detrained relative to your base |
| `0.8 – 1.3` | `OPTIMAL` | The commonly cited "sweet spot" |
| `1.3 – 1.5` | `CAUTION` | Ramping faster than you have adapted |
| `> 1.5` | `HIGH_RISK` | Associated with elevated injury rates |
| — | `UNDEFINED` | Not enough history, or no chronic load |

## Also Included

```python
from acwr import monotony, strain

monotony(week)   # Foster: mean daily load / SD. >2.0 means every day looks the same.
strain(week)     # Foster: total load × monotony.
```

## Three Ways People Get This Wrong

**Reading the ratio before the chronic window is full.** With 10 days of data
and a 28-day chronic window, "chronic" is really just "acute, slightly longer",
and the ratio hovers near 1.0 no matter what you do. This library returns
`ratio = None` until day 28. You can override it with
`require_full_window=False`, but now you know what you are looking at.

**Dropping rest days.** A rest day is a `0.0`, not a missing row. Omit them and
both windows shrink, both means rise, and your chronic load looks far higher
than it is.

**Treating one hard session as a spike.** It usually isn't — a single big day
inside 28 days of history moves the ratio surprisingly little. There is a test
asserting exactly this (`test_single_hard_session_barely_moves_the_ratio`).
Sustained blocks move it; single sessions mostly don't.

## Caveats

The 0.8–1.3 sweet spot is a **population-level heuristic from team-sport
cohorts**, and it has had a genuinely rough time in the methodological
literature — the discretisation into zones, the spurious-correlation critique,
and the small effect sizes are all live arguments. Treat it as one signal
among several, not as a verdict, and definitely not as medical advice.

## Development

```bash
git clone https://github.com/AbdullahKaddoura/acwr.git && cd acwr
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## References

- Hulin et al. (2014), *Br J Sports Med* — the rolling-average formulation
- Williams et al. (2017), *Br J Sports Med* — the EWMA formulation
- Foster (1998), *Med Sci Sports Exerc* — monotony and strain
- Impellizzeri et al. (2020), *Int J Sports Physiol Perform* — the critique worth reading

## License

MIT
