# Generic event CSV

The format `ruck analyze` is built around. Export from Opta, Stats Perform,
SportsCode, Nacsport or your own tagging sheet; if it has a time, a team and an
action, ruck will read it.

## Columns

| Column | Required | Meaning |
| --- | --- | --- |
| `t` | yes | Match clock. Seconds from kickoff, or `MM:SS`, `HH:MM:SS`, or `46'`. Cumulative across halves. |
| `team` | yes | Team in possession for this event. |
| `action` | yes | What happened. See the vocabulary below. |
| `x` | no | Metres from the attacking team's own goal line, 0-100. |
| `y` | no | Metres across the pitch, 0-70. |
| `phase` | no | Phase count in the current possession; 0 is the set piece. Reconstructed if absent. |
| `outcome` | no | How the event ended. See the vocabulary below. |

Also read when present: `period`, `player`, `player_id`, `team_id`, `subtype`,
`end_x`, `end_y`, `possession_id`, `pressure`, `match_id`, `competition`,
`season`.

Header names are matched case- and punctuation-insensitively, and common
aliases are accepted: `time`/`timestamp`/`seconds` for `t`, `event`/`type`/`code`
for `action`, `start_x`/`pitch_x` for `x`, `result`/`success` for `outcome`,
`half` for `period`, and so on. The full alias table is
`COLUMN_ALIASES` in `src/ruck/ingest/generic_csv.py`.

## Coordinates

Two conventions are supported, chosen with `coordinates=`:

- `metres` — `x` in 0-100, `y` in 0-70.
- `percent` — both in 0-100; `y` is rescaled to metres on load.
- `auto` (default) — `percent` if any `y` exceeds 70, else `metres`.

And two orientations, chosen with `direction=`:

- `possession` (default) — coordinates are already normalized so the team in
  possession attacks left-to-right. This is the suite convention: `x = 95`
  always means "five metres out", whichever end the team is playing into.
- `absolute` — `x` is measured in a fixed camera frame. Pass `home_team=` as
  well; away-team events and second-half events are mirrored on load.

If your export has no coordinates at all it still works. The tokenizer emits an
`unknown` zone, the surprisal timeline is still produced, and the attack-shape
section says which parts it could not compute.

## Vocabulary

Action labels are folded (lowercased, punctuation stripped) and looked up in a
fixed table, so `Ball Carry`, `ball_carry` and `CARRY` are the same token. The
canonical actions are:

```
carry  pass  offload  kick  ruck  maul  lineout  scrum  tackle  turnover
penalty  free_kick  knock_on  forward_pass  try  conversion  penalty_goal
drop_goal  card  restart  touch  mark  substitution  end_period  other
```

A label the table does not know becomes `other` with the original preserved in
`subtype`, and the load report tells you how many times each unknown label
appeared. Nothing is dropped silently and nothing is guessed.

Canonical outcomes:

```
complete  incomplete  retained  lost  won  penalty_won  penalty_conceded
points  held  into_touch  in_field  unknown
```

## Minimal example

```csv
t,team,action,x,y,phase,outcome
0,Munster,restart,50.0,35.0,0,complete
4.2,Munster,carry,52.0,30.1,0,retained
6.0,Munster,ruck,54.5,29.4,1,retained
7.8,Munster,pass,55.0,29.0,1,complete
9.1,Munster,carry,58.2,44.0,1,retained
11.0,Munster,ruck,60.0,45.2,2,retained
13.4,Munster,kick,61.0,44.8,2,into_touch
```

Load it:

```python
from ruck.ingest import load_csv

events, report = load_csv("match.csv", competition="URC", season="2024")
print(report.summary())
```
