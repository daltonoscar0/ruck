# mdana5/injury-prediction

Vendored from <https://github.com/mdana5/injury-prediction> at commit `207a09d7ee3d2a06d1a5d4b9889093ccb42c0a6d`.

**Why it is here:** A full injury-prediction pipeline: feature extraction, windowing, an LSTM and a train/test split. Exercises the lookahead, label-bleed and cross-validation checks on code nobody wrote for this audit.

This is a snapshot of a public MIT-licensed repository, redistributed under that licence (see `LICENSE` in this directory). It is included as audit input so `ruck audit demo/` has real third-party code to read offline. Findings reported against it describe patterns in a snapshot of the code, not the competence of its authors, the same patterns appear in most published workload-modelling code, which is the point the audit is making.

Nothing here is imported or executed. `ruck audit` parses it with `ast`.

Refresh with `python scripts/vendor_audit_fixtures.py`.
