"""Bundled demo data.

Everything in here is either a real payload vendored from a public source and
labelled as such, or output of the seeded generator in
:mod:`ruck.demo_data.generate`, which is labelled SYNTHETIC everywhere it
surfaces. The two are never mixed inside one match.
"""

from ruck.demo_data.generate import (
    SYNTHETIC_COMPETITION,
    TEAMS,
    generate_corpus,
    generate_match,
)

__all__ = ["SYNTHETIC_COMPETITION", "TEAMS", "generate_corpus", "generate_match"]
