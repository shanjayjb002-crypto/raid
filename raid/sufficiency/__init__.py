"""Diagram completeness and ambiguity scoring.

Inspects a parsed diagram *before* any generation is attempted and reports how
analysable it currently is for each downstream concern - test generation,
resilience simulation and fuzzing - along with the specific changes that would
improve each one.

Typical use::

    from raid.dsl import parse
    from raid.graph import build_graph
    from raid.sufficiency import score_sufficiency

    report = score_sufficiency(build_graph(parse(source_text)))

    print(report.overall)
    for suggestion in report.suggestions:
        print(f"[{suggestion.severity}] {suggestion.message}")
"""

from .model import (
    CONCERNS,
    SCORES,
    ConcernScore,
    SufficiencyReport,
    Suggestion,
)
from .scorer import score_sufficiency

__all__ = [
    "CONCERNS",
    "SCORES",
    "ConcernScore",
    "SufficiencyReport",
    "Suggestion",
    "score_sufficiency",
]
