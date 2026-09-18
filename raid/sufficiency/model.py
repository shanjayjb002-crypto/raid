"""Result types for the diagram sufficiency scorer."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CONCERNS",
    "SCORES",
    "ConcernScore",
    "SufficiencyReport",
    "Suggestion",
]

#: The three downstream concerns whose readiness is scored.
CONCERNS: tuple[str, ...] = ("testgen", "resilience", "fuzzing")

#: Score vocabulary, worst to best. Higher means *more analysable*, so a
#: diagram scoring "high" is well specified rather than badly broken.
SCORES: tuple[str, ...] = ("low", "medium", "high")

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class Suggestion:
    """One specific, actionable change that would make the diagram analysable.

    ``element_id`` is a :mod:`raid.trace` node id, so the dashboard can jump
    from a suggestion straight to that element's impact view rather than
    matching on names.
    """

    concern: str
    severity: str
    element: str
    element_id: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "concern": self.concern,
            "severity": self.severity,
            "element": self.element,
            "element_id": self.element_id,
            "message": self.message,
        }


@dataclass(frozen=True)
class ConcernScore:
    """How ready the diagram is for one concern.

    ``ready``/``total`` count whatever unit that concern is judged in -
    parameters for test generation, annotation slots for resilience, services
    for fuzzing - which is what makes ``rationale`` able to explain the score
    in the diagram's own terms rather than as an opaque grade.
    """

    concern: str
    score: str
    ratio: float
    ready: int
    total: int
    unit: str
    rationale: str


@dataclass(frozen=True)
class SufficiencyReport:
    """Per-concern readiness scores plus the changes that would improve them."""

    concerns: tuple[ConcernScore, ...] = ()
    suggestions: tuple[Suggestion, ...] = ()

    @property
    def overall(self) -> str:
        """The weakest concern's score.

        Deliberately the minimum rather than an average: a diagram that cannot
        be simulated at all is not rescued by having well-typed parameters, and
        averaging would hide exactly the gap the analyst needs to close first.
        """
        if not self.concerns:
            return "low"
        return min((c.score for c in self.concerns), key=SCORES.index)

    def score_for(self, concern: str) -> ConcernScore:
        """Return the score for one concern.

        Raises:
            KeyError: If the concern is not one of :data:`CONCERNS`.
        """
        for score in self.concerns:
            if score.concern == concern:
                return score
        raise KeyError(f"Unknown concern {concern!r}. Expected one of: {', '.join(CONCERNS)}.")

    def suggestions_for(self, concern: str) -> tuple[Suggestion, ...]:
        """Return only the suggestions raised under one concern."""
        return tuple(s for s in self.suggestions if s.concern == concern)

    def to_table(self) -> list[dict[str, object]]:
        """One row per concern, for a dashboard summary panel."""
        return [
            {
                "concern": score.concern,
                "score": score.score,
                "ready": score.ready,
                "total": score.total,
                "unit": score.unit,
                "rationale": score.rationale,
                "suggestions": len(self.suggestions_for(score.concern)),
            }
            for score in self.concerns
        ]

    def suggestions_to_table(self) -> list[dict[str, object]]:
        """One row per suggestion, most severe first."""
        return [s.to_dict() for s in self.suggestions]
