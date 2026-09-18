"""The chaos test spec: what a mutation broke, and how it differed from healthy."""

from __future__ import annotations

from dataclasses import dataclass

from raid.simulate import ResilienceReport

from .mutations import Mutation

__all__ = ["REGRESSION_KINDS", "ChaosTestSpec"]


#: The regression signals the engine looks for. ``timeout`` and ``cascade`` are
#: the runtime failures; ``latency`` catches a brownout severe enough to matter
#: without quite breaching a budget; ``cycle`` catches a structural defect that
#: no amount of simulation would reveal.
REGRESSION_KINDS: tuple[str, ...] = ("timeout", "cascade", "latency", "cycle")


@dataclass(frozen=True)
class ChaosTestSpec:
    """A reproducible chaos experiment, written because a mutation broke something.

    Holds three things a human needs in order to act: what to do
    (:attr:`description`), what the healthy system does
    (:attr:`baseline_behaviour`), and what happened instead
    (:attr:`observed_behaviour`). Both full simulation reports are kept so the
    dashboard and the trace module can drill in without re-running anything.
    """

    id: str
    flow: str
    path_id: str
    description: str
    mutation: Mutation
    regressions: tuple[str, ...]
    baseline_behaviour: str
    observed_behaviour: str
    severity: float
    baseline_report: ResilienceReport
    mutant_report: ResilienceReport

    def to_dict(self) -> dict[str, object]:
        """Flatten to one row of scalars for a table or dashboard panel."""
        return {
            "id": self.id,
            "flow": self.flow,
            "path_id": self.path_id,
            "description": self.description,
            "mutation_kind": self.mutation.kind,
            "target": self.mutation.service,
            "regressions": ", ".join(self.regressions),
            "baseline_behaviour": self.baseline_behaviour,
            "observed_behaviour": self.observed_behaviour,
            "severity": round(self.severity, 3),
        }
