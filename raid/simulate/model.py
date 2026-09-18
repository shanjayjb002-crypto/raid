"""Inputs and outputs of the resilience simulation.

Everything here is plain data: what to inject (:class:`Failure`), how to run
(:class:`SimulationConfig`), and what came out (:class:`ResilienceReport`).
Keeping the report a dataclass rather than printed text is what lets the
dashboard, the trace module and the tests all consume the same result.

All times are in milliseconds of simulated time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "CallStatistics",
    "Failure",
    "FailureMode",
    "ResilienceReport",
    "ServiceContribution",
    "SimulationConfig",
    "TimeoutViolation",
    "UnknownServiceError",
]


class SimulationError(Exception):
    """Base class for errors raised by the simulation layer."""


class UnknownServiceError(SimulationError):
    """Raised when a failure targets a service the diagram does not declare.

    Injecting a failure on a misspelt service would otherwise simulate a
    perfectly healthy system and quietly report no problems - the most
    dangerous possible outcome for a resilience tool.
    """


#: ``outage`` - the service stops responding entirely, so every call to it runs
#: past its timeout. ``slow`` - the service still responds, but its latency is
#: multiplied; whether that breaches a timeout depends on the declared budget.
FailureMode = Literal["outage", "slow"]


@dataclass(frozen=True)
class Failure:
    """A fault to inject into one service for part or all of a run.

    Args:
        service: The service that degrades. Must be declared in the diagram.
        mode: ``"outage"`` (stops responding) or ``"slow"`` (responds late).
        start_at: When the fault opens, in milliseconds of simulated time.
        duration: How long it lasts. Defaults to the whole run.
        latency_multiplier: For ``"slow"`` mode, the factor applied to sampled
            latency. Ignored for an outage.
    """

    service: str
    mode: FailureMode = "outage"
    start_at: float = 0.0
    duration: float = math.inf
    latency_multiplier: float = 10.0

    def is_active_at(self, time: float) -> bool:
        """Whether this fault is open at the given moment of simulated time."""
        return self.start_at <= time < self.start_at + self.duration

    def describe(self) -> str:
        window = "whole run" if math.isinf(self.duration) else f"{self.start_at:g}-{self.start_at + self.duration:g}ms"
        return f"{self.service} ({self.mode}, {window})"


@dataclass(frozen=True)
class SimulationConfig:
    """Knobs for a simulation run.

    Defaults are deliberately modest rather than realistic: a 50ms mean call
    against a 1000ms timeout means a healthy system never trips, so any timeout
    in a report is attributable to something the analyst actually injected or
    annotated.

    Args:
        runs: Independent replications. Percentiles are taken across these.
        seed: Seeds a private RNG, so results are reproducible without
            disturbing global ``random`` state.
        default_latency_mean: Mean call latency when unannotated, in ms.
        default_latency_stddev: Standard deviation when unannotated. ``None``
            derives it as ``stddev_fraction`` of whatever mean applies, so an
            annotated slow call gets proportionally more jitter.
        stddev_fraction: Used when ``default_latency_stddev`` is ``None``.
        default_timeout: Call timeout when unannotated, in ms.
        capacity: Concurrent requests each service resource will admit.
    """

    runs: int = 200
    seed: int = 0
    default_latency_mean: float = 50.0
    default_latency_stddev: float | None = None
    stddev_fraction: float = 0.2
    default_timeout: float = 1000.0
    capacity: int = 1


@dataclass(frozen=True)
class CallStatistics:
    """Observed behaviour of one call position in the simulated path."""

    step_index: int
    caller: str
    service: str
    method: str
    timeout_ms: float
    executions: int
    timeouts: int
    mean_latency: float

    @property
    def signature(self) -> str:
        return f"{self.caller} -> {self.service} : {self.method}"


@dataclass(frozen=True)
class TimeoutViolation:
    """A call that ran past its timeout budget during the simulation."""

    step_index: int
    caller: str
    service: str
    method: str
    timeout_ms: float
    occurrences: int
    attributed_to: str | None

    @property
    def signature(self) -> str:
        return f"{self.caller} -> {self.service} : {self.method}"

    def describe(self) -> str:
        cause = f" caused by the {self.attributed_to} failure" if self.attributed_to else ""
        return (
            f"{self.signature} exceeded its {self.timeout_ms:g}ms timeout "
            f"in {self.occurrences} run(s){cause}"
        )


@dataclass(frozen=True)
class ServiceContribution:
    """How much one service contributed to end-to-end latency."""

    service: str
    calls: int
    mean_latency: float
    share: float


@dataclass(frozen=True)
class ResilienceReport:
    """The result of simulating one execution path under a set of failures."""

    flow: str
    path_id: str
    preconditions: tuple[str, ...]
    runs: int
    completed_runs: int
    timed_out_runs: int
    latency_mean: float
    latency_p50: float
    latency_p95: float
    calls: tuple[CallStatistics, ...] = ()
    timeouts: tuple[TimeoutViolation, ...] = ()
    critical_path: tuple[ServiceContribution, ...] = ()
    failures: tuple[Failure, ...] = ()

    @property
    def timed_out(self) -> bool:
        """Whether any call breached its timeout during the simulation."""
        return bool(self.timeouts)

    @property
    def call_sequence(self) -> tuple[str, ...]:
        """The method names actually executed, in order."""
        return tuple(call.method for call in self.calls)

    def to_dict(self) -> dict[str, object]:
        """Flatten to one row of scalars for a table or dashboard panel."""
        return {
            "flow": self.flow,
            "path_id": self.path_id,
            "preconditions": " AND ".join(self.preconditions) or "(none)",
            "runs": self.runs,
            "completed_runs": self.completed_runs,
            "timed_out_runs": self.timed_out_runs,
            "latency_mean": round(self.latency_mean, 2),
            "latency_p50": round(self.latency_p50, 2),
            "latency_p95": round(self.latency_p95, 2),
            "timed_out": self.timed_out,
            "timeouts": "; ".join(v.describe() for v in self.timeouts) or "(none)",
            "critical_path": " > ".join(
                f"{c.service} {c.share:.0%}" for c in self.critical_path
            )
            or "(none)",
            "failures": "; ".join(f.describe() for f in self.failures) or "(none)",
        }
