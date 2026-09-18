"""SimPy resilience simulation driven by a RAID graph model.

Simulates a flow's happy path as a chain of timed calls against services
modelled as SimPy resources, optionally with injected failures, and reports
end-to-end latency percentiles, the ranked critical path, and any timeout
breaches with their attributed cause.

All times are in milliseconds of simulated time.

Typical use::

    from raid.dsl import parse
    from raid.graph import build_graph
    from raid.simulate import Failure, run_simulation

    graph = build_graph(parse(source_text))

    baseline = run_simulation(graph, "PlaceOrder")
    degraded = run_simulation(
        graph, "PlaceOrder", failures=[Failure("PaymentService")]
    )
"""

from .engine import parse_duration, run_simulation
from .model import (
    CallStatistics,
    Failure,
    FailureMode,
    ResilienceReport,
    ServiceContribution,
    SimulationConfig,
    SimulationError,
    TimeoutViolation,
    UnknownServiceError,
)

__all__ = [
    "CallStatistics",
    "Failure",
    "FailureMode",
    "ResilienceReport",
    "ServiceContribution",
    "SimulationConfig",
    "SimulationError",
    "TimeoutViolation",
    "UnknownServiceError",
    "parse_duration",
    "run_simulation",
]
