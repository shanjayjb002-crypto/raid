"""Discrete-event resilience simulation of a flow, using SimPy.

Model
-----
Simulated time is **milliseconds**. Each service is a ``simpy.Resource``, so a
call must acquire its callee before it can run and concurrent load queues
naturally. Each interaction is one timed event whose latency is drawn from a
normal distribution, with the mean and spread taken from the interaction's
annotations when present.

Design notes (for the viva)
---------------------------
**Timeouts use SimPy's condition events.** A call is expressed as
``env.timeout(latency) | env.timeout(budget)`` and whichever fires first wins.
This is the idiomatic SimPy formulation, and it means a timeout is *observed*
by the simulation rather than computed afterwards from a latency number.

**A timed-out call aborts the rest of the path.** If a caller gives up on
``charge``, it does not then go on to ``reserve``; modelling it otherwise would
invent behaviour the diagram never described and would understate the blast
radius of a failure.

**A timed-out call still costs the caller its full timeout.** The caller really
did wait that long before giving up, so the budget is what gets recorded
against end-to-end latency and against the callee's share of the critical path.
Without this a failed service would paradoxically look free.

**Latency is clamped at zero.** A normal distribution is unbounded below, and a
negative delay is both physically meaningless and rejected by SimPy. Clamping
skews the mean slightly upward for very wide distributions; that is preferable
to discarding samples, which would skew it too and cost a retry loop.

**Retries are not modelled.** The DSL parses a ``retry`` annotation and the
graph carries it, but nothing here acts on it yet - a call that times out fails
once. That belongs with the chaos/fuzzing phase.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import simpy

from raid.dsl.model import Interaction
from raid.graph import RaidGraph

from .model import (
    CallStatistics,
    Failure,
    ResilienceReport,
    ServiceContribution,
    SimulationConfig,
    TimeoutViolation,
    UnknownServiceError,
)

__all__ = ["parse_duration", "run_simulation"]


_DURATION_PATTERN = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(ms|s|m|h)?\s*$")

#: Milliseconds per unit, for the duration suffixes the DSL grammar allows.
_UNIT_MS = {"ms": 1.0, "s": 1_000.0, "m": 60_000.0, "h": 3_600_000.0, None: 1.0}


def parse_duration(value: object) -> float:
    """Convert an annotation value into milliseconds of simulated time.

    The DSL deliberately stores ``timeout=2s`` verbatim as the string ``"2s"``,
    leaving interpretation to whichever layer has a time base. This is that
    layer, so this is where the conversion lives.

    Bare numbers are read as milliseconds.

    Args:
        value: A duration string (``"250ms"``, ``"2s"``) or a number.

    Returns:
        The duration in milliseconds.

    Raises:
        ValueError: If the value is not a recognisable duration.
    """
    if isinstance(value, bool):
        raise ValueError(f"Not a duration: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)

    match = _DURATION_PATTERN.match(str(value))
    if not match:
        raise ValueError(
            f"Not a duration: {value!r}. Expected a number optionally suffixed "
            f"with ms, s, m or h (for example '250ms' or '2s')."
        )
    amount, unit = match.groups()
    return float(amount) * _UNIT_MS[unit]


def _annotation_ms(
    interaction: Interaction, key: str, default: float | None
) -> float | None:
    """Read one duration-valued annotation, falling back to ``default``."""
    if key not in interaction.annotations:
        return default
    return parse_duration(interaction.annotations[key])


@dataclass
class _CallRecord:
    """Mutable accumulator for one call position across all replications."""

    step_index: int
    interaction: Interaction
    timeout_ms: float
    executions: int = 0
    timeouts: int = 0
    total_latency: float = 0.0
    blamed_on: set[str] = field(default_factory=set)


def _resolve_latency_parameters(
    interaction: Interaction, config: SimulationConfig
) -> tuple[float, float]:
    """Return the (mean, stddev) to sample this interaction's latency from."""
    mean = _annotation_ms(interaction, "latency_mean", config.default_latency_mean)
    stddev = _annotation_ms(interaction, "latency_stddev", None)
    if stddev is None:
        stddev = (
            config.default_latency_stddev
            if config.default_latency_stddev is not None
            else mean * config.stddev_fraction
        )
    return mean, stddev


def _active_failure(
    failures: Sequence[Failure], service: str, now: float
) -> Failure | None:
    """The first injected failure open on ``service`` at time ``now``."""
    for failure in failures:
        if failure.service == service and failure.is_active_at(now):
            return failure
    return None


def _call_process(
    env: simpy.Environment,
    resources: dict[str, simpy.Resource],
    records: list[_CallRecord],
    config: SimulationConfig,
    failures: Sequence[Failure],
    rng: random.Random,
    outcome: dict,
):
    """SimPy process: walk one execution path once, recording what happened."""
    for record in records:
        interaction = record.interaction
        resource = resources[interaction.target]

        with resource.request() as slot:
            yield slot

            failure = _active_failure(failures, interaction.target, env.now)
            mean, stddev = _resolve_latency_parameters(interaction, config)
            latency = max(0.0, rng.gauss(mean, stddev))

            if failure is not None:
                if failure.mode == "outage":
                    # An unresponsive service never replies. Overshooting the
                    # budget guarantees the deadline wins, deterministically.
                    latency = record.timeout_ms * 10.0
                else:
                    latency *= failure.latency_multiplier

            call = env.timeout(latency)
            deadline = env.timeout(record.timeout_ms)
            fired = yield call | deadline

            record.executions += 1
            if call in fired:
                record.total_latency += latency
                continue

            # Timed out: the caller waited the full budget, then gave up.
            record.timeouts += 1
            record.total_latency += record.timeout_ms
            if failure is not None:
                record.blamed_on.add(failure.service)
            outcome["completed"] = False
            outcome["elapsed"] = env.now
            return

    outcome["completed"] = True
    outcome["elapsed"] = env.now


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list.

    Nearest-rank is used rather than an interpolating definition because it
    always returns a value the simulation actually observed, which keeps a
    reported p95 defensible as "a run that really took this long".
    """
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def _validate_failures(graph: RaidGraph, failures: Iterable[Failure]) -> None:
    known = set(graph.graph.nodes)
    for failure in failures:
        if failure.service not in known:
            available = ", ".join(sorted(known)) or "none"
            raise UnknownServiceError(
                f"Cannot inject a failure on {failure.service!r}: no such service "
                f"in this diagram. Declared services: {available}."
            )


def _build_critical_path(
    records: list[_CallRecord],
) -> tuple[ServiceContribution, ...]:
    """Aggregate per-call latency into a ranked per-service contribution.

    Because a flow is a sequential chain, every executed call lies on the
    critical path by construction. What this adds is the *ranking*: which
    services actually dominate end-to-end latency, and so where a failure hurts
    most or an optimisation would pay off.
    """
    totals: dict[str, list[float]] = {}
    for record in records:
        if record.executions == 0:
            continue
        mean = record.total_latency / record.executions
        totals.setdefault(record.interaction.target, []).append(mean)

    grand_total = sum(sum(means) for means in totals.values())
    contributions = [
        ServiceContribution(
            service=service,
            calls=len(means),
            mean_latency=sum(means),
            share=(sum(means) / grand_total) if grand_total else 0.0,
        )
        for service, means in totals.items()
    ]
    contributions.sort(key=lambda c: (-c.share, c.service))
    return tuple(contributions)


def run_simulation(
    graph: RaidGraph,
    flow_name: str,
    failures: Sequence[Failure] | None = None,
    config: SimulationConfig | None = None,
) -> ResilienceReport:
    """Simulate a flow's default (happy) path and report on its resilience.

    The happy path is the first enumerated path - the one taken when every
    ``alt`` guard holds - which is the scenario a resilience question is
    normally asked about ("what does a payment outage do to a successful
    order?").

    Args:
        graph: The graph model of a parsed diagram.
        flow_name: The flow to simulate.
        failures: Faults to inject. Defaults to none, giving a baseline run.
            (``None`` rather than ``[]`` so the default cannot be mutated by a
            caller and leak into later runs.)
        config: Run settings. Defaults to :class:`SimulationConfig`.

    Returns:
        A :class:`ResilienceReport` with p50/p95 end-to-end latency, the ranked
        critical path, and any timeout violations with their attributed cause.

    Raises:
        UnknownFlowError: If the flow is not declared in the diagram.
        UnknownServiceError: If a failure names an undeclared service.
    """
    settings = config if config is not None else SimulationConfig()
    injected = tuple(failures) if failures else ()
    _validate_failures(graph, injected)

    path = graph.enumerate_paths(flow_name)[0]
    rng = random.Random(settings.seed)

    records = [
        _CallRecord(
            step_index=index,
            interaction=interaction,
            timeout_ms=_annotation_ms(interaction, "timeout", settings.default_timeout),
        )
        for index, interaction in enumerate(path, start=1)
    ]

    elapsed_times: list[float] = []
    completed = 0
    for _ in range(settings.runs):
        # A fresh environment per replication keeps runs statistically
        # independent: queueing left over from one run must not bias the next.
        env = simpy.Environment()
        resources = {
            service: simpy.Resource(env, capacity=settings.capacity)
            for service in graph.graph.nodes
        }
        outcome: dict = {"completed": False, "elapsed": 0.0}
        env.process(
            _call_process(env, resources, records, settings, injected, rng, outcome)
        )
        env.run()

        elapsed_times.append(float(outcome["elapsed"]))
        completed += bool(outcome["completed"])

    ordered = sorted(elapsed_times)
    executed = [r for r in records if r.executions]

    return ResilienceReport(
        flow=flow_name,
        path_id=f"{flow_name}.path1",
        preconditions=path.conditions,
        runs=settings.runs,
        completed_runs=completed,
        timed_out_runs=settings.runs - completed,
        latency_mean=(sum(ordered) / len(ordered)) if ordered else 0.0,
        latency_p50=_percentile(ordered, 0.50),
        latency_p95=_percentile(ordered, 0.95),
        calls=tuple(
            CallStatistics(
                step_index=r.step_index,
                line=r.interaction.line,
                caller=r.interaction.source,
                service=r.interaction.target,
                method=r.interaction.method,
                timeout_ms=r.timeout_ms,
                executions=r.executions,
                timeouts=r.timeouts,
                mean_latency=r.total_latency / r.executions,
            )
            for r in executed
        ),
        timeouts=tuple(
            TimeoutViolation(
                step_index=r.step_index,
                line=r.interaction.line,
                caller=r.interaction.source,
                service=r.interaction.target,
                method=r.interaction.method,
                timeout_ms=r.timeout_ms,
                occurrences=r.timeouts,
                attributed_to=sorted(r.blamed_on)[0] if r.blamed_on else None,
            )
            for r in executed
            if r.timeouts
        ),
        critical_path=_build_critical_path(executed),
        failures=injected,
    )
