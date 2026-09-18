"""Fuzzing loop: mutate, simulate, compare against baseline, write specs."""

from __future__ import annotations

from typing import Sequence

import networkx as nx

from raid.graph import RaidGraph
from raid.simulate import ResilienceReport, SimulationConfig, run_simulation

from .model import ChaosTestSpec
from .mutations import Mutation, MutationKind, apply_mutation, generate_mutations

__all__ = ["fuzz_and_report"]

#: A mutant must inflate p95 latency by at least this factor before a slowdown
#: that never breaches a timeout is still worth reporting.
DEFAULT_LATENCY_REGRESSION_FACTOR = 2.0


def _cycles(graph: RaidGraph) -> set[frozenset[str]]:
    """The set of dependency cycles in a service graph, as node sets.

    Normalised to frozensets so the same cycle discovered from a different
    starting node, or via a parallel edge, compares equal. Baseline cycles are
    subtracted from mutant cycles, because a diagram may legitimately already
    contain one (PlaceOrder's ``rejectOrder`` self-call is a cycle) and only a
    *newly introduced* cycle is a finding.
    """
    return {frozenset(cycle) for cycle in nx.simple_cycles(graph.graph)}


def _timeout_signatures(report: ResilienceReport) -> set[str]:
    return {violation.signature for violation in report.timeouts}


def _describe(report: ResilienceReport) -> str:
    """One sentence summarising how a simulated run behaved."""
    timeouts = (
        "; ".join(v.describe() for v in report.timeouts)
        if report.timeouts
        else "no timeouts"
    )
    return (
        f"{report.completed_runs}/{report.runs} runs completed, "
        f"p50 {report.latency_p50:.0f}ms, p95 {report.latency_p95:.0f}ms, {timeouts}"
    )


def _severity(
    baseline: ResilienceReport,
    mutant: ResilienceReport,
    regressions: Sequence[str],
    latency_ratio: float,
) -> float:
    """Rank a finding so the worst surfaces first.

    A weighted sum of normalised components: losing completed runs is the
    heaviest signal (the flow no longer works at all), then introducing a
    timeout, then latency inflation and structural damage. The weights encode
    a judgement about what an engineer should look at first, not a measurement,
    so they are kept explicit here rather than buried in the comparison.
    """
    baseline_rate = baseline.completed_runs / baseline.runs if baseline.runs else 0.0
    mutant_rate = mutant.completed_runs / mutant.runs if mutant.runs else 0.0

    score = 3.0 * max(0.0, baseline_rate - mutant_rate)
    score += 2.0 * ("timeout" in regressions)
    score += 1.0 * min(max(latency_ratio - 1.0, 0.0), 9.0) / 9.0
    score += 1.0 * ("cycle" in regressions)
    return score


def fuzz_and_report(
    graph: RaidGraph,
    flow_name: str,
    num_mutations: int = 20,
    seed: int = 0,
    targets: Sequence[str] | None = None,
    kinds: Sequence[MutationKind] | None = None,
    config: SimulationConfig | None = None,
    latency_regression_factor: float = DEFAULT_LATENCY_REGRESSION_FACTOR,
) -> list[ChaosTestSpec]:
    """Mutate a flow repeatedly and write a chaos spec for every regression.

    Each mutation is applied, simulated, and compared against a single baseline
    run of the unmutated flow. A spec is written only when the mutant is worse
    than the baseline in at least one of four ways:

    * ``timeout`` - a call breached its budget that did not before
    * ``cascade`` - runs that used to complete no longer do
    * ``latency`` - p95 inflated past ``latency_regression_factor``
    * ``cycle`` - a dependency cycle appeared that the baseline lacked

    Mutations that leave the system no worse produce nothing, which is what
    makes a finding meaningful: removing a dependency, for instance, normally
    makes the flow *faster* and is correctly reported as harmless.

    Args:
        graph: The baseline graph model.
        flow_name: The flow to fuzz.
        num_mutations: How many mutations to try.
        seed: Seeds mutation generation, so a run is reproducible.
        targets: Restrict mutation to these services.
        kinds: Restrict to these mutation operators.
        config: Simulation settings shared by the baseline and every mutant.
        latency_regression_factor: p95 inflation needed to count on its own.

    Returns:
        The chaos specs, most severe first.

    Raises:
        UnknownFlowError: If the flow is not declared.
        MutationTargetError: If no requested target is mutable on this path.
    """
    if num_mutations <= 0:
        return []

    settings = config if config is not None else SimulationConfig()
    baseline = run_simulation(graph, flow_name, config=settings)
    baseline_timeouts = _timeout_signatures(baseline)
    baseline_cycles = _cycles(graph)

    mutations = generate_mutations(
        graph, flow_name, count=num_mutations, seed=seed, targets=targets, kinds=kinds
    )

    specs: list[ChaosTestSpec] = []
    for index, mutation in enumerate(mutations, start=1):
        mutant_graph, failures = apply_mutation(graph, flow_name, mutation)
        mutant = run_simulation(
            mutant_graph, flow_name, failures=failures, config=settings
        )

        regressions: list[str] = []

        if _timeout_signatures(mutant) - baseline_timeouts:
            regressions.append("timeout")
        if mutant.completed_runs < baseline.completed_runs:
            regressions.append("cascade")

        latency_ratio = (
            mutant.latency_p95 / baseline.latency_p95 if baseline.latency_p95 else 1.0
        )
        if latency_ratio >= latency_regression_factor:
            regressions.append("latency")

        new_cycles = _cycles(mutant_graph) - baseline_cycles
        if new_cycles:
            regressions.append("cycle")

        if not regressions:
            continue

        observed = _describe(mutant)
        if new_cycles:
            drawn = "; ".join(" + ".join(sorted(cycle)) for cycle in sorted(new_cycles, key=sorted))
            observed += f", new dependency cycle: {drawn}"

        specs.append(
            ChaosTestSpec(
                id=f"{flow_name}.chaos{index:02d}.{mutation.kind}",
                flow=flow_name,
                path_id=baseline.path_id,
                description=mutation.description,
                mutation=mutation,
                regressions=tuple(regressions),
                baseline_behaviour=_describe(baseline),
                observed_behaviour=observed,
                severity=_severity(baseline, mutant, regressions, latency_ratio),
                baseline_report=baseline,
                mutant_report=mutant,
            )
        )

    specs.sort(key=lambda spec: (-spec.severity, spec.id))
    return specs
