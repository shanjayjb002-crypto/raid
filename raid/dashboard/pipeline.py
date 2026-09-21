"""The end-to-end analysis pipeline behind the dashboard.

Deliberately free of any Streamlit import. The UI layer is only rendering, so
everything that could be wrong lives here and is unit tested without a browser
- which is what PROJECT_RULES.md asks of any module wired into the dashboard.

Failure policy
--------------
A parse error propagates: the UI must show it prominently, and there is nothing
useful to render without a diagram. *Everything else* degrades into a warning
and an empty section. A live demo should never lose the whole page because one
stage of seven could not run - an undeclared failure target or a flow with
nothing to mutate should cost that panel, not the presentation.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Sequence

from raid.dsl import Diagram, parse
from raid.fuzz import ChaosTestSpec, MutationTargetError, fuzz_and_report
from raid.graph import RaidGraph, build_graph
from raid.simulate import (
    Failure,
    ResilienceReport,
    SimulationConfig,
    SimulationError,
    run_simulation,
)
from raid.sufficiency import SufficiencyReport, score_sufficiency
from raid.testgen import (
    BoundaryTestCase,
    FunctionalTestCase,
    generate_boundary_cases,
    generate_functional_cases,
)
from raid.trace import ImpactGraph, ImpactSet, build_impact_graph, service_id

__all__ = ["Analysis", "AnalysisOptions", "analyse", "rows_to_csv"]


@dataclass(frozen=True)
class AnalysisOptions:
    """Settings the dashboard exposes as controls.

    Args:
        runs: Simulation replications.
        mutations: How many mutations the fuzzer should try.
        seed: Seeds both simulation and mutation, so a demo is repeatable.
        flow_name: Which flow to analyse. ``None`` selects the first declared.
        failure_service: A service to fail during simulation, or ``None`` for
            a healthy baseline.
    """

    runs: int = 100
    mutations: int = 15
    seed: int = 0
    flow_name: str | None = None
    failure_service: str | None = None


@dataclass(frozen=True)
class Analysis:
    """Everything the seven phases produced for one diagram."""

    source: str
    diagram: Diagram
    graph: RaidGraph
    sufficiency: SufficiencyReport
    impact: ImpactGraph
    flow_name: str | None = None
    flow_names: tuple[str, ...] = ()
    functional: tuple[FunctionalTestCase, ...] = ()
    boundary: tuple[BoundaryTestCase, ...] = ()
    report: ResilienceReport | None = None
    chaos: tuple[ChaosTestSpec, ...] = ()
    warnings: tuple[str, ...] = ()

    def focus_options(self) -> tuple[str, ...]:
        """Service names the user can filter by."""
        return tuple(sorted(self.graph.graph.nodes))

    def slice_for(self, service: str | None) -> ImpactSet:
        """Return everything produced by one service, or everything at all.

        Filtering is delegated to the impact graph rather than re-filtering the
        lists by name, so the dashboard shows exactly what the traceability
        layer claims - one source of truth for "what did this service produce".
        """
        if service is None:
            return ImpactSet(
                functional_tests=self.functional,
                boundary_tests=self.boundary,
                resilience_findings=self.report.timeouts if self.report else (),
                chaos_specs=self.chaos,
            )
        node = service_id(service)
        if node not in self.impact.graph:
            return ImpactSet()
        return self.impact.downstream(node)

    def test_case_rows(self, service: str | None = None) -> list[dict[str, object]]:
        """Functional and boundary cases as one table, ready for CSV."""
        selected = self.slice_for(service)
        return [
            *(case.to_dict() for case in selected.functional_tests),
            *(case.to_dict() for case in selected.boundary_tests),
        ]

    def chaos_rows(self, service: str | None = None) -> list[dict[str, object]]:
        """Chaos specs as a table."""
        return [spec.to_dict() for spec in self.slice_for(service).chaos_specs]


def rows_to_csv(rows: Sequence[dict[str, object]]) -> str:
    """Render table rows as CSV text.

    Handles ragged rows: functional and boundary cases carry different columns,
    and the union is written with blanks rather than dropping either kind.
    """
    if not rows:
        return ""

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, restval="", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def analyse(source: str, options: AnalysisOptions | None = None) -> Analysis:
    """Run every RAID phase over DSL source text.

    Args:
        source: The diagram as DSL text.
        options: Run settings. Defaults to :class:`AnalysisOptions`.

    Returns:
        The combined :class:`Analysis`. Stages that could not run leave their
        section empty and add an entry to ``warnings``.

    Raises:
        DslSyntaxError: If the source does not parse. The caller is expected to
            show the message, which is already written for a human.
    """
    settings = options if options is not None else AnalysisOptions()
    warnings: list[str] = []

    diagram = parse(source)
    graph = build_graph(diagram)
    sufficiency = score_sufficiency(graph)

    flow_names = tuple(graph.flows)
    flow_name = settings.flow_name or (flow_names[0] if flow_names else None)
    if settings.flow_name and settings.flow_name not in flow_names:
        warnings.append(
            f"Flow {settings.flow_name!r} is not in this diagram; "
            f"analysing {flow_names[0]!r} instead." if flow_names else
            f"Flow {settings.flow_name!r} is not in this diagram."
        )
        flow_name = flow_names[0] if flow_names else None

    if flow_name is None:
        warnings.append(
            "This diagram declares no flows, so there is nothing to generate, "
            "simulate or fuzz. Add a `flow <Name>:` block with at least one "
            "interaction."
        )
        return Analysis(
            source=source,
            diagram=diagram,
            graph=graph,
            sufficiency=sufficiency,
            impact=build_impact_graph(graph),
            flow_names=flow_names,
            warnings=tuple(warnings),
        )

    functional = tuple(generate_functional_cases(graph, flow_name))
    boundary = tuple(generate_boundary_cases(graph, flow_name))

    config = SimulationConfig(runs=settings.runs, seed=settings.seed)

    failures: list[Failure] = []
    if settings.failure_service:
        if settings.failure_service in graph.graph.nodes:
            failures.append(Failure(settings.failure_service))
        else:
            warnings.append(
                f"Cannot fail {settings.failure_service!r}: it is not a service in "
                f"this diagram. Showing the healthy baseline instead."
            )

    try:
        report = run_simulation(graph, flow_name, failures=failures, config=config)
    except SimulationError as exc:  # pragma: no cover - guarded by the check above
        warnings.append(f"Simulation could not run: {exc}")
        report = None

    try:
        chaos = tuple(
            fuzz_and_report(
                graph,
                flow_name,
                num_mutations=settings.mutations,
                seed=settings.seed,
                config=config,
            )
        )
    except MutationTargetError as exc:
        warnings.append(f"Fuzzing found nothing to mutate: {exc}")
        chaos = ()

    impact = build_impact_graph(graph, [*functional, *boundary], report, chaos)

    return Analysis(
        source=source,
        diagram=diagram,
        graph=graph,
        sufficiency=sufficiency,
        impact=impact,
        flow_name=flow_name,
        flow_names=flow_names,
        functional=functional,
        boundary=boundary,
        report=report,
        chaos=chaos,
        warnings=tuple(warnings),
    )
