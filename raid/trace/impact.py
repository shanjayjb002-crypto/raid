"""The cross-concern impact graph: diagram elements to everything they produced.

Shape of the graph
------------------
A ``networkx.DiGraph`` whose edges all run *downstream*, from diagram element
to generated artifact::

    flow ─────────┐
                  ├──> interaction ──> functional test
    service ──────┘         │      └──> boundary test
       │                    └──────└──> resilience finding
       └────────────────────────────> chaos spec

Both endpoints of a call point at it, tagged ``caller`` or ``callee``, so a
timeout is reachable from the service that suffered it as well as the one that
caused it.

Because every edge runs the same way, the two required queries are just the two
directions of reachability: ``downstream`` is ``nx.descendants``, ``sources`` is
``nx.ancestors`` filtered to diagram kinds. Nothing else needs to be maintained,
and the two directions cannot drift apart - which is what the round-trip test
asserts.

Why a graph rather than a dict of sets
--------------------------------------
Reachability here is genuinely transitive: a service reaches a test case only
*through* an interaction. An index would have to precompute and store every
transitive closure, and then keep them consistent. Letting networkx walk the
edges keeps one representation with no derived state to invalidate, and makes
the intermediate hops (which interaction linked this service to this test?)
available rather than flattened away.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import networkx as nx

from raid.dsl.model import Branch, Flow, Interaction, Step
from raid.fuzz import ChaosTestSpec
from raid.graph import RaidGraph
from raid.simulate import ResilienceReport
from raid.testgen import BoundaryTestCase, FunctionalTestCase, GeneratedTestCase

from .model import (
    ARTIFACT_KINDS,
    ImpactSet,
    SourceSet,
    UnknownElementError,
    artifact_id,
    flow_id,
    interaction_id,
    service_id,
)

__all__ = ["ImpactGraph", "build_impact_graph"]


def _walk(steps: Sequence[Step]) -> Iterable[Interaction]:
    """Yield every interaction in a step tree, both arms of every branch."""
    for step in steps:
        if isinstance(step, Interaction):
            yield step
        elif isinstance(step, Branch):
            yield from _walk(step.steps)
            yield from _walk(step.else_steps)


class ImpactGraph:
    """Bidirectional index from diagram elements to generated artifacts."""

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph

    # -- queries ----------------------------------------------------------

    def _require(self, node_id: str) -> None:
        if node_id not in self.graph:
            raise UnknownElementError(
                f"{node_id!r} is not in this impact graph. Node ids are built by "
                f"service_id(), flow_id(), interaction_id() or artifact_id()."
            )

    def downstream(self, node_id: str) -> ImpactSet:
        """Return everything the given diagram element produced.

        Reachability is transitive, so querying a service returns the artifacts
        of every call it participates in, and querying a flow returns the whole
        diagram's output.

        Args:
            node_id: A node id, from :func:`service_id`, :func:`flow_id` or
                :func:`interaction_id`.

        Returns:
            The downstream artifacts, grouped by concern.

        Raises:
            UnknownElementError: If the node is not in the graph.
        """
        self._require(node_id)

        payloads: dict[str, list] = {kind: [] for kind in ARTIFACT_KINDS}
        interactions: list[Interaction] = []
        artifact_ids: list[str] = []

        for reached in nx.descendants(self.graph, node_id):
            data = self.graph.nodes[reached]
            kind = data["kind"]
            if kind == "interaction":
                interactions.append(data["payload"])
            elif kind in payloads:
                payloads[kind].append(data["payload"])
                artifact_ids.append(reached)

        interactions.sort(key=lambda i: i.line)
        return ImpactSet(
            interactions=tuple(interactions),
            functional_tests=tuple(sorted(payloads["functional_test"], key=lambda c: c.id)),
            boundary_tests=tuple(sorted(payloads["boundary_test"], key=lambda c: c.id)),
            resilience_findings=tuple(
                sorted(payloads["resilience_finding"], key=lambda v: v.step_index)
            ),
            chaos_specs=tuple(sorted(payloads["chaos_spec"], key=lambda s: s.id)),
            artifact_ids=tuple(sorted(artifact_ids)),
        )

    def sources(self, node_id: str) -> SourceSet:
        """Return exactly the diagram elements that produced the given artifact.

        Args:
            node_id: An artifact node id, from :func:`artifact_id`.

        Returns:
            The services, flows and interactions upstream of it.

        Raises:
            UnknownElementError: If the node is not in the graph.
        """
        self._require(node_id)

        services: list[str] = []
        flows: list[str] = []
        interactions: list[Interaction] = []

        for reached in nx.ancestors(self.graph, node_id):
            data = self.graph.nodes[reached]
            if data["kind"] == "service":
                services.append(data["payload"])
            elif data["kind"] == "flow":
                flows.append(data["payload"])
            elif data["kind"] == "interaction":
                interactions.append(data["payload"])

        interactions.sort(key=lambda i: i.line)
        return SourceSet(
            services=tuple(sorted(services)),
            flows=tuple(sorted(flows)),
            interactions=tuple(interactions),
        )

    def to_table(self) -> list[dict[str, object]]:
        """One row per diagram element, counting what it produced.

        This is the coverage view: an element with zeros across the board is
        one the analysis never reached, which is a finding in its own right for
        the sufficiency scorer.
        """
        rows = []
        for node, data in self.graph.nodes(data=True):
            if data["kind"] not in ("service", "flow", "interaction"):
                continue
            impact = self.downstream(node)
            rows.append(
                {
                    "element": data["label"],
                    "kind": data["kind"],
                    "functional_tests": len(impact.functional_tests),
                    "boundary_tests": len(impact.boundary_tests),
                    "resilience_findings": len(impact.resilience_findings),
                    "chaos_specs": len(impact.chaos_specs),
                    "total": impact.total_artifacts,
                }
            )
        rows.sort(key=lambda row: (-row["total"], row["kind"], row["element"]))
        return rows


def build_impact_graph(
    raid_graph: RaidGraph,
    test_cases: Sequence[GeneratedTestCase] = (),
    resilience_report: ResilienceReport | None = None,
    chaos_specs: Sequence[ChaosTestSpec] = (),
) -> ImpactGraph:
    """Link every diagram element to the artifacts produced from it.

    Artifacts are anchored to interactions by ``(flow, source line)``. Anything
    that cannot be anchored - a test case naming a flow this graph does not
    contain - is skipped rather than creating a dangling node, so the graph
    stays a faithful index of this diagram.

    Args:
        raid_graph: The graph model of the parsed diagram.
        test_cases: Functional and/or boundary cases, in any mix.
        resilience_report: A simulation report, or ``None`` if none has run.
        chaos_specs: Chaos specs from the fuzzer.

    Returns:
        The populated :class:`ImpactGraph`.
    """
    graph = nx.DiGraph()

    for service in raid_graph.graph.nodes:
        graph.add_node(service_id(service), kind="service", label=service, payload=service)

    # Diagram structure: a flow contains its interactions, and both endpoints
    # of a call point at it.
    for flow_name, flow in raid_graph.flows.items():
        graph.add_node(flow_id(flow_name), kind="flow", label=flow_name, payload=flow_name)

        for interaction in _walk(flow.steps):
            node = interaction_id(flow_name, interaction.line)
            graph.add_node(
                node,
                kind="interaction",
                label=f"{interaction.source} -> {interaction.target} : {interaction.method}",
                payload=interaction,
            )
            graph.add_edge(flow_id(flow_name), node, relation="contains")
            graph.add_edge(service_id(interaction.source), node, relation="caller")
            graph.add_edge(service_id(interaction.target), node, relation="callee")

    def _link(source_node: str, target_node: str, relation: str) -> None:
        if source_node in graph:
            graph.add_edge(source_node, target_node, relation=relation)

    for case in test_cases:
        kind = "boundary_test" if isinstance(case, BoundaryTestCase) else "functional_test"
        node = artifact_id(case)
        graph.add_node(node, kind=kind, label=case.id, payload=case)

        owning = (
            [case.interaction]
            if isinstance(case, BoundaryTestCase)
            else list(case.interactions)
        )
        for interaction in owning:
            _link(interaction_id(case.flow, interaction.line), node, "exercises")

    if resilience_report is not None:
        for violation in resilience_report.timeouts:
            node = artifact_id(violation)
            graph.add_node(
                node,
                kind="resilience_finding",
                label=violation.describe(),
                payload=violation,
            )
            _link(
                interaction_id(resilience_report.flow, violation.line), node, "observed_on"
            )

    for spec in chaos_specs:
        node = artifact_id(spec)
        graph.add_node(node, kind="chaos_spec", label=spec.description, payload=spec)

        # Every chaos spec is produced by the service its mutation targets.
        _link(service_id(spec.mutation.service), node, "mutates")
        # A structural mutation additionally names the call it rewrote, which
        # is what makes the caller reachable from it too.
        if spec.mutation.method is not None:
            _link(interaction_id(spec.flow, spec.mutation.line), node, "mutates")

    return ImpactGraph(graph)
