"""Construction of a :class:`~raid.graph.model.RaidGraph` from a parsed diagram."""

from __future__ import annotations

import networkx as nx

from raid.dsl.model import Branch, Diagram, Interaction, Step

from .model import RaidGraph, branch_arms

__all__ = ["build_graph"]


def _walk(steps: list[Step], conditions: tuple[str, ...]):
    """Yield every interaction in a step tree with the conditions guarding it.

    Unlike path enumeration this visits each interaction exactly once, whatever
    arm it sits in: an edge in the service graph records that a call *can*
    happen and under what guard, not how many executions reach it.

    The implicit empty arm of an ``alt`` with no ``else`` contributes no
    interactions, so it is skipped here - it matters for enumerating paths, not
    for drawing edges.
    """
    for step in steps:
        if isinstance(step, Interaction):
            yield step, conditions
        elif isinstance(step, Branch):
            for arm_steps, arm_condition in branch_arms(step):
                if arm_steps:
                    yield from _walk(arm_steps, conditions + (arm_condition,))


def build_graph(diagram: Diagram) -> RaidGraph:
    """Build the graph model for a parsed diagram.

    Services become nodes and interactions become directed edges. Each edge
    carries the method name, its typed parameters, any annotations, the flow it
    belongs to, its source line, and the branch conditions it sits under.

    A service that is used in a flow but never declared is still added as a
    node, flagged ``declared=False``. Parsing stays a purely syntactic step, so
    an undeclared service is treated as a diagram *completeness* problem for the
    sufficiency scorer to report - not a hard failure that would stop the rest
    of the analysis running.

    Args:
        diagram: The AST produced by :func:`raid.dsl.parse`.

    Returns:
        The populated graph model.
    """
    graph = nx.MultiDiGraph()

    # Declared services first, so node order follows the diagram as written.
    for service in diagram.services:
        graph.add_node(service.name, declared=True, line=service.line)

    flows = {}
    for flow in diagram.flows:
        flows[flow.name] = flow
        for interaction, conditions in _walk(flow.steps, ()):
            for endpoint in (interaction.source, interaction.target):
                if endpoint not in graph:
                    graph.add_node(endpoint, declared=False, line=None)

            graph.add_edge(
                interaction.source,
                interaction.target,
                method=interaction.method,
                parameters=interaction.parameters,
                annotations=interaction.annotations,
                flow=flow.name,
                line=interaction.line,
                conditions=conditions,
            )

    return RaidGraph(graph=graph, flows=flows)
