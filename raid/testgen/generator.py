"""Generation of functional and boundary test cases from a :class:`RaidGraph`."""

from __future__ import annotations

from typing import Iterator

from raid.dsl.model import Interaction
from raid.graph import ExecutionPath, RaidGraph

from .boundary import ParameterClassifier, get_classifier
from .model import BoundaryTestCase, FunctionalTestCase

__all__ = ["generate_boundary_cases", "generate_functional_cases"]


def _call_signature(interaction: Interaction) -> str:
    """Render an interaction as the call an assertion would look for."""
    return f"{interaction.source} -> {interaction.target} : {interaction.method}"


def _paths(
    graph: RaidGraph, flow_name: str | None
) -> Iterator[tuple[str, int, ExecutionPath]]:
    """Yield ``(flow_name, 1-based path number, path)`` for the requested flows.

    Passing ``flow_name=None`` covers every flow in the diagram, which is what
    the dashboard needs to build a whole-diagram suite in one call. Naming a
    flow that does not exist propagates ``UnknownFlowError`` from the graph
    layer rather than silently producing an empty suite.
    """
    names = [flow_name] if flow_name is not None else list(graph.flows)
    for name in names:
        for index, path in enumerate(graph.enumerate_paths(name), start=1):
            yield name, index, path


def generate_functional_cases(
    graph: RaidGraph, flow_name: str | None = None
) -> list[FunctionalTestCase]:
    """Generate one functional test case per execution path.

    Each case asserts that the path's calls happen, in order, given the branch
    conditions that select it. Because path enumeration already treats each
    combination of branch arms as distinct, every branch in the diagram
    necessarily gets its own case.

    Args:
        graph: The graph model of a parsed diagram.
        flow_name: The flow to generate for, or ``None`` for every flow.

    Returns:
        Cases in deterministic order: flow order, then path order.

    Raises:
        UnknownFlowError: If ``flow_name`` names no declared flow.
    """
    cases = []
    for flow, index, path in _paths(graph, flow_name):
        path_id = f"{flow}.path{index}"
        cases.append(
            FunctionalTestCase(
                id=f"{path_id}.functional",
                flow=flow,
                path_id=path_id,
                path_index=index,
                preconditions=path.conditions,
                interactions=path.interactions,
                expected_call_sequence=tuple(_call_signature(i) for i in path),
            )
        )
    return cases


def generate_boundary_cases(
    graph: RaidGraph,
    flow_name: str | None = None,
    classifier: ParameterClassifier | None = None,
) -> list[BoundaryTestCase]:
    """Generate boundary-value cases for every typed parameter on every path.

    A parameter is classified from its name and declared type (see
    :mod:`raid.testgen.boundary`), and each boundary its class implies becomes
    one test case attached to the execution path the call sits on.

    Parameters are deliberately re-generated per path rather than once per
    interaction: the same call reached under different branch conditions is a
    different test, because the state the system is in when it receives the
    edge-case value differs. The path's conditions are carried onto each case
    as its preconditions.

    Args:
        graph: The graph model of a parsed diagram.
        flow_name: The flow to generate for, or ``None`` for every flow.
        classifier: An alternative classifier, mainly for testing. Defaults to
            the shared cached one.

    Returns:
        Cases in deterministic order: flow, path, step, parameter, boundary.

    Raises:
        UnknownFlowError: If ``flow_name`` names no declared flow.
    """
    model = classifier if classifier is not None else get_classifier()

    cases = []
    for flow, index, path in _paths(graph, flow_name):
        path_id = f"{flow}.path{index}"
        for step_index, interaction in enumerate(path, start=1):
            for parameter in interaction.parameters:
                proposal = model.propose(parameter)
                for candidate in proposal.candidates:
                    cases.append(
                        BoundaryTestCase(
                            id=(
                                f"{path_id}.step{step_index}.{interaction.method}"
                                f".{parameter.name}.{candidate.case}"
                            ),
                            flow=flow,
                            path_id=path_id,
                            path_index=index,
                            preconditions=path.conditions,
                            step_index=step_index,
                            interaction=interaction,
                            parameter=parameter,
                            inferred_class=proposal.inferred_class,
                            confidence=proposal.confidence,
                            boundary_case=candidate.case,
                            value=candidate.value,
                        )
                    )
    return cases
