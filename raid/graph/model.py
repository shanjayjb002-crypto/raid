"""Graph model of a parsed RAID diagram, and execution-path enumeration.

Design notes (for the viva)
---------------------------
Two different structures are needed, and conflating them would lose
information, so :class:`RaidGraph` holds both:

*The service graph* (``RaidGraph.graph``) is the architectural view: one node
per service, one edge per interaction. It answers structural questions -
fan-in/fan-out, reachability, which services sit on the most paths - which is
what the threat-flagging and sufficiency phases need.

*The flow definitions* (``RaidGraph.flows``) keep the ordered, nested step tree
that came out of the parser. This is what path enumeration walks.

Why not encode a flow's control flow into the service graph as well? Because a
service appears exactly once as a node but may appear many times within a
single flow, at different points and under different guards. A path through the
service graph is therefore *not* the same thing as a path through a flow, and
treating them as one would generate paths the diagram never described. Instead
each edge records the branch conditions it sits under (see
:func:`~raid.graph.builder.build_graph`), so the graph still represents
alt/else without pretending to be a control-flow graph.

``networkx.MultiDiGraph`` is used rather than ``DiGraph`` because a pair of
services routinely communicates via several distinct methods - in the worked
example ``OrderService -> InventoryService`` carries both ``checkStock`` and
``reserve``. A plain ``DiGraph`` would silently collapse those into one edge and
lose an interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import networkx as nx

from raid.dsl.model import Branch, Flow, Interaction, Step


class GraphError(Exception):
    """Base class for errors raised by the RAID graph layer."""


class UnknownFlowError(GraphError):
    """Raised when a flow is requested by a name the diagram does not declare."""


@dataclass(frozen=True)
class ExecutionPath:
    """One complete way of executing a flow, start to finish.

    A path is the ordered list of interactions that actually run, together with
    the branch conditions that select it. Iterating an ``ExecutionPath`` yields
    its interactions, so it can be used directly wherever an ordered list of
    interactions is expected.

    ``conditions`` is what makes a generated test explainable: a path labelled
    ``("stock_available", "not payment_success")`` documents its own reason for
    existing far better than an index into a list of paths.
    """

    interactions: tuple[Interaction, ...] = ()
    conditions: tuple[str, ...] = ()

    def __iter__(self) -> Iterator[Interaction]:
        return iter(self.interactions)

    def __len__(self) -> int:
        return len(self.interactions)


def branch_arms(branch: Branch) -> list[tuple[list[Step], str]]:
    """Return the mutually exclusive arms of a branch as (steps, condition).

    A branch always has at least two arms. When the diagram writes an explicit
    ``else``, both arms carry the name the author gave them. When it does not,
    the second arm is the *implicit empty arm*: the guard was false and nothing
    ran.

    That implicit arm is the important case. UML gives an ``alt`` fragment with
    a single operand an implicit empty operand for when no guard holds, and the
    same reading is what RAID needs: an ``alt payment_success:`` with no
    ``else`` still describes a system in which payment can fail. Dropping that
    arm would silently drop the failure path from the generated test suite -
    precisely the path most worth testing.
    """
    arms: list[tuple[list[Step], str]] = [(branch.steps, branch.condition)]
    if branch.else_condition is not None:
        arms.append((branch.else_steps, branch.else_condition))
    else:
        arms.append(([], f"not {branch.condition}"))
    return arms


def _expand(steps: list[Step]) -> list[tuple[list[Interaction], list[str]]]:
    """Expand a step list into every (interactions, conditions) combination.

    Walks the steps in order, carrying a working set of partial paths. A plain
    interaction is appended to every partial path; a branch multiplies the set,
    replacing each partial path with one copy per arm. Nested branches recurse,
    so the arms of an inner branch multiply out within the arm of its parent.

    The resulting order is deterministic - document order, and for each branch
    the ``alt`` arm before the ``else`` arm - which keeps generated test suites
    stable between runs and therefore reviewable in a diff.
    """
    partials: list[tuple[list[Interaction], list[str]]] = [([], [])]

    for step in steps:
        if isinstance(step, Interaction):
            partials = [(ints + [step], conds) for ints, conds in partials]
            continue

        expanded: list[tuple[list[Interaction], list[str]]] = []
        for ints, conds in partials:
            for arm_steps, arm_condition in branch_arms(step):
                for arm_ints, arm_conds in _expand(arm_steps):
                    expanded.append(
                        (ints + arm_ints, conds + [arm_condition] + arm_conds)
                    )
        partials = expanded

    return partials


@dataclass
class RaidGraph:
    """A parsed diagram expressed as a service graph plus its flow definitions."""

    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    flows: dict[str, Flow] = field(default_factory=dict)

    def enumerate_paths(self, flow_name: str) -> list[ExecutionPath]:
        """Return every distinct execution path through the named flow.

        Each path is one combination of branch arms, so a flow with two
        independent ``alt``/``else`` fragments yields four paths, and nesting
        multiplies within an arm rather than across the whole flow.

        "Distinct" means a distinct set of branch choices, not a distinct list
        of interactions: two arms may happen to invoke the same methods, but
        they are reached under different conditions and so are different
        executions that each deserve their own test.

        Args:
            flow_name: The name of a flow declared in the diagram.

        Returns:
            The paths in deterministic order: document order, ``alt`` arm
            before ``else`` arm. A flow with no branches yields exactly one
            path.

        Raises:
            UnknownFlowError: If no flow of that name was declared.
        """
        try:
            flow = self.flows[flow_name]
        except KeyError:
            known = ", ".join(sorted(self.flows)) or "none"
            raise UnknownFlowError(
                f"No flow named {flow_name!r} in this diagram. Declared flows: {known}."
            ) from None

        return [
            ExecutionPath(interactions=tuple(ints), conditions=tuple(conds))
            for ints, conds in _expand(flow.steps)
        ]
