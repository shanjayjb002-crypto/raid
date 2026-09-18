"""Mutation operators over a :class:`RaidGraph`, and their random generation.

The four operators fall into two families, and the distinction matters:

*Runtime* mutations (``kill_service``, ``inflate_latency``) leave the diagram
alone and express themselves as faults injected into the simulation. They ask
"what if this service misbehaved at run time?"

*Structural* mutations (``remove_edge``, ``inject_cycle``) rewrite the flow
itself and produce a new graph. They ask "what if the architecture were
different?"

Both are expressed through one :class:`Mutation` type whose :func:`apply_mutation`
returns ``(graph_to_simulate, failures_to_inject)``, so the engine can treat
every mutation identically no matter which family it came from.

Why random search rather than a genetic algorithm
-------------------------------------------------
Each individual here is a *single* mutation drawn from four operators over a
handful of services - a search space small enough to sample directly, and one
with nothing to recombine, since two mutations of different kinds have no
meaningful crossover. A GA earns its keep once individuals are *combinations*
of mutations and the fitness landscape rewards finding interacting faults (a
brownout that only breaks things while another service is already degraded).
That is the natural next step for this module, and the point at which ``deap``
would start paying for itself.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Literal, Sequence

from raid.dsl.model import Branch, Diagram, Flow, Interaction, Service, Step
from raid.graph import RaidGraph, build_graph
from raid.simulate import Failure

__all__ = [
    "MUTATION_KINDS",
    "Mutation",
    "MutationKind",
    "MutationTargetError",
    "apply_mutation",
    "generate_mutations",
]


class MutationTargetError(Exception):
    """Raised when no useful mutation can be built for the requested target.

    Surfaced rather than swallowed: silently returning an empty mutation set
    for a misspelt service would let a fuzzing run report "no findings" about
    a service it never actually touched.
    """


MutationKind = Literal["kill_service", "remove_edge", "inject_cycle", "inflate_latency"]

MUTATION_KINDS: tuple[MutationKind, ...] = (
    "kill_service",
    "remove_edge",
    "inject_cycle",
    "inflate_latency",
)

#: Outage windows sampled by the generator, in milliseconds. The short 20ms
#: entry is deliberate: it usually closes before the call it targets even
#: starts, so some kills are legitimately harmless and the detector has to
#: discriminate rather than flag everything.
_DURATIONS: tuple[float, ...] = (20.0, 1_000.0, 2_500.0, math.inf)

_MULTIPLIERS: tuple[float, ...] = (2.0, 5.0, 20.0, 50.0, 100.0)


@dataclass(frozen=True)
class Mutation:
    """One fault or structural change to apply to a diagram.

    Args:
        kind: Which operator this is.
        service: The service being mutated. For structural operators this is
            the *callee* of the interaction being rewritten.
        method: The interaction's method, for structural operators.
        line: The source line of the interaction being rewritten. Interactions
            are addressed by line because the grammar puts exactly one on each,
            which makes it a stable identity that survives rebuilding the tree.
        start_at: When a runtime fault opens, in milliseconds.
        duration: How long a runtime fault lasts.
        multiplier: Latency factor for ``inflate_latency``.
    """

    kind: MutationKind
    service: str
    method: str | None = None
    line: int = 0
    start_at: float = 0.0
    duration: float = math.inf
    multiplier: float = 10.0

    @property
    def description(self) -> str:
        """A one-line description in the language of a chaos experiment."""
        window = "the whole run" if math.isinf(self.duration) else f"{self.duration:g}ms"
        if self.kind == "kill_service":
            return f"kill {self.service} for {window}"
        if self.kind == "inflate_latency":
            return f"inflate {self.service} latency x{self.multiplier:g} for {window}"
        if self.kind == "remove_edge":
            return f"remove dependency on {self.service} : {self.method}"
        return f"inject cycle {self.service} -> caller of {self.method}"


# --------------------------------------------------------------------------
# Structural rewriting
# --------------------------------------------------------------------------


def _rewrite(steps: list[Step], line: int, replacement) -> list[Step]:
    """Rebuild a step tree, replacing the interaction on ``line``.

    ``replacement`` maps the matched interaction to the list of steps that
    should stand in its place - empty to delete it, or several to expand it.

    New objects are built throughout rather than mutating in place: the caller's
    graph must survive being used as the base for many mutants.
    """
    rewritten: list[Step] = []
    for step in steps:
        if isinstance(step, Interaction):
            if step.line == line:
                rewritten.extend(replacement(step))
            else:
                rewritten.append(step)
        else:
            rewritten.append(
                Branch(
                    condition=step.condition,
                    steps=_rewrite(step.steps, line, replacement),
                    else_condition=step.else_condition,
                    else_steps=_rewrite(step.else_steps, line, replacement),
                    line=step.line,
                )
            )
    return rewritten


def _rebuild(graph: RaidGraph, flow_name: str, steps: list[Step]) -> RaidGraph:
    """Build a fresh graph in which ``flow_name`` has the given steps."""
    services = [
        Service(name=name, line=data.get("line") or 0)
        for name, data in graph.graph.nodes(data=True)
        if data.get("declared")
    ]
    flows = [
        replace(flow, steps=steps) if name == flow_name else flow
        for name, flow in graph.flows.items()
    ]
    return build_graph(Diagram(services=services, flows=flows))


def apply_mutation(
    graph: RaidGraph, flow_name: str, mutation: Mutation
) -> tuple[RaidGraph, tuple[Failure, ...]]:
    """Express a mutation as a graph to simulate plus faults to inject.

    Runtime mutations return the original graph untouched alongside a failure;
    structural mutations return a rebuilt graph and no failures.

    Args:
        graph: The baseline graph.
        flow_name: The flow the mutation applies to.
        mutation: The mutation to apply.

    Returns:
        ``(graph, failures)`` ready to hand to ``run_simulation``.
    """
    if mutation.kind == "kill_service":
        return graph, (
            Failure(
                service=mutation.service,
                mode="outage",
                start_at=mutation.start_at,
                duration=mutation.duration,
            ),
        )

    if mutation.kind == "inflate_latency":
        return graph, (
            Failure(
                service=mutation.service,
                mode="slow",
                start_at=mutation.start_at,
                duration=mutation.duration,
                latency_multiplier=mutation.multiplier,
            ),
        )

    flow = graph.flows[flow_name]

    if mutation.kind == "remove_edge":
        steps = _rewrite(flow.steps, mutation.line, lambda _: [])
        return _rebuild(graph, flow_name, steps), ()

    # inject_cycle: follow the call with a reply in the opposite direction, so
    # the two services now depend on each other.
    def _with_callback(interaction: Interaction) -> list[Step]:
        callback = Interaction(
            source=interaction.target,
            target=interaction.source,
            method=f"{interaction.method}Callback",
            parameters=[],
            annotations={},
            line=interaction.line,
        )
        return [interaction, callback]

    steps = _rewrite(flow.steps, mutation.line, _with_callback)
    return _rebuild(graph, flow_name, steps), ()


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _callees(graph: RaidGraph, flow_name: str) -> dict[str, list[Interaction]]:
    """Map each service *receiving* a call on the happy path to those calls.

    Only callees are mutable targets. Phase 4 showed that failing a service
    which merely originates calls changes nothing, so offering it as a target
    would generate mutations guaranteed to find nothing.
    """
    by_service: dict[str, list[Interaction]] = {}
    for interaction in graph.enumerate_paths(flow_name)[0]:
        by_service.setdefault(interaction.target, []).append(interaction)
    return by_service


def _draw(
    kind: MutationKind, service: str, interaction: Interaction, rng: random.Random
) -> Mutation:
    """Build one mutation, setting only the fields its kind actually uses.

    Leaving irrelevant fields at their defaults is what makes two behaviourally
    identical mutations compare equal: a ``kill_service`` ignores
    ``multiplier``, so sampling one would otherwise produce several distinct
    objects describing exactly the same experiment.
    """
    if kind == "kill_service":
        return Mutation(kind=kind, service=service, duration=rng.choice(_DURATIONS))
    if kind == "inflate_latency":
        return Mutation(
            kind=kind,
            service=service,
            duration=rng.choice(_DURATIONS),
            multiplier=rng.choice(_MULTIPLIERS),
        )
    return Mutation(
        kind=kind,
        service=service,
        method=interaction.method,
        line=interaction.line,
    )


def generate_mutations(
    graph: RaidGraph,
    flow_name: str,
    count: int,
    seed: int = 0,
    targets: Sequence[str] | None = None,
    kinds: Sequence[MutationKind] | None = None,
) -> list[Mutation]:
    """Randomly generate ``count`` mutations against a flow's happy path.

    Args:
        graph: The baseline graph.
        flow_name: The flow to mutate.
        count: How many mutations to produce.
        seed: Seeds a private RNG so a fuzzing run is reproducible.
        targets: Restrict mutation to these services. Defaults to every
            service that receives a call on the happy path.
        kinds: Restrict to these operators. Defaults to all four.

    Returns:
        Up to ``count`` distinct mutations, in generation order. Fewer are
        returned when the mutation space is smaller than ``count`` - there is
        only one way to remove a given dependency, for instance.

    Raises:
        MutationTargetError: If no requested target receives a call on this
            path, which would otherwise yield mutations that cannot bite.
    """
    candidates = _callees(graph, flow_name)

    if targets is not None:
        usable = {name: calls for name, calls in candidates.items() if name in targets}
        if not usable:
            known = ", ".join(sorted(candidates)) or "none"
            requested = ", ".join(targets)
            raise MutationTargetError(
                f"No mutable target among [{requested}] for flow {flow_name!r}. "
                f"A service must receive a call on the happy path to be worth "
                f"mutating; this path calls: {known}."
            )
        candidates = usable

    operators = tuple(kinds) if kinds else MUTATION_KINDS
    rng = random.Random(seed)
    services = sorted(candidates)

    seen: set[Mutation] = set()
    mutations: list[Mutation] = []
    # Sampling is with replacement, so draws are retried until `count` distinct
    # mutations exist or the space is exhausted - re-simulating an identical
    # fault only costs time and fills the report with duplicate specs.
    for _ in range(max(count * 20, 200)):
        if len(mutations) == count:
            break
        service = rng.choice(services)
        interaction = rng.choice(candidates[service])
        mutation = _draw(rng.choice(operators), service, interaction, rng)
        if mutation in seen:
            continue
        seen.add(mutation)
        mutations.append(mutation)
    return mutations
