"""Mutation engine and chaos test spec writer.

Repeatedly mutates a flow - killing a service, removing a dependency,
injecting a cycle, or inflating a latency distribution - simulates each mutant,
and writes a :class:`ChaosTestSpec` whenever the mutant is measurably worse
than the unmutated baseline.

Typical use::

    from raid.dsl import parse
    from raid.graph import build_graph
    from raid.fuzz import fuzz_and_report

    graph = build_graph(parse(source_text))
    for spec in fuzz_and_report(graph, "PlaceOrder", num_mutations=20):
        print(spec.description)
        print("  expected:", spec.baseline_behaviour)
        print("  observed:", spec.observed_behaviour)
"""

from .engine import DEFAULT_LATENCY_REGRESSION_FACTOR, fuzz_and_report
from .model import REGRESSION_KINDS, ChaosTestSpec
from .mutations import (
    MUTATION_KINDS,
    Mutation,
    MutationKind,
    MutationTargetError,
    apply_mutation,
    generate_mutations,
)

__all__ = [
    "DEFAULT_LATENCY_REGRESSION_FACTOR",
    "MUTATION_KINDS",
    "REGRESSION_KINDS",
    "ChaosTestSpec",
    "Mutation",
    "MutationKind",
    "MutationTargetError",
    "apply_mutation",
    "fuzz_and_report",
    "generate_mutations",
]
