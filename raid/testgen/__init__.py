"""Test case generation from a RAID graph model.

Two generators sit on top of :meth:`RaidGraph.enumerate_paths`:

* :func:`generate_functional_cases` - one case per execution path, asserting
  the expected call sequence under that path's branch conditions.
* :func:`generate_boundary_cases` - edge-case values for every typed
  parameter, inferred by a local scikit-learn classifier and attached to the
  execution path the call sits on.

Both return dataclasses; :func:`to_table` flattens either into a list of dicts
for display or export.

Typical use::

    from raid.dsl import parse
    from raid.graph import build_graph
    from raid.testgen import generate_functional_cases, to_table

    graph = build_graph(parse(source_text))
    table = to_table(generate_functional_cases(graph, "PlaceOrder"))
"""

from .boundary import (
    BOUNDARY_CASES,
    TRAINING_EXAMPLES,
    BoundaryCandidate,
    BoundaryProposal,
    ParameterClassifier,
    get_classifier,
)
from .generator import generate_boundary_cases, generate_functional_cases
from .model import BoundaryTestCase, FunctionalTestCase, GeneratedTestCase, to_table

__all__ = [
    "BOUNDARY_CASES",
    "TRAINING_EXAMPLES",
    "BoundaryCandidate",
    "BoundaryProposal",
    "BoundaryTestCase",
    "FunctionalTestCase",
    "GeneratedTestCase",
    "ParameterClassifier",
    "generate_boundary_cases",
    "generate_functional_cases",
    "get_classifier",
    "to_table",
]
