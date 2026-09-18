"""Graph model built from a parsed RAID diagram.

Turns the AST produced by :mod:`raid.dsl` into a ``networkx`` service graph and
provides enumeration of every distinct execution path through a flow, which is
the input to test generation, resilience simulation and impact tracing.

Typical use::

    from raid.dsl import parse
    from raid.graph import build_graph

    graph = build_graph(parse(source_text))
    for path in graph.enumerate_paths("PlaceOrder"):
        print(path.conditions, [i.method for i in path])
"""

from .builder import build_graph
from .model import ExecutionPath, GraphError, RaidGraph, UnknownFlowError

__all__ = [
    "ExecutionPath",
    "GraphError",
    "RaidGraph",
    "UnknownFlowError",
    "build_graph",
]
