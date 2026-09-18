"""RAID's custom architecture DSL: grammar, parser and data model.

This package is the only entry point for diagram input in RAID. Every later
phase consumes a :class:`~raid.dsl.model.Diagram` produced here, which is what
keeps the rest of the system independent of the concrete syntax.

Typical use::

    from raid.dsl import parse, DslSyntaxError

    try:
        diagram = parse(source_text)
    except DslSyntaxError as exc:
        print(exc)
"""

from .errors import DslError, DslSyntaxError
from .model import (
    AnnotationValue,
    Branch,
    Diagram,
    Flow,
    Interaction,
    Parameter,
    Service,
    Step,
)
from .parser import parse

__all__ = [
    "AnnotationValue",
    "Branch",
    "Diagram",
    "DslError",
    "DslSyntaxError",
    "Flow",
    "Interaction",
    "Parameter",
    "Service",
    "Step",
    "parse",
]
