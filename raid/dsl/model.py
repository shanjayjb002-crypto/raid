"""Data model for the RAID architecture DSL.

Design notes (for the viva)
---------------------------
The parse tree produced by lark is deliberately *not* the representation the
rest of RAID works with. Lark trees are generic (``Tree(data, children)``) and
force every downstream module to know grammar rule names. Instead the parser
lowers the tree into the small set of dataclasses below, so that later phases
(graph building, path enumeration, simulation) depend on a stable, typed
vocabulary rather than on grammar internals. Changing the concrete syntax then
only breaks the grammar and the transformer, never the analysis code.

The shape of these classes mirrors the DSL's concrete syntax one-for-one
(``service`` -> :class:`Service`, ``flow`` -> :class:`Flow`, ``alt``/``else``
-> :class:`Branch`). That correspondence is intentional: it makes the mapping
from source text to model trivially explainable, and keeps the transformer free
of any interpretation or normalisation logic.

Every construct carries the source ``line`` it came from. This is what the
``trace`` module later uses to link a generated test, a resilience finding or a
threat flag back to the exact diagram element that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

#: The value of an interaction annotation such as ``retry=3`` or ``timeout=2s``.
#:
#: Numbers are coerced to ``int``/``float``; quoted strings are unquoted;
#: durations (``2s``, ``250ms``) and bare words are kept as strings. Durations
#: are deliberately *not* converted to seconds here - the DSL layer's job is to
#: record what was written, and the simulation layer decides what a duration
#: means in its own time base.
AnnotationValue = Union[int, float, str]


@dataclass(frozen=True)
class Parameter:
    """A single typed parameter of an interaction, e.g. ``itemId:int``.

    ``type`` is kept as the raw type name written in the diagram. The DSL does
    not have a fixed type system: the boundary-value test generator in
    ``testgen`` is what attaches meaning (ranges, edge cases) to names like
    ``int`` or ``float``, so the parser stays agnostic.
    """

    name: str
    type: str


@dataclass
class Interaction:
    """One directed call between two services within a flow.

    Corresponds to ``Source -> Target : method(param:type, ...) [key=value]``.

    Services are referred to by name rather than by object reference. Resolving
    those names against declared services is the ``graph`` module's job, which
    keeps parsing a pure syntactic step and lets the ``sufficiency`` scorer
    report undeclared services as a diagram completeness problem rather than a
    parse failure.
    """

    source: str
    target: str
    method: str
    parameters: list[Parameter] = field(default_factory=list)
    annotations: dict[str, AnnotationValue] = field(default_factory=dict)
    line: int = 0


@dataclass
class Branch:
    """A conditional fragment: ``alt <condition>:`` with an optional ``else``.

    Both arms hold their own ordered list of steps, and a step may itself be a
    :class:`Branch`, which is what gives the DSL arbitrary nesting. Path
    enumeration in ``testgen`` walks this structure and yields one path per
    combination of arms taken.

    ``else_condition`` is ``None`` when no ``else`` clause was written. The DSL
    requires a named condition on ``else`` (rather than a bare ``else:``) so
    that every generated test path has a human-readable label explaining why it
    was taken - which is precisely what makes the generated test suite
    reviewable.
    """

    condition: str
    steps: list[Step] = field(default_factory=list)
    else_condition: str | None = None
    else_steps: list[Step] = field(default_factory=list)
    line: int = 0


#: Anything that can appear in the body of a flow or a branch arm.
Step = Union[Interaction, Branch]


@dataclass
class Service:
    """A declared participant in the architecture."""

    name: str
    line: int = 0


@dataclass
class Flow:
    """A named, ordered scenario through the architecture."""

    name: str
    steps: list[Step] = field(default_factory=list)
    line: int = 0


@dataclass
class Diagram:
    """The root of a parsed DSL document: everything one source file declares."""

    services: list[Service] = field(default_factory=list)
    flows: list[Flow] = field(default_factory=list)
