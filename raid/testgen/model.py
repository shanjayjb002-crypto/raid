"""Dataclasses describing generated test cases, and their table export.

Two kinds of case are produced, sharing a common head so they can be listed,
filtered and exported together:

* :class:`FunctionalTestCase` - one per execution path. Asserts that a given
  sequence of calls happens, under a given set of branch conditions.
* :class:`BoundaryTestCase` - one per (path, parameter, boundary) triple.
  Asserts behaviour at an edge-case value, along a specific path.

Both carry ``path_id``, which is what links a boundary case back to the
functional case it varies, and ultimately back to the diagram element that
produced them - the relationship the ``trace`` module will later walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from raid.dsl.model import Interaction, Parameter

__all__ = [
    "BoundaryTestCase",
    "FunctionalTestCase",
    "GeneratedTestCase",
    "to_table",
]


def _short_repr(value: object, limit: int = 80) -> str:
    """Render a value for a table cell, truncated so one case cannot swamp it.

    Boundary values are deliberately extreme - the ``very_long`` text case is
    10,000 characters - so the table export must bound them. The full value
    stays on the dataclass; only the exported cell is shortened.
    """
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass(frozen=True)
class GeneratedTestCase:
    """Fields common to every generated case."""

    id: str
    flow: str
    path_id: str
    path_index: int
    preconditions: tuple[str, ...]

    @property
    def precondition_text(self) -> str:
        """The preconditions as one readable clause, for tables and reports."""
        return " AND ".join(self.preconditions) if self.preconditions else "(none)"

    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class FunctionalTestCase(GeneratedTestCase):
    """Asserts that one execution path produces its expected call sequence."""

    interactions: tuple[Interaction, ...]
    expected_call_sequence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": "functional",
            "flow": self.flow,
            "path_id": self.path_id,
            "preconditions": self.precondition_text,
            "steps": len(self.interactions),
            "expected_call_sequence": " | ".join(self.expected_call_sequence),
        }


@dataclass(frozen=True)
class BoundaryTestCase(GeneratedTestCase):
    """Asserts behaviour when one parameter takes an edge-case value.

    ``step_index`` is the 1-based position of the owning interaction within the
    path, which is what keeps ids unique when a flow calls the same method more
    than once.
    """

    step_index: int
    interaction: Interaction
    parameter: Parameter
    inferred_class: str
    confidence: float
    boundary_case: str
    value: object

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": "boundary",
            "flow": self.flow,
            "path_id": self.path_id,
            "preconditions": self.precondition_text,
            "call": f"{self.interaction.source} -> {self.interaction.target} : {self.interaction.method}",
            "parameter": self.parameter.name,
            "declared_type": self.parameter.type,
            "inferred_class": self.inferred_class,
            "boundary_case": self.boundary_case,
            "value": _short_repr(self.value),
            "confidence": round(self.confidence, 3),
        }


def to_table(cases: Sequence[GeneratedTestCase]) -> list[dict[str, object]]:
    """Export generated cases as a list of dicts, one flat row each.

    Every cell is a scalar, so the result can be handed straight to a
    ``pandas.DataFrame`` or ``st.dataframe`` without further massaging. Mixing
    functional and boundary cases in one call is allowed; the rows simply carry
    different keys, distinguished by the ``kind`` column.
    """
    return [case.to_dict() for case in cases]
