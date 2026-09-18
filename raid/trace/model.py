"""Node identity and query results for the cross-concern impact graph.

Node ids are strings prefixed by kind (``service:PaymentService``,
``interaction:PlaceOrder:8``). A prefix is used rather than bare names because
one namespace has to hold diagram elements and generated artifacts at once, and
a service could otherwise collide with a flow of the same name.

Interactions are addressed by ``(flow, source line)``. The grammar puts exactly
one interaction on a line, which makes the line a stable identity that survives
the tree being rebuilt - the same property the fuzzer relies on to rewrite a
flow, and the reason phase 1 records a line on every construct.
"""

from __future__ import annotations

from dataclasses import dataclass

from raid.dsl.model import Interaction
from raid.fuzz import ChaosTestSpec
from raid.simulate import TimeoutViolation
from raid.testgen import BoundaryTestCase, FunctionalTestCase, GeneratedTestCase

__all__ = [
    "ARTIFACT_KINDS",
    "ELEMENT_KINDS",
    "ImpactSet",
    "SourceSet",
    "UnknownElementError",
    "artifact_id",
    "flow_id",
    "interaction_id",
    "service_id",
]


class UnknownElementError(KeyError):
    """Raised when a queried node is not in the impact graph.

    A ``KeyError`` subclass so it reads naturally as a lookup failure, but with
    a message naming what was asked for - a silent empty result would look
    exactly like "this element produced nothing", which is the opposite of the
    truth.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


#: Node kinds that come from the diagram.
ELEMENT_KINDS: tuple[str, ...] = ("service", "flow", "interaction")

#: Node kinds that are generated downstream.
ARTIFACT_KINDS: tuple[str, ...] = (
    "functional_test",
    "boundary_test",
    "resilience_finding",
    "chaos_spec",
)


def service_id(name: str) -> str:
    """Node id for a service."""
    return f"service:{name}"


def flow_id(name: str) -> str:
    """Node id for a flow."""
    return f"flow:{name}"


def interaction_id(flow: str, line: int) -> str:
    """Node id for one interaction, addressed by its flow and source line."""
    return f"interaction:{flow}:{line}"


def artifact_id(artifact: object) -> str:
    """Node id for any generated artifact.

    Dispatches on type so a caller can hand over whatever object it already
    holds - a test case, a timeout violation, a chaos spec - without needing to
    know the naming scheme.

    Raises:
        TypeError: If the object is not a recognised artifact.
    """
    if isinstance(artifact, ChaosTestSpec):
        return f"chaos:{artifact.id}"
    if isinstance(artifact, GeneratedTestCase):
        return f"test:{artifact.id}"
    if isinstance(artifact, TimeoutViolation):
        return f"finding:{artifact.signature}"
    raise TypeError(f"Not a traceable artifact: {type(artifact).__name__}")


@dataclass(frozen=True)
class ImpactSet:
    """Everything downstream of one diagram element, split by concern."""

    interactions: tuple[Interaction, ...] = ()
    functional_tests: tuple[FunctionalTestCase, ...] = ()
    boundary_tests: tuple[BoundaryTestCase, ...] = ()
    resilience_findings: tuple[TimeoutViolation, ...] = ()
    chaos_specs: tuple[ChaosTestSpec, ...] = ()
    artifact_ids: tuple[str, ...] = ()

    @property
    def total_artifacts(self) -> int:
        return len(self.artifact_ids)

    def is_empty(self) -> bool:
        """Whether this element produced nothing at all - a coverage gap."""
        return self.total_artifacts == 0


@dataclass(frozen=True)
class SourceSet:
    """The diagram elements that produced one artifact."""

    services: tuple[str, ...] = ()
    flows: tuple[str, ...] = ()
    interactions: tuple[Interaction, ...] = ()
