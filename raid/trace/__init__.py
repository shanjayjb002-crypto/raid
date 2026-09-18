"""Cross-concern traceability between diagram elements and generated artifacts.

Builds one index linking every service, flow and interaction to the functional
tests, boundary tests, resilience findings and chaos specs derived from it, and
answers it in both directions.

Typical use::

    from raid.trace import build_impact_graph, service_id, artifact_id

    impact = build_impact_graph(graph, test_cases, report, chaos_specs)

    # Forward: what did this service produce?
    produced = impact.downstream(service_id("PaymentService"))
    print(produced.functional_tests, produced.resilience_findings)

    # Reverse: where did this chaos spec come from?
    print(impact.sources(artifact_id(spec)).services)
"""

from .impact import ImpactGraph, build_impact_graph
from .model import (
    ARTIFACT_KINDS,
    ELEMENT_KINDS,
    ImpactSet,
    SourceSet,
    UnknownElementError,
    artifact_id,
    flow_id,
    interaction_id,
    service_id,
)

__all__ = [
    "ARTIFACT_KINDS",
    "ELEMENT_KINDS",
    "ImpactGraph",
    "ImpactSet",
    "SourceSet",
    "UnknownElementError",
    "artifact_id",
    "build_impact_graph",
    "flow_id",
    "interaction_id",
    "service_id",
]
