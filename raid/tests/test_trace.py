"""Unit tests for the cross-concern traceability layer.

The contract under test is bidirectional: a diagram element must reach every
artifact it produced across all three concerns, and every artifact must name
exactly the diagram elements it came from. The round-trip test at the end is
the real proof - it asserts the two directions agree for every artifact in the
graph rather than for a hand-picked few.
"""

import textwrap

import pytest

from raid.dsl import parse
from raid.fuzz import fuzz_and_report
from raid.graph import build_graph
from raid.simulate import Failure, SimulationConfig, run_simulation
from raid.testgen import generate_boundary_cases, generate_functional_cases
from raid.trace import (
    ImpactGraph,
    ImpactSet,
    SourceSet,
    UnknownElementError,
    artifact_id,
    build_impact_graph,
    flow_id,
    interaction_id,
    service_id,
)


def dsl(source: str) -> str:
    return textwrap.dedent(source).lstrip("\n")


PLACE_ORDER = dsl(
    """
    service PaymentService
    service OrderService
    service InventoryService

    flow PlaceOrder:
      OrderService -> InventoryService : checkStock(itemId:int, qty:int)
      alt stock_available:
        OrderService -> PaymentService : charge(amount:float, cardId:string) [timeout=2s]
        alt payment_success:
          OrderService -> InventoryService : reserve(itemId:int, qty:int)
      else out_of_stock:
        OrderService -> OrderService : rejectOrder()
    """
)

# Source lines of the interactions, used to address them directly.
LINE_CHECKSTOCK = 6
LINE_CHARGE = 8
LINE_RESERVE = 10
LINE_REJECT = 12

FAST = SimulationConfig(runs=20, seed=7)


@pytest.fixture(scope="module")
def graph():
    return build_graph(parse(PLACE_ORDER))


@pytest.fixture(scope="module")
def artifacts(graph):
    """Everything the earlier phases produce for this diagram."""
    functional = generate_functional_cases(graph, "PlaceOrder")
    boundary = generate_boundary_cases(graph, "PlaceOrder")
    report = run_simulation(
        graph, "PlaceOrder", failures=[Failure("PaymentService")], config=FAST
    )
    specs = fuzz_and_report(
        graph,
        "PlaceOrder",
        num_mutations=12,
        seed=0,
        targets=["PaymentService"],
        config=FAST,
    )
    return functional, boundary, report, specs


@pytest.fixture(scope="module")
def impact(graph, artifacts):
    functional, boundary, report, specs = artifacts
    return build_impact_graph(graph, [*functional, *boundary], report, specs)


# ==========================================================================
# Construction
# ==========================================================================


def test_build_returns_an_impact_graph(impact):
    assert isinstance(impact, ImpactGraph)
    assert impact.graph.number_of_nodes() > 0


def test_every_diagram_element_is_indexed(impact):
    for service in ("PaymentService", "OrderService", "InventoryService"):
        assert service_id(service) in impact.graph
    assert flow_id("PlaceOrder") in impact.graph
    for line in (LINE_CHECKSTOCK, LINE_CHARGE, LINE_RESERVE, LINE_REJECT):
        assert interaction_id("PlaceOrder", line) in impact.graph


def test_artifacts_are_indexed(impact, artifacts):
    functional, boundary, report, specs = artifacts

    for case in [*functional, *boundary]:
        assert artifact_id(case) in impact.graph
    for violation in report.timeouts:
        assert artifact_id(violation) in impact.graph
    for spec in specs:
        assert artifact_id(spec) in impact.graph


def test_building_with_no_simulation_or_chaos_results_is_allowed(graph, artifacts):
    """The dashboard may have generated tests before anything has been run."""
    functional, _, _, _ = artifacts

    impact = build_impact_graph(graph, functional)

    result = impact.downstream(service_id("PaymentService"))
    assert result.functional_tests
    assert result.resilience_findings == ()
    assert result.chaos_specs == ()


# ==========================================================================
# Forward lookup: diagram element -> everything downstream
# ==========================================================================


def test_querying_payment_service_returns_all_three_concerns(impact):
    """The headline requirement, across tests, resilience and chaos."""
    result = impact.downstream(service_id("PaymentService"))

    assert isinstance(result, ImpactSet)
    assert result.functional_tests
    assert result.boundary_tests
    assert result.resilience_findings
    assert result.chaos_specs


def test_payment_service_reaches_only_the_paths_that_charge(impact):
    result = impact.downstream(service_id("PaymentService"))

    # path3 rejects the order and never touches PaymentService.
    assert [c.id for c in result.functional_tests] == [
        "PlaceOrder.path1.functional",
        "PlaceOrder.path2.functional",
    ]


def test_payment_service_reaches_only_its_own_parameters(impact):
    result = impact.downstream(service_id("PaymentService"))

    assert {c.parameter.name for c in result.boundary_tests} == {"amount", "cardId"}
    assert all(c.interaction.target == "PaymentService" for c in result.boundary_tests)


def test_payment_service_reaches_the_charge_timeout(impact):
    result = impact.downstream(service_id("PaymentService"))

    assert [v.method for v in result.resilience_findings] == ["charge"]
    assert result.resilience_findings[0].service == "PaymentService"


def test_payment_service_reaches_its_chaos_specs(impact):
    result = impact.downstream(service_id("PaymentService"))

    assert result.chaos_specs
    assert all(s.mutation.service == "PaymentService" for s in result.chaos_specs)


def test_inventory_service_does_not_inherit_payment_findings(impact):
    """Impact must not leak sideways between unrelated services."""
    result = impact.downstream(service_id("InventoryService"))

    assert {c.parameter.name for c in result.boundary_tests} == {"itemId", "qty"}
    assert result.resilience_findings == ()
    assert result.chaos_specs == ()


def test_querying_an_interaction_returns_what_that_call_produced(impact):
    result = impact.downstream(interaction_id("PlaceOrder", LINE_CHARGE))

    assert {c.method for c in result.interactions} == set()
    assert [c.id for c in result.functional_tests] == [
        "PlaceOrder.path1.functional",
        "PlaceOrder.path2.functional",
    ]
    assert {c.parameter.name for c in result.boundary_tests} == {"amount", "cardId"}
    assert [v.method for v in result.resilience_findings] == ["charge"]


def test_querying_the_flow_returns_everything(impact, artifacts):
    functional, boundary, report, specs = artifacts

    result = impact.downstream(flow_id("PlaceOrder"))

    assert len(result.functional_tests) == len(functional)
    assert len(result.boundary_tests) == len(boundary)
    assert len(result.resilience_findings) == len(report.timeouts)


def test_calling_service_also_reaches_the_timeout_it_suffered(impact):
    """OrderService is the caller whose charge call timed out."""
    result = impact.downstream(service_id("OrderService"))

    assert [v.method for v in result.resilience_findings] == ["charge"]


def test_unknown_element_is_rejected(impact):
    with pytest.raises(UnknownElementError) as excinfo:
        impact.downstream(service_id("NoSuchService"))

    assert "NoSuchService" in str(excinfo.value)


# ==========================================================================
# Reverse lookup: artifact -> the diagram elements that produced it
# ==========================================================================


def test_querying_a_chaos_spec_returns_payment_service_as_its_source(impact, artifacts):
    """The headline reverse requirement."""
    _, _, _, specs = artifacts
    kill = next(s for s in specs if s.mutation.kind == "kill_service")

    sources = impact.sources(artifact_id(kill))

    assert isinstance(sources, SourceSet)
    assert sources.services == ("PaymentService",)


def test_querying_a_structural_chaos_spec_returns_both_ends_of_the_call(
    impact, artifacts
):
    _, _, _, specs = artifacts
    structural = [s for s in specs if s.mutation.kind in ("remove_edge", "inject_cycle")]

    if not structural:
        pytest.skip("this seed produced no structural chaos specs")

    sources = impact.sources(artifact_id(structural[0]))

    assert set(sources.services) == {"OrderService", "PaymentService"}
    assert [i.method for i in sources.interactions] == ["charge"]


def test_querying_a_functional_test_returns_the_services_it_exercises(
    impact, artifacts
):
    functional, _, _, _ = artifacts
    happy = next(c for c in functional if c.id == "PlaceOrder.path1.functional")

    sources = impact.sources(artifact_id(happy))

    assert set(sources.services) == {
        "OrderService",
        "InventoryService",
        "PaymentService",
    }
    assert sources.flows == ("PlaceOrder",)
    assert [i.method for i in sources.interactions] == [
        "checkStock",
        "charge",
        "reserve",
    ]


def test_the_rejected_path_test_never_names_payment_service(impact, artifacts):
    functional, _, _, _ = artifacts
    rejected = next(c for c in functional if c.id == "PlaceOrder.path3.functional")

    sources = impact.sources(artifact_id(rejected))

    assert "PaymentService" not in sources.services


def test_querying_a_boundary_test_returns_its_single_interaction(impact, artifacts):
    _, boundary, _, _ = artifacts
    amount_case = next(c for c in boundary if c.parameter.name == "amount")

    sources = impact.sources(artifact_id(amount_case))

    assert [i.method for i in sources.interactions] == ["charge"]
    assert set(sources.services) == {"OrderService", "PaymentService"}


def test_querying_a_resilience_finding_returns_caller_and_callee(impact, artifacts):
    _, _, report, _ = artifacts
    violation = report.timeouts[0]

    sources = impact.sources(artifact_id(violation))

    assert set(sources.services) == {"OrderService", "PaymentService"}
    assert [i.method for i in sources.interactions] == ["charge"]


def test_sources_of_an_unknown_artifact_is_rejected(impact):
    with pytest.raises(UnknownElementError):
        impact.sources("chaos:does-not-exist")


# ==========================================================================
# The two directions must agree
# ==========================================================================


def test_every_artifact_round_trips_through_its_sources(impact):
    """For every artifact, each of its sources must reach it going forward."""
    artifacts = [
        node
        for node, data in impact.graph.nodes(data=True)
        if data["kind"]
        in ("functional_test", "boundary_test", "resilience_finding", "chaos_spec")
    ]

    assert artifacts

    for artifact in artifacts:
        sources = impact.sources(artifact)
        element_ids = [
            *(service_id(s) for s in sources.services),
            *(flow_id(f) for f in sources.flows),
        ]
        assert element_ids, f"{artifact} has no diagram source"

        for element in element_ids:
            reachable = impact.downstream(element).artifact_ids
            assert artifact in reachable, f"{element} does not reach {artifact}"


def test_summary_exports_a_row_per_diagram_element(impact):
    rows = impact.to_table()

    assert all(isinstance(row, dict) for row in rows)
    by_element = {row["element"]: row for row in rows}

    payment = by_element["PaymentService"]
    assert payment["kind"] == "service"
    assert payment["functional_tests"] == 2
    assert payment["resilience_findings"] == 1
    assert payment["chaos_specs"] > 0

    for value in payment.values():
        assert isinstance(value, (str, int, float, bool)) or value is None
