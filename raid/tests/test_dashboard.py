"""Unit tests for the dashboard's analysis pipeline.

The Streamlit app in ``raid.dashboard.app`` is a thin rendering layer over
:func:`raid.dashboard.analyse`. All the logic lives in the pipeline so it can
be tested here without a browser, which is what PROJECT_RULES.md requires of anything
wired into the dashboard.

These tests deliberately exercise diagrams other than PlaceOrder: the demo must
survive arbitrary valid input, including degenerate cases a live audience might
type.
"""

import csv
import io
import textwrap

import pytest

from raid.dsl import DslSyntaxError
from raid.dashboard import Analysis, AnalysisOptions, analyse, rows_to_csv


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

SIMPLE = dsl(
    """
    service Client
    service ApiGateway

    flow Ping:
      Client -> ApiGateway : ping(requestId:string)
    """
)

FAST = AnalysisOptions(runs=10, mutations=6, seed=0)


# ==========================================================================
# The worked example
# ==========================================================================


def test_place_order_runs_the_whole_pipeline():
    result = analyse(PLACE_ORDER, FAST)

    assert isinstance(result, Analysis)
    assert result.flow_name == "PlaceOrder"
    assert len(result.functional) == 3
    assert result.boundary
    assert result.report is not None
    assert result.impact.graph.number_of_nodes() > 0


def test_sufficiency_is_computed_even_though_the_diagram_is_analysable():
    result = analyse(PLACE_ORDER, FAST)

    assert result.sufficiency.concerns
    assert result.sufficiency.overall in ("low", "medium", "high")


def test_focus_options_are_the_declared_services():
    result = analyse(PLACE_ORDER, FAST)

    assert set(result.focus_options()) == {
        "PaymentService",
        "OrderService",
        "InventoryService",
    }


# ==========================================================================
# Arbitrary valid input - the demo must not depend on one example
# ==========================================================================


def test_a_minimal_single_call_diagram_works():
    result = analyse(SIMPLE, FAST)

    assert result.flow_name == "Ping"
    assert len(result.functional) == 1
    assert result.functional[0].expected_call_sequence == (
        "Client -> ApiGateway : ping",
    )
    assert result.boundary
    assert result.report is not None


def test_a_diagram_with_services_but_no_flow_does_not_crash():
    result = analyse(dsl("service A\nservice B\n"), FAST)

    assert result.flow_name is None
    assert result.functional == ()
    assert result.boundary == ()
    assert result.report is None
    assert result.chaos == ()
    assert result.warnings


def test_an_empty_diagram_does_not_crash():
    result = analyse("", FAST)

    assert result.flow_name is None
    assert result.sufficiency.overall == "low"
    assert result.warnings


def test_an_untyped_sparse_diagram_still_produces_output():
    """Incomplete input must still give useful results, not an error."""
    result = analyse(
        dsl(
            """
            service A
            service B

            flow F:
              A -> B : doThing(value, other)
            """
        ),
        FAST,
    )

    assert result.functional
    assert result.boundary  # inferred from names alone
    assert result.sufficiency.suggestions_for("testgen")


def test_a_flow_with_only_a_self_call_works():
    result = analyse(
        dsl(
            """
            service Loner

            flow Solo:
              Loner -> Loner : tick(n:int)
            """
        ),
        FAST,
    )

    assert len(result.functional) == 1
    assert result.report is not None


def test_multiple_flows_analyse_the_first_by_default():
    source = dsl(
        """
        service A
        service B

        flow First:
          A -> B : one(x:int)

        flow Second:
          A -> B : two(y:int)
        """
    )

    result = analyse(source, FAST)

    assert result.flow_name == "First"
    assert result.flow_names == ("First", "Second")


def test_a_specific_flow_can_be_selected():
    source = dsl(
        """
        service A
        service B

        flow First:
          A -> B : one(x:int)

        flow Second:
          A -> B : two(y:int)
        """
    )

    result = analyse(source, AnalysisOptions(runs=10, mutations=4, flow_name="Second"))

    assert result.flow_name == "Second"
    assert [c.method for c in result.functional[0].interactions] == ["two"]


# ==========================================================================
# Failure injection
# ==========================================================================


def test_injecting_a_failure_surfaces_a_timeout():
    result = analyse(
        PLACE_ORDER, AnalysisOptions(runs=10, mutations=4, failure_service="PaymentService")
    )

    assert result.report.timed_out
    assert result.report.timeouts[0].method == "charge"


def test_an_unknown_failure_service_warns_instead_of_crashing():
    """A stale selection must degrade to a warning, never break the demo."""
    result = analyse(
        PLACE_ORDER, AnalysisOptions(runs=10, mutations=4, failure_service="Ghost")
    )

    assert result.report is not None
    assert any("Ghost" in w for w in result.warnings)


# ==========================================================================
# Parse errors
# ==========================================================================


def test_a_parse_error_is_raised_for_the_ui_to_catch():
    with pytest.raises(DslSyntaxError) as excinfo:
        analyse("flow Broken:\n  A B : c()\n", FAST)

    message = str(excinfo.value)
    assert message.startswith("RAID DSL syntax error")
    assert "Traceback" not in message


# ==========================================================================
# Filtering via the impact graph
# ==========================================================================


def test_unfiltered_slice_contains_everything():
    result = analyse(PLACE_ORDER, FAST)

    everything = result.slice_for(None)

    assert len(everything.functional_tests) == len(result.functional)
    assert len(everything.boundary_tests) == len(result.boundary)
    assert len(everything.chaos_specs) == len(result.chaos)


def test_filtering_by_a_service_narrows_every_table():
    result = analyse(
        PLACE_ORDER, AnalysisOptions(runs=10, mutations=4, failure_service="PaymentService")
    )

    payment = result.slice_for("PaymentService")

    assert [c.id for c in payment.functional_tests] == [
        "PlaceOrder.path1.functional",
        "PlaceOrder.path2.functional",
    ]
    assert {c.parameter.name for c in payment.boundary_tests} == {"amount", "cardId"}
    assert [v.method for v in payment.resilience_findings] == ["charge"]


def test_filtering_by_an_unrelated_service_excludes_other_findings():
    result = analyse(
        PLACE_ORDER, AnalysisOptions(runs=10, mutations=4, failure_service="PaymentService")
    )

    inventory = result.slice_for("InventoryService")

    assert {c.parameter.name for c in inventory.boundary_tests} == {"itemId", "qty"}
    assert inventory.resilience_findings == ()


def test_filtering_by_an_unknown_service_is_empty_not_an_error():
    result = analyse(PLACE_ORDER, FAST)

    assert result.slice_for("NoSuchService").is_empty()


# ==========================================================================
# Export
# ==========================================================================


def test_test_case_rows_combine_functional_and_boundary():
    result = analyse(PLACE_ORDER, FAST)

    rows = result.test_case_rows()

    kinds = {row["kind"] for row in rows}
    assert kinds == {"functional", "boundary"}
    assert len(rows) == len(result.functional) + len(result.boundary)


def test_rows_to_csv_produces_a_parsable_document():
    result = analyse(PLACE_ORDER, FAST)

    text = rows_to_csv(result.test_case_rows())
    parsed = list(csv.DictReader(io.StringIO(text)))

    assert len(parsed) == len(result.test_case_rows())
    assert "id" in parsed[0]


def test_rows_to_csv_handles_ragged_rows():
    """Functional and boundary rows carry different keys; both must export."""
    text = rows_to_csv([{"a": 1}, {"b": 2}])
    parsed = list(csv.DictReader(io.StringIO(text)))

    assert {"a", "b"} <= set(parsed[0])


def test_rows_to_csv_of_nothing_is_empty_not_a_crash():
    assert rows_to_csv([]) == ""
