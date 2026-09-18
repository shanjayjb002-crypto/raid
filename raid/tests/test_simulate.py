"""Unit tests for the SimPy resilience simulation.

Simulated time is measured in milliseconds throughout (see
:mod:`raid.simulate.engine`). All runs are seeded, so every assertion here is
on a deterministic result rather than on a distribution that might occasionally
misbehave in CI.
"""

import textwrap

import pytest

from raid.dsl import parse
from raid.graph import UnknownFlowError, build_graph
from raid.simulate import (
    Failure,
    ResilienceReport,
    SimulationConfig,
    UnknownServiceError,
    parse_duration,
    run_simulation,
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


@pytest.fixture(scope="module")
def graph():
    return build_graph(parse(PLACE_ORDER))


FAST = SimulationConfig(runs=50, seed=7)


# ==========================================================================
# Duration parsing - the DSL stores durations verbatim, simulation reads them
# ==========================================================================


@pytest.mark.parametrize(
    "text, expected_ms",
    [
        ("250ms", 250.0),
        ("2s", 2000.0),
        ("1.5s", 1500.0),
        ("1m", 60_000.0),
        ("1h", 3_600_000.0),
        (500, 500.0),
        (2.5, 2.5),
    ],
)
def test_parse_duration_converts_to_milliseconds(text, expected_ms):
    assert parse_duration(text) == expected_ms


def test_parse_duration_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_duration("soon")


# ==========================================================================
# Baseline - no failures injected
# ==========================================================================


def test_baseline_run_completes_without_timeout(graph):
    """The required baseline: a healthy system finishes every run in budget."""
    report = run_simulation(graph, "PlaceOrder", config=FAST)

    assert isinstance(report, ResilienceReport)
    assert report.timed_out is False
    assert report.timeouts == ()
    assert report.timed_out_runs == 0
    assert report.completed_runs == FAST.runs


def test_baseline_simulates_the_happy_path(graph):
    """The default path is the one where every alt guard holds."""
    report = run_simulation(graph, "PlaceOrder", config=FAST)

    assert report.path_id == "PlaceOrder.path1"
    assert report.preconditions == ("stock_available", "payment_success")
    assert report.call_sequence == ("checkStock", "charge", "reserve")


def test_baseline_reports_plausible_end_to_end_latency(graph):
    report = run_simulation(graph, "PlaceOrder", config=FAST)

    # Three calls at the 50ms default mean, so roughly 150ms end to end.
    assert 100.0 < report.latency_p50 < 220.0
    assert report.latency_p95 >= report.latency_p50
    assert report.latency_mean > 0


def test_latencies_are_never_negative(graph):
    """A normal distribution can sample below zero; SimPy cannot wait a negative."""
    config = SimulationConfig(
        runs=200, seed=1, default_latency_mean=5.0, default_latency_stddev=50.0
    )

    report = run_simulation(graph, "PlaceOrder", config=config)

    assert report.latency_p50 >= 0.0
    assert all(c.mean_latency >= 0.0 for c in report.critical_path)


def test_critical_path_covers_the_services_the_happy_path_calls(graph):
    report = run_simulation(graph, "PlaceOrder", config=FAST)

    assert {c.service for c in report.critical_path} == {
        "InventoryService",
        "PaymentService",
    }


def test_critical_path_is_ranked_by_contribution(graph):
    report = run_simulation(graph, "PlaceOrder", config=FAST)

    shares = [c.share for c in report.critical_path]
    assert shares == sorted(shares, reverse=True)
    assert sum(shares) == pytest.approx(1.0)


def test_inventory_dominates_the_critical_path_because_it_is_called_twice(graph):
    """checkStock and reserve both hit InventoryService, so it carries ~2/3."""
    report = run_simulation(graph, "PlaceOrder", config=FAST)

    top = report.critical_path[0]
    assert top.service == "InventoryService"
    assert top.calls == 2
    assert 0.55 < top.share < 0.78


def test_runs_are_deterministic_for_a_fixed_seed(graph):
    first = run_simulation(graph, "PlaceOrder", config=SimulationConfig(runs=30, seed=42))
    second = run_simulation(graph, "PlaceOrder", config=SimulationConfig(runs=30, seed=42))

    assert first.latency_p50 == second.latency_p50
    assert first.latency_p95 == second.latency_p95


def test_different_seeds_give_different_samples(graph):
    first = run_simulation(graph, "PlaceOrder", config=SimulationConfig(runs=30, seed=1))
    second = run_simulation(graph, "PlaceOrder", config=SimulationConfig(runs=30, seed=2))

    assert first.latency_p50 != second.latency_p50


# ==========================================================================
# Failure injection
# ==========================================================================


def test_payment_failure_makes_the_dependent_call_exceed_its_timeout(graph):
    """The required failure case.

    PaymentService stops responding, so OrderService's `charge` call - which
    declares `[timeout=2s]` - must be reported as exceeding that timeout.
    """
    report = run_simulation(
        graph, "PlaceOrder", failures=[Failure("PaymentService")], config=FAST
    )

    assert report.timed_out is True
    assert len(report.timeouts) == 1

    violation = report.timeouts[0]
    assert violation.caller == "OrderService"
    assert violation.service == "PaymentService"
    assert violation.method == "charge"
    assert violation.timeout_ms == 2000.0
    assert violation.occurrences == FAST.runs
    assert violation.attributed_to == "PaymentService"


def test_violations_and_call_stats_carry_the_source_line(graph):
    """Findings must anchor back to the diagram element that produced them.

    The trace layer joins a resilience finding to its interaction by source
    line, so a violation that does not carry one cannot be attributed.
    """
    report = run_simulation(
        graph, "PlaceOrder", failures=[Failure("PaymentService")], config=FAST
    )

    assert report.timeouts[0].line == 8  # the `charge` interaction
    assert [call.line for call in report.calls] == [6, 8]


def test_every_run_fails_when_the_service_is_fully_out(graph):
    report = run_simulation(
        graph, "PlaceOrder", failures=[Failure("PaymentService")], config=FAST
    )

    assert report.timed_out_runs == FAST.runs
    assert report.completed_runs == 0


def test_a_timed_out_call_aborts_the_rest_of_the_path(graph):
    """If charge never returns, reserve is never reached."""
    report = run_simulation(
        graph, "PlaceOrder", failures=[Failure("PaymentService")], config=FAST
    )

    assert "reserve" not in {c.method for c in report.calls}


def test_failure_dominates_the_critical_path(graph):
    """The waiting caller burns the full timeout, so the failure shows up."""
    report = run_simulation(
        graph, "PlaceOrder", failures=[Failure("PaymentService")], config=FAST
    )

    assert report.critical_path[0].service == "PaymentService"
    assert report.latency_p50 > 2000.0


def test_failing_an_uninvolved_service_changes_nothing(graph):
    baseline = run_simulation(graph, "PlaceOrder", config=FAST)
    report = run_simulation(
        graph, "PlaceOrder", failures=[Failure("OrderService")], config=FAST
    )

    # OrderService is only ever a caller on the happy path, never a callee.
    assert report.timed_out is False
    assert report.latency_p50 == baseline.latency_p50


def test_slow_mode_degrades_latency_without_necessarily_timing_out(graph):
    """A brownout is not an outage: responses still arrive, just late."""
    baseline = run_simulation(graph, "PlaceOrder", config=FAST)
    report = run_simulation(
        graph,
        "PlaceOrder",
        failures=[Failure("PaymentService", mode="slow", latency_multiplier=5.0)],
        config=FAST,
    )

    assert report.timed_out is False
    assert report.completed_runs == FAST.runs
    assert report.latency_p50 > baseline.latency_p50


def test_slow_mode_can_be_severe_enough_to_breach_the_timeout(graph):
    report = run_simulation(
        graph,
        "PlaceOrder",
        failures=[Failure("PaymentService", mode="slow", latency_multiplier=100.0)],
        config=FAST,
    )

    assert report.timed_out is True
    assert report.timeouts[0].method == "charge"


def test_a_failure_that_has_already_ended_does_not_bite(graph):
    """The charge call starts around 50ms, after this failure window closes."""
    report = run_simulation(
        graph,
        "PlaceOrder",
        failures=[Failure("PaymentService", start_at=0.0, duration=10.0)],
        config=FAST,
    )

    assert report.timed_out is False


def test_a_failure_that_starts_later_still_bites(graph):
    """checkStock takes ~50ms, so a failure opening at 10ms catches charge."""
    report = run_simulation(
        graph,
        "PlaceOrder",
        failures=[Failure("PaymentService", start_at=10.0, duration=10_000.0)],
        config=FAST,
    )

    assert report.timed_out is True


def test_report_records_the_injected_failures(graph):
    failures = [Failure("PaymentService")]

    report = run_simulation(graph, "PlaceOrder", failures=failures, config=FAST)

    assert report.failures == tuple(failures)


def test_failure_on_an_undeclared_service_is_rejected(graph):
    """A typo in a failure name must not silently simulate a healthy system."""
    with pytest.raises(UnknownServiceError) as excinfo:
        run_simulation(graph, "PlaceOrder", failures=[Failure("PaymentSrvice")])

    message = str(excinfo.value)
    assert "PaymentSrvice" in message
    assert "PaymentService" in message


def test_unknown_flow_is_rejected(graph):
    with pytest.raises(UnknownFlowError):
        run_simulation(graph, "NoSuchFlow", config=FAST)


# ==========================================================================
# Annotation overrides
# ==========================================================================


def test_latency_annotations_override_the_defaults():
    graph = build_graph(
        parse(
            dsl(
                """
                service A
                service B

                flow F:
                  A -> B : slowCall() [latency_mean=500ms, latency_stddev=1ms]
                """
            )
        )
    )

    report = run_simulation(graph, "F", config=FAST)

    assert 480.0 < report.latency_p50 < 520.0


def test_timeout_annotation_is_respected():
    graph = build_graph(
        parse(
            dsl(
                """
                service A
                service B

                flow F:
                  A -> B : tightCall() [latency_mean=100ms, timeout=10ms]
                """
            )
        )
    )

    report = run_simulation(graph, "F", config=FAST)

    assert report.timed_out is True
    assert report.timeouts[0].timeout_ms == 10.0


def test_default_timeout_applies_when_unannotated(graph):
    config = SimulationConfig(runs=10, seed=3, default_timeout=25.0)

    report = run_simulation(graph, "PlaceOrder", config=config)

    # checkStock has no timeout annotation, so the config default applies and
    # the 50ms default latency breaches it.
    assert report.timed_out is True
    assert report.timeouts[0].method == "checkStock"
    assert report.timeouts[0].timeout_ms == 25.0


# ==========================================================================
# Report export
# ==========================================================================


def test_report_exports_to_a_flat_dict(graph):
    report = run_simulation(
        graph, "PlaceOrder", failures=[Failure("PaymentService")], config=FAST
    )

    row = report.to_dict()

    for key in (
        "flow",
        "path_id",
        "runs",
        "latency_p50",
        "latency_p95",
        "timed_out",
        "critical_path",
        "failures",
    ):
        assert key in row
    for value in row.values():
        assert isinstance(value, (str, int, float, bool)) or value is None
