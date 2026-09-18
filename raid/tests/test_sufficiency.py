"""Unit tests for the diagram sufficiency scorer.

Two fixtures anchor the tests: a deliberately sparse diagram that should score
badly on every concern, and a well-specified one that should score well. The
sparse case checks that suggestions are not merely present but *specific* -
naming the parameter, the call and the annotation to add - since a scorer that
says "needs more detail" is no use to anyone.
"""

import textwrap

import pytest

from raid.dsl import parse
from raid.graph import build_graph
from raid.sufficiency import (
    CONCERNS,
    SCORES,
    SufficiencyReport,
    Suggestion,
    score_sufficiency,
)


def dsl(source: str) -> str:
    return textwrap.dedent(source).lstrip("\n")


SPARSE = dsl(
    """
    service OrderService
    service PaymentService
    service LonelyService

    flow PlaceOrder:
      OrderService -> PaymentService : charge(amount, cardId)
      OrderService -> PaymentService : refund(orderId:string)
    """
)

WELL_SPECIFIED = dsl(
    """
    service OrderService
    service PaymentService

    flow PlaceOrder:
      OrderService -> PaymentService : charge(amount:float) [timeout=2s, retry=3]
      PaymentService -> OrderService : confirm(orderId:string) [timeout=1s, retry=2]
    """
)


def build(source: str):
    return build_graph(parse(source))


@pytest.fixture(scope="module")
def sparse():
    return score_sufficiency(build(SPARSE))


@pytest.fixture(scope="module")
def rich():
    return score_sufficiency(build(WELL_SPECIFIED))


# ==========================================================================
# Shape of the report
# ==========================================================================


def test_report_covers_every_concern(sparse):
    assert isinstance(sparse, SufficiencyReport)
    assert {c.concern for c in sparse.concerns} == set(CONCERNS)


def test_scores_use_the_declared_vocabulary(sparse, rich):
    for report in (sparse, rich):
        for concern in report.concerns:
            assert concern.score in SCORES


def test_every_suggestion_is_attributed_and_actionable(sparse):
    assert sparse.suggestions
    for suggestion in sparse.suggestions:
        assert isinstance(suggestion, Suggestion)
        assert suggestion.concern in CONCERNS
        assert suggestion.severity in SCORES
        assert suggestion.message
        assert suggestion.element


# ==========================================================================
# The sparse diagram - suggestions must be non-empty and specific
# ==========================================================================


def test_sparse_diagram_produces_suggestions(sparse):
    assert len(sparse.suggestions) > 0


def test_sparse_diagram_scores_badly_on_every_concern(sparse):
    assert sparse.score_for("testgen").score == "low"
    assert sparse.score_for("resilience").score == "low"
    assert sparse.overall == "low"


def test_untyped_parameters_are_flagged_by_name_and_call(sparse):
    """The suggestion must name the parameter and the method, not just complain."""
    messages = [s.message for s in sparse.suggestions_for("testgen")]

    assert any("'amount'" in m and "charge" in m for m in messages)
    assert any("'cardId'" in m and "charge" in m for m in messages)


def test_untyped_parameter_suggestion_says_what_it_unlocks(sparse):
    message = next(
        s.message for s in sparse.suggestions_for("testgen") if "'amount'" in s.message
    )

    assert message.startswith("add a type to")
    assert "boundary test" in message


def test_typed_parameters_are_not_flagged(sparse):
    """`orderId:string` is fully specified, so nothing should be said about it."""
    messages = [s.message for s in sparse.suggestions_for("testgen")]

    assert not any("orderId" in m for m in messages)


def test_missing_timeout_is_flagged_per_interaction(sparse):
    messages = [s.message for s in sparse.suggestions_for("resilience")]

    assert any("[timeout=" in m and "charge" in m for m in messages)
    assert any("[timeout=" in m and "refund" in m for m in messages)


def test_missing_timeout_suggestion_says_what_it_unlocks(sparse):
    message = next(
        s.message for s in sparse.suggestions_for("resilience") if "[timeout=" in s.message
    )

    assert "resilience simulation" in message


def test_missing_retry_is_flagged_more_gently_than_missing_timeout(sparse):
    """A missing timeout blocks simulation; a missing retry only limits it."""
    timeout = next(s for s in sparse.suggestions_for("resilience") if "[timeout=" in s.message)
    retry = next(s for s in sparse.suggestions_for("resilience") if "[retry=" in s.message)

    assert timeout.severity == "high"
    assert retry.severity == "low"


def test_isolated_service_is_flagged_for_fuzzing(sparse):
    messages = [s.message for s in sparse.suggestions_for("fuzzing")]

    assert any("LonelyService" in m for m in messages)
    assert any("nothing to mutate" in m for m in messages)


def test_a_service_that_is_never_called_is_flagged(sparse):
    """OrderService only ever calls out, so killing it would change nothing."""
    messages = [s.message for s in sparse.suggestions_for("fuzzing")]

    assert any("OrderService" in m for m in messages)


def test_a_called_service_is_not_flagged_for_fuzzing(sparse):
    messages = [s.message for s in sparse.suggestions_for("fuzzing")]

    assert not any("PaymentService" in m for m in messages)


def test_suggestions_link_back_to_a_trace_node_id(sparse):
    """Every suggestion addresses an element the impact graph also knows."""
    for suggestion in sparse.suggestions:
        assert suggestion.element_id.startswith(("service:", "interaction:", "flow:"))


def test_suggestions_are_ranked_most_severe_first(sparse):
    order = {"high": 0, "medium": 1, "low": 2}
    ranks = [order[s.severity] for s in sparse.suggestions]

    assert ranks == sorted(ranks)


# ==========================================================================
# The well-specified diagram - nothing to say
# ==========================================================================


def test_well_specified_diagram_scores_high_everywhere(rich):
    assert rich.score_for("testgen").score == "high"
    assert rich.score_for("resilience").score == "high"
    assert rich.score_for("fuzzing").score == "high"
    assert rich.overall == "high"


def test_well_specified_diagram_has_no_suggestions(rich):
    assert rich.suggestions == ()


def test_partial_annotation_scores_between_the_two():
    """Timeouts but no retries is half-ready, and must not score as either extreme."""
    report = score_sufficiency(
        build(
            dsl(
                """
                service A
                service B

                flow F:
                  A -> B : one(x:int) [timeout=1s]
                  B -> A : two(y:int) [timeout=1s]
                """
            )
        )
    )

    assert report.score_for("resilience").score == "medium"
    assert all("[retry=" in s.message for s in report.suggestions_for("resilience"))


# ==========================================================================
# Completeness problems deferred here by earlier phases
# ==========================================================================


def test_undeclared_service_is_reported_here():
    """Phase 2 deliberately left this to the sufficiency scorer."""
    report = score_sufficiency(
        build(
            dsl(
                """
                service OrderService

                flow F:
                  OrderService -> GhostService : ping(id:int) [timeout=1s, retry=1]
                """
            )
        )
    )

    messages = [s.message for s in report.suggestions_for("fuzzing")]

    assert any("GhostService" in m and "never declared" in m for m in messages)


def test_empty_diagram_is_not_analysable():
    report = score_sufficiency(build(""))

    assert report.overall == "low"
    assert all(c.score == "low" for c in report.concerns)


# ==========================================================================
# Export
# ==========================================================================


def test_report_exports_concern_scores_as_a_table(sparse):
    rows = sparse.to_table()

    assert len(rows) == len(CONCERNS)
    for row in rows:
        for key in ("concern", "score", "ready", "total", "suggestions"):
            assert key in row
        for value in row.values():
            assert isinstance(value, (str, int, float, bool)) or value is None


def test_suggestions_export_as_a_table(sparse):
    rows = sparse.suggestions_to_table()

    assert len(rows) == len(sparse.suggestions)
    for row in rows:
        for key in ("concern", "severity", "element", "message"):
            assert key in row
        for value in row.values():
            assert isinstance(value, (str, int, float, bool)) or value is None
