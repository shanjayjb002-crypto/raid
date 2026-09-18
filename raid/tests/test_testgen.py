"""Unit tests for functional and boundary-value test generation.

The PlaceOrder expectations here are worked out by hand from the diagram (see
``test_graph.py`` for the derivation of its three execution paths) and asserted
element by element rather than by count.
"""

import textwrap

import pytest

from raid.dsl import Parameter, parse
from raid.graph import build_graph
from raid.testgen import (
    BoundaryTestCase,
    FunctionalTestCase,
    ParameterClassifier,
    generate_boundary_cases,
    generate_functional_cases,
    get_classifier,
    to_table,
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
        OrderService -> PaymentService : charge(amount:float, cardId:string)
        alt payment_success:
          OrderService -> InventoryService : reserve(itemId:int, qty:int)
      else out_of_stock:
        OrderService -> OrderService : rejectOrder()
    """
)


@pytest.fixture(scope="module")
def graph():
    return build_graph(parse(PLACE_ORDER))


# ==========================================================================
# Part A - functional test cases
# ==========================================================================


def test_every_branch_produces_its_own_functional_test_case(graph):
    """Three execution paths through PlaceOrder means three functional cases."""
    cases = generate_functional_cases(graph, "PlaceOrder")

    assert len(cases) == 3
    assert all(isinstance(c, FunctionalTestCase) for c in cases)
    assert [c.id for c in cases] == [
        "PlaceOrder.path1.functional",
        "PlaceOrder.path2.functional",
        "PlaceOrder.path3.functional",
    ]


def test_functional_case_ids_are_distinct(graph):
    cases = generate_functional_cases(graph, "PlaceOrder")

    assert len({c.id for c in cases}) == len(cases)


def test_each_functional_case_records_the_branch_that_led_to_it(graph):
    """Preconditions are exactly the alt/else guards selecting that path."""
    cases = generate_functional_cases(graph, "PlaceOrder")

    assert [list(c.preconditions) for c in cases] == [
        ["stock_available", "payment_success"],
        ["stock_available", "not payment_success"],
        ["out_of_stock"],
    ]


def test_no_two_functional_cases_share_preconditions(graph):
    """Distinct branches must not collapse into duplicate test cases."""
    cases = generate_functional_cases(graph, "PlaceOrder")

    assert len({c.preconditions for c in cases}) == len(cases)


def test_functional_case_carries_the_ordered_interactions(graph):
    cases = generate_functional_cases(graph, "PlaceOrder")

    assert [[i.method for i in c.interactions] for c in cases] == [
        ["checkStock", "charge", "reserve"],
        ["checkStock", "charge"],
        ["checkStock", "rejectOrder"],
    ]


def test_expected_call_sequence_is_the_assertion_for_the_happy_path(graph):
    cases = generate_functional_cases(graph, "PlaceOrder")

    assert cases[0].expected_call_sequence == (
        "OrderService -> InventoryService : checkStock",
        "OrderService -> PaymentService : charge",
        "OrderService -> InventoryService : reserve",
    )


def test_expected_call_sequence_for_the_rejected_path(graph):
    cases = generate_functional_cases(graph, "PlaceOrder")

    assert cases[2].expected_call_sequence == (
        "OrderService -> InventoryService : checkStock",
        "OrderService -> OrderService : rejectOrder",
    )


def test_functional_cases_link_to_their_execution_path(graph):
    cases = generate_functional_cases(graph, "PlaceOrder")

    assert [c.path_id for c in cases] == [
        "PlaceOrder.path1",
        "PlaceOrder.path2",
        "PlaceOrder.path3",
    ]


def test_generating_without_a_flow_name_covers_every_flow():
    graph = build_graph(
        parse(
            dsl(
                """
                flow First:
                  A -> B : one()

                flow Second:
                  alt ok:
                    A -> B : two()
                  else bad:
                    A -> B : three()
                """
            )
        )
    )

    cases = generate_functional_cases(graph)

    assert [c.id for c in cases] == [
        "First.path1.functional",
        "Second.path1.functional",
        "Second.path2.functional",
    ]


def test_functional_cases_export_to_a_table_of_dicts(graph):
    cases = generate_functional_cases(graph, "PlaceOrder")

    table = to_table(cases)

    assert isinstance(table, list)
    assert all(isinstance(row, dict) for row in table)
    assert len(table) == 3
    assert table[0]["id"] == "PlaceOrder.path1.functional"
    assert table[0]["flow"] == "PlaceOrder"
    assert table[0]["preconditions"] == "stock_available AND payment_success"
    assert "checkStock" in table[0]["expected_call_sequence"]


def test_table_rows_are_flat_scalars(graph):
    """A table row must be renderable directly by Streamlit/pandas."""
    table = to_table(generate_functional_cases(graph, "PlaceOrder"))

    for row in table:
        for value in row.values():
            assert isinstance(value, (str, int, float, bool)) or value is None


# ==========================================================================
# Part B - boundary-value inference
# ==========================================================================


def test_classifier_infers_numeric_for_a_float_amount():
    classifier = get_classifier()

    inferred, confidence = classifier.classify(Parameter("amount", "float"))

    assert inferred == "numeric"
    assert 0.0 < confidence <= 1.0


def test_classifier_infers_identifier_for_a_string_id():
    classifier = get_classifier()

    inferred, _ = classifier.classify(Parameter("cardId", "string"))

    assert inferred == "identifier"


def test_classifier_infers_email_for_an_email_parameter():
    classifier = get_classifier()

    inferred, _ = classifier.classify(Parameter("email", "string"))

    assert inferred == "email"


def test_classifier_infers_text_for_free_text_parameters():
    classifier = get_classifier()

    inferred, _ = classifier.classify(Parameter("description", "string"))

    assert inferred == "text"


def test_declared_type_separates_an_int_id_from_a_string_id():
    """Boundary classes describe the value space, not the semantic role.

    `itemId:int` and `cardId:string` are both identifiers by name, but their
    boundaries differ entirely: an int id has zero/negative/max, a string id
    has empty/malformed. The declared type is what must decide this, so it is
    a feature of the classifier rather than an afterthought.
    """
    classifier = get_classifier()

    assert classifier.classify(Parameter("itemId", "int"))[0] == "numeric"
    assert classifier.classify(Parameter("cardId", "string"))[0] == "identifier"


def test_classifier_generalises_to_an_unseen_name_via_its_type():
    """An unrecognised name still gets a usable class from its declared type."""
    classifier = get_classifier()

    inferred, _ = classifier.classify(Parameter("wibbleFrobnicator", "int"))

    assert inferred == "numeric"


def test_proposal_lists_concrete_boundary_values():
    classifier = get_classifier()

    proposal = classifier.propose(Parameter("amount", "float"))

    cases = {c.case: c.value for c in proposal.candidates}
    assert "zero" in cases and "negative" in cases
    assert cases["zero"] == 0.0
    assert cases["negative"] < 0


def test_proposal_respects_the_declared_type_when_rendering_values():
    classifier = get_classifier()

    as_int = {c.case: c.value for c in classifier.propose(Parameter("qty", "int")).candidates}

    assert as_int["zero"] == 0
    assert isinstance(as_int["zero"], int)


def test_at_least_one_boundary_case_per_typed_parameter(graph):
    """Every typed parameter, on every path, must get boundary coverage."""
    cases = generate_boundary_cases(graph, "PlaceOrder")

    # Work out every parameter occurrence the diagram actually contains.
    expected = set()
    for path_number, path in enumerate(graph.enumerate_paths("PlaceOrder"), start=1):
        for step_number, interaction in enumerate(path, start=1):
            for parameter in interaction.parameters:
                expected.add((f"PlaceOrder.path{path_number}", step_number, parameter.name))

    covered = {(c.path_id, c.step_index, c.parameter.name) for c in cases}

    assert expected, "the fixture must contain typed parameters"
    assert expected <= covered
    assert all(isinstance(c, BoundaryTestCase) for c in cases)


def test_every_boundary_case_names_a_class_and_a_case(graph):
    cases = generate_boundary_cases(graph, "PlaceOrder")

    for case in cases:
        assert case.inferred_class
        assert case.boundary_case
        assert 0.0 < case.confidence <= 1.0


def test_boundary_case_ids_are_unique(graph):
    cases = generate_boundary_cases(graph, "PlaceOrder")

    assert len({c.id for c in cases}) == len(cases)


def test_boundary_cases_link_back_to_a_real_functional_path(graph):
    functional = generate_functional_cases(graph, "PlaceOrder")
    boundary = generate_boundary_cases(graph, "PlaceOrder")

    known_paths = {c.path_id for c in functional}

    assert {c.path_id for c in boundary} <= known_paths


def test_boundary_cases_inherit_the_preconditions_of_their_path(graph):
    """A boundary case only makes sense under the guards that reach its call."""
    cases = generate_boundary_cases(graph, "PlaceOrder")

    charge_cases = [c for c in cases if c.interaction.method == "charge"]

    assert charge_cases
    assert all(c.preconditions[0] == "stock_available" for c in charge_cases)


def test_amount_parameter_gets_numeric_boundary_cases(graph):
    cases = generate_boundary_cases(graph, "PlaceOrder")

    amount_cases = [c for c in cases if c.parameter.name == "amount"]

    assert amount_cases
    assert all(c.inferred_class == "numeric" for c in amount_cases)
    assert {"zero", "negative", "max_value"} <= {c.boundary_case for c in amount_cases}


def test_interaction_without_parameters_produces_no_boundary_cases(graph):
    cases = generate_boundary_cases(graph, "PlaceOrder")

    assert not [c for c in cases if c.interaction.method == "rejectOrder"]


def test_boundary_cases_export_to_a_table_of_dicts(graph):
    cases = generate_boundary_cases(graph, "PlaceOrder")

    table = to_table(cases)

    assert len(table) == len(cases)
    row = table[0]
    for key in (
        "id",
        "flow",
        "path_id",
        "parameter",
        "declared_type",
        "inferred_class",
        "boundary_case",
        "value",
        "confidence",
    ):
        assert key in row


def test_boundary_table_truncates_very_long_values():
    """A `very_long` string case must not blow up the exported table."""
    graph = build_graph(
        parse(
            dsl(
                """
                flow F:
                  A -> B : register(description:string)
                """
            )
        )
    )

    table = to_table(generate_boundary_cases(graph, "F"))

    assert all(len(row["value"]) <= 80 for row in table)


def test_inferred_class_is_a_plain_python_string(graph):
    """sklearn returns numpy scalars; they must not leak into the data model.

    A ``numpy.str_`` compares equal to ``str`` so most assertions pass anyway,
    but it serialises oddly and drags a numpy dependency into anything that
    consumes a test case.
    """
    classifier = get_classifier()
    inferred, confidence = classifier.classify(Parameter("amount", "float"))

    assert type(inferred) is str
    assert type(confidence) is float
    assert all(type(c) is str for c in classifier.classes)
    assert all(type(c.inferred_class) is str for c in generate_boundary_cases(graph))


def test_classifier_is_cached_between_calls():
    """Training is not free; the module-level classifier must be reused."""
    assert get_classifier() is get_classifier()


def test_classifier_can_be_constructed_independently():
    """The classifier is testable on its own, without any diagram."""
    classifier = ParameterClassifier()

    inferred, _ = classifier.classify(Parameter("qty", "int"))

    assert inferred == "numeric"


def test_generated_boundary_cases_are_deterministic(graph):
    """Two runs must produce identical suites, or diffs become unreviewable."""
    first = generate_boundary_cases(graph, "PlaceOrder")
    second = generate_boundary_cases(graph, "PlaceOrder")

    assert [c.id for c in first] == [c.id for c in second]
    assert [c.value for c in first] == [c.value for c in second]
