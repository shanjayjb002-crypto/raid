"""Unit tests for the RAID architecture DSL parser.

These tests pin down the contract of :func:`raid.dsl.parse`: given DSL source
text, it returns a :class:`~raid.dsl.model.Diagram` describing the declared
services and flows, or raises :class:`~raid.dsl.errors.DslSyntaxError` with a
human-readable message.
"""

import textwrap

import pytest

from raid.dsl import (
    Branch,
    Diagram,
    DslSyntaxError,
    Flow,
    Interaction,
    Parameter,
    Service,
    parse,
)


def dsl(source: str) -> str:
    """Strip the leading newline and common indentation from a test fixture."""
    return textwrap.dedent(source).lstrip("\n")


# --------------------------------------------------------------------------
# Service declarations
# --------------------------------------------------------------------------


def test_parses_service_declarations():
    diagram = parse(
        dsl(
            """
            service PaymentService
            service OrderService
            """
        )
    )

    assert isinstance(diagram, Diagram)
    assert [s.name for s in diagram.services] == ["PaymentService", "OrderService"]
    assert all(isinstance(s, Service) for s in diagram.services)


def test_service_records_source_line():
    diagram = parse(
        dsl(
            """
            service PaymentService
            service OrderService
            """
        )
    )

    assert [s.line for s in diagram.services] == [1, 2]


def test_empty_source_yields_empty_diagram():
    diagram = parse("")

    assert diagram.services == []
    assert diagram.flows == []


def test_blank_lines_between_declarations_are_ignored():
    diagram = parse(
        dsl(
            """
            service PaymentService


            service OrderService
            """
        )
    )

    assert [s.name for s in diagram.services] == ["PaymentService", "OrderService"]


# --------------------------------------------------------------------------
# Simple flows (no branching)
# --------------------------------------------------------------------------


def test_parses_simple_flow_without_branches():
    diagram = parse(
        dsl(
            """
            service OrderService
            service InventoryService

            flow PlaceOrder:
              OrderService -> InventoryService : checkStock(itemId:int, qty:int)
              OrderService -> InventoryService : reserve(itemId:int, qty:int)
            """
        )
    )

    assert len(diagram.flows) == 1
    flow = diagram.flows[0]
    assert isinstance(flow, Flow)
    assert flow.name == "PlaceOrder"
    assert len(flow.steps) == 2
    assert all(isinstance(step, Interaction) for step in flow.steps)

    first = flow.steps[0]
    assert first.source == "OrderService"
    assert first.target == "InventoryService"
    assert first.method == "checkStock"


def test_parses_interaction_parameters_with_types():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              OrderService -> PaymentService : charge(amount:float, cardId:string)
            """
        )
    )

    interaction = diagram.flows[0].steps[0]
    assert interaction.parameters == [
        Parameter(name="amount", type="float"),
        Parameter(name="cardId", type="string"),
    ]


def test_parses_untyped_parameters():
    """A parameter may omit its type, leaving it ``None``.

    Under-specified diagrams are the normal early state, and RAID's job is to
    score that incompleteness rather than refuse to read it - the same reading
    that lets an undeclared service through as a completeness problem instead
    of a parse error. If the grammar forced a type here, the sufficiency
    scorer's test-generation concern could never fire.
    """
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              OrderService -> PaymentService : charge(amount, cardId)
            """
        )
    )

    assert diagram.flows[0].steps[0].parameters == [
        Parameter(name="amount", type=None),
        Parameter(name="cardId", type=None),
    ]


def test_parses_a_mix_of_typed_and_untyped_parameters():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              OrderService -> PaymentService : charge(amount, cardId:string)
            """
        )
    )

    assert diagram.flows[0].steps[0].parameters == [
        Parameter(name="amount", type=None),
        Parameter(name="cardId", type="string"),
    ]


def test_parses_interaction_with_no_parameters():
    diagram = parse(
        dsl(
            """
            flow Reject:
              OrderService -> OrderService : rejectOrder()
            """
        )
    )

    interaction = diagram.flows[0].steps[0]
    assert interaction.method == "rejectOrder"
    assert interaction.parameters == []


def test_self_interaction_is_allowed():
    diagram = parse(
        dsl(
            """
            flow Reject:
              OrderService -> OrderService : rejectOrder()
            """
        )
    )

    interaction = diagram.flows[0].steps[0]
    assert interaction.source == interaction.target == "OrderService"


def test_interaction_records_source_line():
    diagram = parse(
        dsl(
            """
            service OrderService

            flow PlaceOrder:
              OrderService -> OrderService : rejectOrder()
            """
        )
    )

    assert diagram.flows[0].steps[0].line == 4


# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------


def test_parses_annotations_on_an_interaction():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              OrderService -> PaymentService : charge(amount:float) [retry=3, timeout=2s]
            """
        )
    )

    interaction = diagram.flows[0].steps[0]
    assert interaction.annotations == {"retry": 3, "timeout": "2s"}


def test_annotation_values_are_coerced_by_kind():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              A -> B : m() [retries=3, rate=0.5, timeout=250ms, mode=exponential, note="be careful"]
            """
        )
    )

    assert diagram.flows[0].steps[0].annotations == {
        "retries": 3,
        "rate": 0.5,
        "timeout": "250ms",
        "mode": "exponential",
        "note": "be careful",
    }


def test_interaction_without_annotations_has_empty_mapping():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              A -> B : m()
            """
        )
    )

    assert diagram.flows[0].steps[0].annotations == {}


def test_annotations_are_allowed_on_an_interaction_inside_a_branch():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              alt stock_available:
                A -> B : charge(amount:float) [retry=3]
            """
        )
    )

    branch = diagram.flows[0].steps[0]
    assert branch.steps[0].annotations == {"retry": 3}


# --------------------------------------------------------------------------
# Branching
# --------------------------------------------------------------------------


def test_parses_alt_without_else():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              alt stock_available:
                A -> B : reserve(itemId:int)
            """
        )
    )

    branch = diagram.flows[0].steps[0]
    assert isinstance(branch, Branch)
    assert branch.condition == "stock_available"
    assert len(branch.steps) == 1
    assert branch.else_condition is None
    assert branch.else_steps == []


def test_parses_alt_with_else():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              alt stock_available:
                A -> B : reserve(itemId:int)
              else out_of_stock:
                A -> A : rejectOrder()
            """
        )
    )

    branch = diagram.flows[0].steps[0]
    assert branch.condition == "stock_available"
    assert branch.else_condition == "out_of_stock"
    assert [s.method for s in branch.steps] == ["reserve"]
    assert [s.method for s in branch.else_steps] == ["rejectOrder"]


def test_parses_nested_alt_else_from_the_reference_example():
    """The worked example from the project specification must round-trip."""
    diagram = parse(
        dsl(
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
    )

    assert [s.name for s in diagram.services] == [
        "PaymentService",
        "OrderService",
        "InventoryService",
    ]

    flow = diagram.flows[0]
    assert flow.name == "PlaceOrder"
    assert len(flow.steps) == 2

    check_stock, outer = flow.steps
    assert check_stock.method == "checkStock"
    assert isinstance(outer, Branch)

    # Outer alt body: one interaction, then a nested alt.
    assert outer.condition == "stock_available"
    charge, inner = outer.steps
    assert charge.method == "charge"
    assert charge.parameters == [
        Parameter(name="amount", type="float"),
        Parameter(name="cardId", type="string"),
    ]

    # Nested alt has no else of its own.
    assert isinstance(inner, Branch)
    assert inner.condition == "payment_success"
    assert [s.method for s in inner.steps] == ["reserve"]
    assert inner.else_condition is None
    assert inner.else_steps == []

    # The else belongs to the OUTER alt, not the nested one.
    assert outer.else_condition == "out_of_stock"
    assert [s.method for s in outer.else_steps] == ["rejectOrder"]


def test_branch_records_source_line():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              A -> B : m()
              alt ok:
                A -> B : n()
            """
        )
    )

    assert diagram.flows[0].steps[1].line == 3


def test_branch_may_contain_multiple_steps():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              alt ok:
                A -> B : one()
                A -> B : two()
              else bad:
                A -> B : three()
                A -> B : four()
            """
        )
    )

    branch = diagram.flows[0].steps[0]
    assert [s.method for s in branch.steps] == ["one", "two"]
    assert [s.method for s in branch.else_steps] == ["three", "four"]


def test_branch_may_follow_a_branch_at_the_same_level():
    diagram = parse(
        dsl(
            """
            flow PlaceOrder:
              alt first:
                A -> B : one()
              alt second:
                A -> B : two()
            """
        )
    )

    steps = diagram.flows[0].steps
    assert len(steps) == 2
    assert [b.condition for b in steps] == ["first", "second"]


# --------------------------------------------------------------------------
# Multiple flows
# --------------------------------------------------------------------------


def test_parses_multiple_flows():
    diagram = parse(
        dsl(
            """
            service OrderService
            service InventoryService

            flow PlaceOrder:
              OrderService -> InventoryService : checkStock(itemId:int)
              alt stock_available:
                OrderService -> InventoryService : reserve(itemId:int)

            flow CancelOrder:
              OrderService -> InventoryService : release(itemId:int)

            flow AuditOrder:
              OrderService -> OrderService : writeAuditLog(orderId:string)
            """
        )
    )

    assert [f.name for f in diagram.flows] == ["PlaceOrder", "CancelOrder", "AuditOrder"]
    assert len(diagram.flows[0].steps) == 2
    assert len(diagram.flows[1].steps) == 1
    assert diagram.flows[2].steps[0].method == "writeAuditLog"


def test_flows_do_not_leak_steps_into_each_other():
    diagram = parse(
        dsl(
            """
            flow First:
              A -> B : one()

            flow Second:
              A -> B : two()
            """
        )
    )

    assert [s.method for s in diagram.flows[0].steps] == ["one"]
    assert [s.method for s in diagram.flows[1].steps] == ["two"]


# --------------------------------------------------------------------------
# Error reporting
# --------------------------------------------------------------------------


def test_missing_arrow_raises_clear_parse_error():
    source = dsl(
        """
        service OrderService

        flow PlaceOrder:
          OrderService InventoryService : checkStock(itemId:int)
        """
    )

    with pytest.raises(DslSyntaxError) as excinfo:
        parse(source)

    message = str(excinfo.value)
    assert "line 4" in message
    assert "InventoryService" in message


def test_parse_error_message_is_readable_and_not_a_lark_dump():
    source = dsl(
        """
        flow PlaceOrder:
          OrderService InventoryService : checkStock()
        """
    )

    with pytest.raises(DslSyntaxError) as excinfo:
        parse(source)

    message = str(excinfo.value)
    # A human-facing message: names the language, the place, and what it wanted.
    assert message.startswith("RAID DSL syntax error")
    assert "Expected" in message
    # It must not leak lark's internal rule/state machinery at the user.
    assert "Traceback" not in message
    assert "__" not in message


def test_flow_without_a_body_raises_parse_error():
    source = dsl(
        """
        flow PlaceOrder:
        service OrderService
        """
    )

    with pytest.raises(DslSyntaxError):
        parse(source)


def test_flow_without_a_body_says_an_indented_block_was_expected():
    """The message must point at the real fix (indent the body), not a newline."""
    source = dsl(
        """
        flow PlaceOrder:
        service OrderService
        """
    )

    with pytest.raises(DslSyntaxError) as excinfo:
        parse(source)

    assert "indented block" in str(excinfo.value)


def test_error_at_end_of_document_reports_a_usable_location():
    """A construct left open at EOF has no token position; say so in words."""
    source = dsl(
        """
        flow PlaceOrder:
          A -> B : checkStock(itemId:int
        """
    )

    with pytest.raises(DslSyntaxError) as excinfo:
        parse(source)

    message = str(excinfo.value)
    assert "end of the document" in message
    assert "unknown" not in message


def test_unknown_keyword_raises_parse_error():
    source = dsl(
        """
        component OrderService
        """
    )

    with pytest.raises(DslSyntaxError):
        parse(source)


def test_unterminated_parameter_list_raises_parse_error():
    source = dsl(
        """
        flow PlaceOrder:
          A -> B : checkStock(itemId:int
        """
    )

    with pytest.raises(DslSyntaxError):
        parse(source)


def test_inconsistent_indentation_raises_parse_error():
    source = dsl(
        """
        flow PlaceOrder:
          alt ok:
              A -> B : one()
            A -> B : two()
        """
    )

    with pytest.raises(DslSyntaxError):
        parse(source)
