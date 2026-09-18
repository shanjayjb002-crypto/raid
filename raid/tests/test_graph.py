"""Unit tests for the RAID graph model and path enumeration.

The expected execution paths in these tests are worked out by hand from the
diagram source and asserted element by element, rather than by counting. A
count assertion would pass for the wrong set of paths, which is precisely the
failure mode that matters here: path enumeration feeds test generation, so a
silently missing path means a silently missing test.
"""

import textwrap

import pytest

from raid.dsl import parse
from raid.graph import ExecutionPath, RaidGraph, UnknownFlowError, build_graph


def dsl(source: str) -> str:
    """Strip the leading newline and common indentation from a test fixture."""
    return textwrap.dedent(source).lstrip("\n")


# The worked example carried forward from phase 1.
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


def build(source: str) -> RaidGraph:
    return build_graph(parse(source))


def methods(path: ExecutionPath) -> list[str]:
    return [interaction.method for interaction in path]


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------


def test_each_declared_service_becomes_a_node():
    graph = build(PLACE_ORDER)

    assert set(graph.graph.nodes) == {
        "PaymentService",
        "OrderService",
        "InventoryService",
    }


def test_declared_services_are_marked_declared():
    graph = build(PLACE_ORDER)

    assert all(graph.graph.nodes[n]["declared"] for n in graph.graph.nodes)


def test_service_used_but_never_declared_is_added_and_flagged():
    """Undeclared services are a completeness problem, not a parse failure."""
    graph = build(
        dsl(
            """
            service OrderService

            flow PlaceOrder:
              OrderService -> GhostService : ping()
            """
        )
    )

    assert "GhostService" in graph.graph.nodes
    assert graph.graph.nodes["GhostService"]["declared"] is False
    assert graph.graph.nodes["OrderService"]["declared"] is True


def test_each_interaction_becomes_a_directed_edge():
    graph = build(PLACE_ORDER)

    assert graph.graph.number_of_edges() == 4
    assert graph.graph.has_edge("OrderService", "InventoryService")
    assert graph.graph.has_edge("OrderService", "PaymentService")
    # Direction matters: nothing calls back into OrderService from Payment.
    assert not graph.graph.has_edge("PaymentService", "OrderService")


def test_parallel_edges_between_the_same_pair_are_kept_separately():
    """checkStock and reserve both run OrderService -> InventoryService."""
    graph = build(PLACE_ORDER)

    edges = graph.graph["OrderService"]["InventoryService"]
    assert sorted(data["method"] for data in edges.values()) == [
        "checkStock",
        "reserve",
    ]


def test_self_interaction_becomes_a_self_loop():
    graph = build(PLACE_ORDER)

    edges = graph.graph["OrderService"]["OrderService"]
    assert [data["method"] for data in edges.values()] == ["rejectOrder"]


def test_edge_carries_method_parameters_and_flow():
    graph = build(PLACE_ORDER)

    charge = next(
        data
        for _, _, data in graph.graph.edges(data=True)
        if data["method"] == "charge"
    )
    assert charge["flow"] == "PlaceOrder"
    assert [(p.name, p.type) for p in charge["parameters"]] == [
        ("amount", "float"),
        ("cardId", "string"),
    ]
    # Line 8 of PLACE_ORDER: 3 service decls, a blank, `flow`, checkStock, alt.
    assert charge["line"] == 8


def test_edge_carries_annotations():
    graph = build(
        dsl(
            """
            service A
            service B

            flow F:
              A -> B : charge(amount:float) [retry=3, timeout=2s]
            """
        )
    )

    _, _, data = next(iter(graph.graph.edges(data=True)))
    assert data["annotations"] == {"retry": 3, "timeout": "2s"}


def test_edge_records_the_branch_conditions_it_sits_under():
    """Branch context lives on the edge, so the graph itself represents alt/else."""
    graph = build(PLACE_ORDER)

    by_method = {
        data["method"]: data["conditions"]
        for _, _, data in graph.graph.edges(data=True)
    }
    assert by_method["checkStock"] == ()
    assert by_method["charge"] == ("stock_available",)
    assert by_method["reserve"] == ("stock_available", "payment_success")
    assert by_method["rejectOrder"] == ("out_of_stock",)


def test_flows_are_available_by_name():
    graph = build(PLACE_ORDER)

    assert list(graph.flows) == ["PlaceOrder"]


# --------------------------------------------------------------------------
# Path enumeration - the PlaceOrder example, worked out by hand
# --------------------------------------------------------------------------
#
#   checkStock                          always runs
#   alt stock_available:
#     charge
#     alt payment_success:              no else -> may simply not fire
#       reserve
#   else out_of_stock:
#     rejectOrder
#
# Taking the outer alt's two arms, and the inner alt's taken/not-taken arms:
#
#   1. stock_available + payment_success       -> checkStock, charge, reserve
#   2. stock_available + NOT payment_success   -> checkStock, charge
#   3. out_of_stock                            -> checkStock, rejectOrder
#
# Three paths. Note path 2 exists only because an `alt` with no `else` still
# has an implicit "guard was false" branch - that is the payment-failure case,
# exactly the path a resilience test suite must cover.
# --------------------------------------------------------------------------


def test_place_order_enumerates_exactly_the_three_expected_paths():
    graph = build(PLACE_ORDER)

    paths = graph.enumerate_paths("PlaceOrder")

    assert [methods(p) for p in paths] == [
        ["checkStock", "charge", "reserve"],
        ["checkStock", "charge"],
        ["checkStock", "rejectOrder"],
    ]


def test_place_order_paths_carry_the_conditions_that_select_them():
    graph = build(PLACE_ORDER)

    paths = graph.enumerate_paths("PlaceOrder")

    assert [list(p.conditions) for p in paths] == [
        ["stock_available", "payment_success"],
        ["stock_available", "not payment_success"],
        ["out_of_stock"],
    ]


def test_place_order_path_interactions_are_the_real_ast_objects():
    """A path must carry full interaction detail, not just method names."""
    graph = build(PLACE_ORDER)

    happy = graph.enumerate_paths("PlaceOrder")[0]
    checkstock, charge, reserve = happy.interactions

    assert (checkstock.source, checkstock.target) == ("OrderService", "InventoryService")
    assert (charge.source, charge.target) == ("OrderService", "PaymentService")
    assert (reserve.source, reserve.target) == ("OrderService", "InventoryService")
    assert [(p.name, p.type) for p in charge.parameters] == [
        ("amount", "float"),
        ("cardId", "string"),
    ]


def test_rejected_path_never_touches_the_payment_service():
    graph = build(PLACE_ORDER)

    rejected = graph.enumerate_paths("PlaceOrder")[2]

    assert "PaymentService" not in {i.target for i in rejected}


# --------------------------------------------------------------------------
# Path enumeration - branching rules in isolation
# --------------------------------------------------------------------------


def test_flow_without_branches_has_exactly_one_path():
    graph = build(
        dsl(
            """
            flow Simple:
              A -> B : one()
              B -> C : two()
            """
        )
    )

    paths = graph.enumerate_paths("Simple")

    assert [methods(p) for p in paths] == [["one", "two"]]
    assert paths[0].conditions == ()


def test_alt_with_else_yields_one_path_per_arm():
    graph = build(
        dsl(
            """
            flow F:
              alt ok:
                A -> B : yes()
              else bad:
                A -> B : no()
            """
        )
    )

    paths = graph.enumerate_paths("F")

    assert [methods(p) for p in paths] == [["yes"], ["no"]]
    assert [list(p.conditions) for p in paths] == [["ok"], ["bad"]]


def test_alt_without_else_yields_a_taken_and_a_skipped_path():
    graph = build(
        dsl(
            """
            flow F:
              A -> B : before()
              alt ok:
                A -> B : inside()
              A -> B : after()
            """
        )
    )

    paths = graph.enumerate_paths("F")

    assert [methods(p) for p in paths] == [
        ["before", "inside", "after"],
        ["before", "after"],
    ]
    assert [list(p.conditions) for p in paths] == [["ok"], ["not ok"]]


def test_sequential_branches_multiply_paths():
    """Two independent alt/else fragments in sequence give 2 x 2 = 4 paths."""
    graph = build(
        dsl(
            """
            flow F:
              alt a:
                A -> B : one()
              else not_a:
                A -> B : two()
              alt b:
                A -> B : three()
              else not_b:
                A -> B : four()
            """
        )
    )

    paths = graph.enumerate_paths("F")

    assert [methods(p) for p in paths] == [
        ["one", "three"],
        ["one", "four"],
        ["two", "three"],
        ["two", "four"],
    ]
    assert [list(p.conditions) for p in paths] == [
        ["a", "b"],
        ["a", "not_b"],
        ["not_a", "b"],
        ["not_a", "not_b"],
    ]


def test_deeply_nested_branches_enumerate_correctly():
    """Three levels of nesting, each arm named, gives one path per leaf."""
    graph = build(
        dsl(
            """
            flow F:
              alt a:
                alt b:
                  A -> B : ab()
                else not_b:
                  A -> B : anb()
              else not_a:
                alt c:
                  A -> B : nac()
                else not_c:
                  A -> B : nanc()
            """
        )
    )

    paths = graph.enumerate_paths("F")

    assert [methods(p) for p in paths] == [["ab"], ["anb"], ["nac"], ["nanc"]]
    assert [list(p.conditions) for p in paths] == [
        ["a", "b"],
        ["a", "not_b"],
        ["not_a", "c"],
        ["not_a", "not_c"],
    ]


def test_steps_after_a_branch_appear_on_every_path():
    graph = build(
        dsl(
            """
            flow F:
              alt ok:
                A -> B : inside()
              else bad:
                A -> B : other()
              A -> B : always()
            """
        )
    )

    paths = graph.enumerate_paths("F")

    assert [methods(p) for p in paths] == [["inside", "always"], ["other", "always"]]


def test_each_flow_enumerates_independently():
    graph = build(
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

    assert [methods(p) for p in graph.enumerate_paths("First")] == [["one"]]
    assert [methods(p) for p in graph.enumerate_paths("Second")] == [["two"], ["three"]]


# --------------------------------------------------------------------------
# ExecutionPath behaviour and error handling
# --------------------------------------------------------------------------


def test_execution_path_is_an_ordered_sequence_of_interactions():
    graph = build(PLACE_ORDER)

    path = graph.enumerate_paths("PlaceOrder")[0]

    assert len(path) == 3
    assert [i.method for i in path] == ["checkStock", "charge", "reserve"]
    assert list(path) == list(path.interactions)


def test_enumerate_paths_on_unknown_flow_raises_a_clear_error():
    graph = build(PLACE_ORDER)

    with pytest.raises(UnknownFlowError) as excinfo:
        graph.enumerate_paths("NoSuchFlow")

    message = str(excinfo.value)
    assert "NoSuchFlow" in message
    # The message should help by naming what *is* available.
    assert "PlaceOrder" in message


def test_empty_diagram_builds_an_empty_graph():
    graph = build("")

    assert graph.graph.number_of_nodes() == 0
    assert graph.flows == {}
