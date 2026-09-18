"""Unit tests for the mutation engine and chaos test spec writer.

Fuzzing is randomised, so "reliably surfaces a finding" is asserted across a
range of seeds rather than on one lucky run - a single-seed test would pass
while the engine was in fact flaky.
"""

import textwrap

import pytest

from raid.dsl import parse
from raid.graph import build_graph
from raid.fuzz import (
    ChaosTestSpec,
    Mutation,
    MutationTargetError,
    apply_mutation,
    fuzz_and_report,
    generate_mutations,
)
from raid.simulate import SimulationConfig


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


FAST = SimulationConfig(runs=20, seed=7)


# ==========================================================================
# The headline requirement
# ==========================================================================


@pytest.mark.parametrize("seed", range(10))
def test_payment_service_fuzzing_reliably_surfaces_a_chaos_spec(graph, seed):
    """Across every seed tried, mutating PaymentService must find something."""
    specs = fuzz_and_report(
        graph,
        "PlaceOrder",
        num_mutations=12,
        seed=seed,
        targets=["PaymentService"],
        config=FAST,
    )

    assert specs
    assert all(isinstance(s, ChaosTestSpec) for s in specs)


def test_a_payment_kill_spec_reports_the_charge_timeout(graph):
    specs = fuzz_and_report(
        graph, "PlaceOrder", num_mutations=12, seed=0, targets=["PaymentService"], config=FAST
    )

    kills = [s for s in specs if s.mutation.kind == "kill_service"]

    assert kills
    spec = kills[0]
    assert "timeout" in spec.regressions
    assert "charge" in spec.observed_behaviour
    assert "2000" in spec.observed_behaviour


def test_spec_describes_the_mutation_in_plain_words(graph):
    specs = fuzz_and_report(
        graph, "PlaceOrder", num_mutations=12, seed=0, targets=["PaymentService"], config=FAST
    )

    kills = [s for s in specs if s.mutation.kind == "kill_service"]

    assert kills
    assert kills[0].description.startswith("kill PaymentService for ")


def test_spec_contrasts_baseline_against_what_actually_happened(graph):
    specs = fuzz_and_report(
        graph, "PlaceOrder", num_mutations=12, seed=0, targets=["PaymentService"], config=FAST
    )

    spec = specs[0]

    assert spec.baseline_behaviour
    assert spec.observed_behaviour
    assert spec.baseline_behaviour != spec.observed_behaviour
    # The baseline for this diagram is healthy, and the spec must say so.
    assert "no timeouts" in spec.baseline_behaviour


def test_every_spec_names_at_least_one_regression(graph):
    specs = fuzz_and_report(
        graph, "PlaceOrder", num_mutations=15, seed=3, targets=["PaymentService"], config=FAST
    )

    assert specs
    assert all(s.regressions for s in specs)


def test_specs_are_ranked_most_severe_first(graph):
    specs = fuzz_and_report(
        graph, "PlaceOrder", num_mutations=15, seed=1, targets=["PaymentService"], config=FAST
    )

    severities = [s.severity for s in specs]
    assert severities == sorted(severities, reverse=True)


def test_fuzzing_is_deterministic_for_a_fixed_seed(graph):
    kwargs = dict(
        num_mutations=10, seed=5, targets=["PaymentService"], config=FAST
    )

    first = fuzz_and_report(graph, "PlaceOrder", **kwargs)
    second = fuzz_and_report(graph, "PlaceOrder", **kwargs)

    assert [s.id for s in first] == [s.id for s in second]
    assert [s.description for s in first] == [s.description for s in second]


def test_requesting_no_mutations_yields_no_specs(graph):
    assert fuzz_and_report(graph, "PlaceOrder", num_mutations=0, config=FAST) == []


def test_specs_export_to_flat_dicts(graph):
    specs = fuzz_and_report(
        graph, "PlaceOrder", num_mutations=10, seed=0, targets=["PaymentService"], config=FAST
    )

    row = specs[0].to_dict()

    for key in (
        "id",
        "flow",
        "description",
        "mutation_kind",
        "regressions",
        "baseline_behaviour",
        "observed_behaviour",
        "severity",
    ):
        assert key in row
    for value in row.values():
        assert isinstance(value, (str, int, float, bool)) or value is None


# ==========================================================================
# Mutation generation
# ==========================================================================


def test_all_four_mutation_kinds_can_be_generated(graph):
    mutations = generate_mutations(
        graph, "PlaceOrder", count=200, seed=0, targets=["PaymentService"]
    )

    assert {m.kind for m in mutations} == {
        "kill_service",
        "remove_edge",
        "inject_cycle",
        "inflate_latency",
    }


def test_generated_mutations_respect_the_requested_target(graph):
    mutations = generate_mutations(
        graph, "PlaceOrder", count=30, seed=0, targets=["PaymentService"]
    )

    assert all(m.service == "PaymentService" for m in mutations)


def test_generation_count_is_honoured(graph):
    assert len(generate_mutations(graph, "PlaceOrder", count=7, seed=0)) == 7


def test_generated_mutations_are_distinct(graph):
    """Fields a mutation kind ignores must not make two identical faults differ.

    A `kill_service` does not use `multiplier`, so two kills of the same
    service for the same duration are the same experiment however the unused
    field was sampled. Without normalising, the engine re-simulates identical
    mutants and writes duplicate specs.
    """
    mutations = generate_mutations(
        graph, "PlaceOrder", count=40, seed=0, targets=["PaymentService"]
    )

    assert len(set(mutations)) == len(mutations)
    assert len({m.description for m in mutations}) == len(mutations)


def test_generation_stops_when_the_mutation_space_is_exhausted(graph):
    """Only one distinct edge removal exists for a single call."""
    mutations = generate_mutations(
        graph,
        "PlaceOrder",
        count=25,
        seed=0,
        targets=["PaymentService"],
        kinds=["remove_edge"],
    )

    assert len(mutations) == 1


def test_fuzzing_does_not_write_duplicate_specs(graph):
    specs = fuzz_and_report(
        graph, "PlaceOrder", num_mutations=25, seed=0, targets=["PaymentService"], config=FAST
    )

    assert len({s.description for s in specs}) == len(specs)


def test_unknown_target_service_is_rejected(graph):
    with pytest.raises(MutationTargetError) as excinfo:
        generate_mutations(graph, "PlaceOrder", count=5, seed=0, targets=["PaymntService"])

    assert "PaymntService" in str(excinfo.value)


def test_a_service_that_is_only_a_caller_is_rejected(graph):
    """OrderService never receives a call on the happy path, so mutating it is a no-op."""
    with pytest.raises(MutationTargetError) as excinfo:
        generate_mutations(graph, "PlaceOrder", count=5, seed=0, targets=["OrderService"])

    message = str(excinfo.value)
    assert "OrderService" in message
    assert "InventoryService" in message or "PaymentService" in message


# ==========================================================================
# Applying mutations
# ==========================================================================


def test_kill_service_becomes_an_injected_outage(graph):
    mutation = Mutation(
        kind="kill_service", service="PaymentService", duration=5000.0
    )

    mutated, failures = apply_mutation(graph, "PlaceOrder", mutation)

    assert mutated is graph
    assert len(failures) == 1
    assert failures[0].service == "PaymentService"
    assert failures[0].mode == "outage"
    assert failures[0].duration == 5000.0


def test_inflate_latency_becomes_an_injected_brownout(graph):
    mutation = Mutation(
        kind="inflate_latency", service="PaymentService", multiplier=25.0
    )

    _, failures = apply_mutation(graph, "PlaceOrder", mutation)

    assert failures[0].mode == "slow"
    assert failures[0].latency_multiplier == 25.0


def test_remove_edge_drops_the_call_from_the_flow(graph):
    mutation = Mutation(
        kind="remove_edge", service="PaymentService", method="charge", line=8
    )

    mutated, failures = apply_mutation(graph, "PlaceOrder", mutation)

    assert failures == ()
    methods = [i.method for i in mutated.enumerate_paths("PlaceOrder")[0]]
    assert methods == ["checkStock", "reserve"]


def test_inject_cycle_adds_a_reverse_dependency(graph):
    mutation = Mutation(
        kind="inject_cycle", service="PaymentService", method="charge", line=8
    )

    mutated, failures = apply_mutation(graph, "PlaceOrder", mutation)

    assert failures == ()
    assert mutated.graph.has_edge("PaymentService", "OrderService")
    methods = [i.method for i in mutated.enumerate_paths("PlaceOrder")[0]]
    assert len(methods) == 4


def test_applying_a_structural_mutation_leaves_the_original_graph_untouched(graph):
    """Mutants must be copies; corrupting the input would poison later runs."""
    before_edges = graph.graph.number_of_edges()
    before_methods = [i.method for i in graph.enumerate_paths("PlaceOrder")[0]]

    apply_mutation(
        graph,
        "PlaceOrder",
        Mutation(kind="remove_edge", service="PaymentService", method="charge", line=8),
    )
    apply_mutation(
        graph,
        "PlaceOrder",
        Mutation(kind="inject_cycle", service="PaymentService", method="charge", line=8),
    )

    assert graph.graph.number_of_edges() == before_edges
    assert [i.method for i in graph.enumerate_paths("PlaceOrder")[0]] == before_methods


# ==========================================================================
# Regression detection
# ==========================================================================


def test_removing_a_dependency_is_benign_and_reports_nothing(graph):
    """A negative control: dropping a call makes the flow faster, not worse.

    A mutation engine that reported a finding for every mutation would be
    rubber-stamping rather than detecting, so this must stay empty.
    """
    specs = fuzz_and_report(
        graph,
        "PlaceOrder",
        num_mutations=5,
        seed=0,
        targets=["PaymentService"],
        kinds=["remove_edge"],
        config=FAST,
    )

    assert specs == []


def test_injecting_a_cycle_is_reported_as_a_structural_regression(graph):
    specs = fuzz_and_report(
        graph,
        "PlaceOrder",
        num_mutations=5,
        seed=0,
        targets=["PaymentService"],
        kinds=["inject_cycle"],
        config=FAST,
    )

    assert specs
    assert "cycle" in specs[0].regressions
    assert "PaymentService" in specs[0].observed_behaviour


def test_a_severe_brownout_is_reported_as_a_timeout_regression(graph):
    specs = fuzz_and_report(
        graph,
        "PlaceOrder",
        num_mutations=6,
        seed=0,
        targets=["PaymentService"],
        kinds=["inflate_latency"],
        config=FAST,
    )

    assert specs
    assert any("timeout" in s.regressions or "latency" in s.regressions for s in specs)


def test_killing_a_service_breaks_every_run(graph):
    specs = fuzz_and_report(
        graph,
        "PlaceOrder",
        num_mutations=6,
        seed=0,
        targets=["PaymentService"],
        kinds=["kill_service"],
        config=FAST,
    )

    biting = [s for s in specs if "cascade" in s.regressions]

    assert biting
    assert biting[0].mutant_report.completed_runs == 0
    assert biting[0].baseline_report.completed_runs == FAST.runs
