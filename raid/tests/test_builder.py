"""Unit tests for the visual builder's DSL compiler.

The builder is an input layer only, so the contract is narrow and strict: what
it emits must be byte-identical to the equivalent hand-written DSL, must always
parse, and must degrade to a warning rather than to broken text while a form is
half filled in.
"""

import pytest

from raid.dashboard.builder import (
    LOGIN_DSL,
    VisualDiagram,
    VisualFlow,
    VisualInteraction,
    VisualParameter,
    compile_dsl,
    login_example,
    validate,
)
from raid.dsl import parse
from raid.graph import build_graph


def interaction(**kwargs) -> VisualInteraction:
    base = dict(source="A", target="B", method="call")
    base.update(kwargs)
    return VisualInteraction(**base)


def one_flow(*interactions, services=("A", "B")) -> VisualDiagram:
    return VisualDiagram(
        services=list(services),
        flows=[VisualFlow(name="F", interactions=list(interactions))],
    )


# ==========================================================================
# The headline requirement: both input modes produce identical text
# ==========================================================================


def test_visual_login_example_compiles_to_the_shared_dsl_exactly():
    """Building Login visually must equal the text the code editor opens with."""
    assert compile_dsl(login_example()) == LOGIN_DSL


def test_the_shared_dsl_parses():
    diagram = parse(LOGIN_DSL)

    assert [s.name for s in diagram.services] == ["Client", "AuthService", "UserStore"]
    assert [i.method for i in diagram.flows[0].steps] == [
        "authenticate",
        "loadProfile",
        "issueToken",
    ]


def test_compiled_output_round_trips_through_the_real_parser():
    graph = build_graph(parse(compile_dsl(login_example())))

    assert set(graph.graph.nodes) == {"Client", "AuthService", "UserStore"}
    assert [i.method for i in graph.enumerate_paths("Login")[0]] == [
        "authenticate",
        "loadProfile",
        "issueToken",
    ]


def test_annotations_survive_the_round_trip():
    diagram = parse(compile_dsl(login_example()))
    first = diagram.flows[0].steps[0]

    assert first.annotations == {"timeout": "1000ms", "retry": 2}


def test_the_populated_example_raises_no_warnings():
    assert validate(login_example()) == []


# ==========================================================================
# Emission details
# ==========================================================================


def test_services_and_flows_are_separated_by_a_blank_line():
    text = compile_dsl(one_flow(interaction(method="ping")))

    assert text == "service A\nservice B\n\nflow F:\n  A -> B : ping()\n"


def test_parameters_render_with_their_types():
    text = compile_dsl(
        one_flow(
            interaction(
                parameters=[VisualParameter("amount", "float"), VisualParameter("id", "int")]
            )
        )
    )

    assert "call(amount:float, id:int)" in text


def test_an_untyped_parameter_renders_without_a_colon():
    text = compile_dsl(one_flow(interaction(parameters=[VisualParameter("amount", None)])))

    assert "call(amount)" in text
    assert parse(text).flows[0].steps[0].parameters[0].type is None


def test_timeout_is_emitted_in_milliseconds():
    text = compile_dsl(one_flow(interaction(timeout_ms=250)))

    assert "[timeout=250ms]" in text


def test_retry_alone_is_emitted():
    text = compile_dsl(one_flow(interaction(retry=4)))

    assert "[retry=4]" in text


def test_both_annotations_share_one_bracket():
    text = compile_dsl(one_flow(interaction(timeout_ms=100, retry=2)))

    assert "[timeout=100ms, retry=2]" in text


def test_no_annotations_means_no_brackets():
    text = compile_dsl(one_flow(interaction()))

    assert "[" not in text


# ==========================================================================
# Branch grouping - the user never types alt or else
# ==========================================================================


def test_a_tagged_interaction_becomes_an_alt_block():
    text = compile_dsl(one_flow(interaction(method="charge", condition="ok", arm="alt")))

    assert text.endswith("flow F:\n  alt ok:\n    A -> B : charge()\n")


def test_consecutive_rows_sharing_a_condition_group_into_one_block():
    text = compile_dsl(
        one_flow(
            interaction(method="one", condition="ok", arm="alt"),
            interaction(method="two", condition="ok", arm="alt"),
        )
    )

    assert "  alt ok:\n    A -> B : one()\n    A -> B : two()\n" in text
    assert text.count("alt ok:") == 1


def test_an_alternative_arm_becomes_an_else_block():
    text = compile_dsl(
        one_flow(
            interaction(method="yes", condition="granted", arm="alt"),
            interaction(method="no", condition="denied", arm="else"),
        )
    )

    assert "  alt granted:\n    A -> B : yes()\n  else denied:\n    A -> B : no()\n" in text


def test_the_generated_branch_enumerates_as_two_paths():
    text = compile_dsl(
        one_flow(
            interaction(method="yes", condition="granted", arm="alt"),
            interaction(method="no", condition="denied", arm="else"),
        )
    )
    graph = build_graph(parse(text))

    assert [
        [i.method for i in path] for path in graph.enumerate_paths("F")
    ] == [["yes"], ["no"]]


def test_untagged_rows_stay_at_the_top_level_around_a_branch():
    text = compile_dsl(
        one_flow(
            interaction(method="before"),
            interaction(method="inside", condition="ok", arm="alt"),
            interaction(method="after"),
        )
    )

    assert text.endswith(
        "flow F:\n"
        "  A -> B : before()\n"
        "  alt ok:\n"
        "    A -> B : inside()\n"
        "  A -> B : after()\n"
    )


def test_two_separate_conditions_become_two_branches():
    text = compile_dsl(
        one_flow(
            interaction(method="one", condition="first", arm="alt"),
            interaction(method="two", condition="second", arm="alt"),
        )
    )

    assert "alt first:" in text and "alt second:" in text
    assert "else" not in text


def test_an_orphan_else_is_written_as_its_own_branch_and_warned_about():
    """An alternative arm with no main arm before it must still parse."""
    diagram = one_flow(interaction(method="only", condition="denied", arm="else"))

    text = compile_dsl(diagram)

    assert "alt denied:" in text
    parse(text)  # must not raise
    assert any("no main branch" in w for w in validate(diagram))


# ==========================================================================
# Half-finished and invalid states must never emit broken DSL
# ==========================================================================


def test_an_empty_model_compiles_to_empty_text():
    assert compile_dsl(VisualDiagram()) == ""


def test_an_incomplete_interaction_is_omitted_not_emitted_broken():
    diagram = one_flow(interaction(method=""))

    text = compile_dsl(diagram)

    assert "flow F" not in text
    parse(text)
    assert any("incomplete" in w for w in validate(diagram))


def test_a_service_name_with_a_space_is_omitted_and_warned_about():
    diagram = VisualDiagram(services=["Auth Service"], flows=[])

    assert compile_dsl(diagram) == ""
    assert any("not usable" in w for w in validate(diagram))


def test_a_reserved_word_service_name_is_rejected():
    diagram = VisualDiagram(services=["alt"], flows=[])

    assert compile_dsl(diagram) == ""
    assert any("not usable" in w for w in validate(diagram))


def test_an_interaction_referencing_a_deleted_service_still_parses_and_warns():
    """The demo case: a service is deleted while an interaction still uses it."""
    diagram = one_flow(interaction(source="A", target="Deleted"), services=["A"])

    text = compile_dsl(diagram)

    parse(text)  # must not raise
    assert "A -> Deleted : call()" in text
    assert any("not a declared service" in w for w in validate(diagram))


def test_warnings_preserve_the_exact_case_of_names():
    """DSL names are case-sensitive, so a warning must not re-case them.

    `str.capitalize()` lowercases everything after the first character, which
    silently turned "in 'Login'" into "in 'login'" — a name the user could not
    then find in their own diagram.
    """
    diagram = VisualDiagram(
        services=["Client"],
        flows=[
            VisualFlow(
                name="LoginFlow",
                interactions=[interaction(source="Client", target="UserStore")],
            )
        ],
    )

    warnings = validate(diagram)

    assert any("'LoginFlow'" in w for w in warnings)
    assert not any("'loginflow'" in w for w in warnings)


def test_a_flow_with_no_interactions_is_omitted_and_warned_about():
    diagram = VisualDiagram(services=["A"], flows=[VisualFlow(name="Empty")])

    assert "flow Empty" not in compile_dsl(diagram)
    assert any("no interactions" in w for w in validate(diagram))


def test_a_bad_parameter_name_is_dropped_but_the_call_survives():
    diagram = one_flow(
        interaction(parameters=[VisualParameter("ok", "int"), VisualParameter("not ok", "int")])
    )

    text = compile_dsl(diagram)

    assert "call(ok:int)" in text
    parse(text)
    assert any("not a usable name" in w for w in validate(diagram))


def test_warnings_are_raised_for_an_entirely_empty_model():
    warnings = validate(VisualDiagram())

    assert any("No services" in w for w in warnings)
    assert any("No flows" in w for w in warnings)


@pytest.mark.parametrize(
    "diagram",
    [
        VisualDiagram(),
        VisualDiagram(services=["A"]),
        one_flow(),
        one_flow(interaction(source="")),
        one_flow(interaction(condition="bad name", arm="alt")),
        VisualDiagram(services=["A", "A"], flows=[VisualFlow("F")]),
    ],
)
def test_every_degenerate_state_still_compiles_to_parseable_text(diagram):
    """Whatever the form state, the output must never crash the parser."""
    parse(compile_dsl(diagram))
