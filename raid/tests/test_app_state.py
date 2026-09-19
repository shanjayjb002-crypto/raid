"""UI-level tests for the dashboard's input state, via Streamlit's AppTest.

These drive the real app script in-process - no browser - so the guarantees
that matter for a live demo are enforced by the suite rather than checked by
eye: that both inputs start genuinely empty, that the placeholder is never
mistaken for content, and that Load/Clear act on the active input only.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from raid.dashboard.builder import LOGIN_DSL

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def fresh() -> AppTest:
    """A freshly loaded app, as a user sees it on first visit or refresh."""
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    return app


def code_box(app: AppTest):
    return app.text_area(key="code_text")


def generated_dsl(app: AppTest) -> str:
    """The 'Generated DSL' preview block from the visual tab."""
    return app.code[0].value if len(app.code) else ""


def analysed(app: AppTest) -> bool:
    """Whether the pipeline ran, i.e. the report sections were rendered."""
    return any("Diagram sufficiency" in md.value for md in app.subheader)


# ==========================================================================
# A fresh load must be empty on both sides
# ==========================================================================


def test_the_app_loads_without_error():
    app = fresh()

    assert not app.exception


def test_code_editor_starts_empty():
    """The example must not be sitting in the box as deletable content."""
    app = fresh()

    assert code_box(app).value == ""


def test_code_editor_shows_only_a_placeholder_hint():
    """The hint is a placeholder attribute, not a value."""
    app = fresh()

    box = code_box(app)
    assert box.placeholder
    assert "service OrderService" in box.placeholder
    assert box.value == ""


def test_visual_builder_starts_with_no_services_or_flows():
    app = fresh()

    diagram = app.session_state["visual"]
    assert diagram.services == []
    assert diagram.flows == []


def test_generated_dsl_preview_starts_empty():
    app = fresh()

    assert generated_dsl(app) == ""


def test_nothing_is_analysed_on_a_fresh_load():
    """Placeholder text must never be treated as real input."""
    app = fresh()

    assert not analysed(app)
    assert any("Nothing to analyse yet" in info.value for info in app.info)


def test_both_sides_are_empty_after_switching_tabs():
    """Switching modes must not carry content from one into the other."""
    app = fresh()

    app.button(key="swap_mode").click().run()

    assert app.session_state["input_mode"] == "code"
    assert code_box(app).value == ""
    assert app.session_state["visual"].services == []
    assert not analysed(app)


# ==========================================================================
# Load example acts on the active input only
# ==========================================================================


def test_load_example_fills_the_visual_builder_when_it_is_active():
    app = fresh()
    assert app.session_state["input_mode"] == "visual"

    app.button(key="load_example").click().run()

    diagram = app.session_state["visual"]
    assert diagram.services == ["Client", "AuthService", "UserStore"]
    assert [f.name for f in diagram.flows] == ["Login"]
    assert generated_dsl(app) == LOGIN_DSL.rstrip("\n")


def test_loading_into_the_visual_builder_leaves_the_code_editor_empty():
    app = fresh()

    app.button(key="load_example").click().run()

    assert code_box(app).value == ""


def test_load_example_fills_the_code_editor_when_it_is_active():
    app = fresh()
    app.button(key="swap_mode").click().run()

    app.button(key="load_example").click().run()

    assert code_box(app).value == LOGIN_DSL


def test_loading_into_the_code_editor_leaves_the_visual_builder_empty():
    app = fresh()
    app.button(key="swap_mode").click().run()

    app.button(key="load_example").click().run()

    assert app.session_state["visual"].services == []


def test_the_loaded_example_actually_runs_the_pipeline():
    app = fresh()

    app.button(key="load_example").click().run()

    assert not app.exception
    assert analysed(app)


# ==========================================================================
# Clear all acts on the active input only
# ==========================================================================


def test_clear_all_empties_the_visual_builder():
    app = fresh()
    app.button(key="load_example").click().run()
    assert app.session_state["visual"].services

    app.button(key="clear_all").click().run()

    assert app.session_state["visual"].services == []
    assert app.session_state["visual"].flows == []
    assert generated_dsl(app) == ""
    assert not analysed(app)


def test_clear_all_empties_the_code_editor():
    app = fresh()
    app.button(key="swap_mode").click().run()
    app.button(key="load_example").click().run()
    assert code_box(app).value == LOGIN_DSL

    app.button(key="clear_all").click().run()

    assert code_box(app).value == ""
    assert not analysed(app)


def test_clearing_one_mode_leaves_the_other_untouched():
    """Load into both, clear only the active one."""
    app = fresh()
    app.button(key="load_example").click().run()          # visual
    app.button(key="swap_mode").click().run()             # now code
    app.button(key="load_example").click().run()          # code
    assert app.session_state["visual"].services
    assert code_box(app).value == LOGIN_DSL

    app.button(key="clear_all").click().run()             # clears code only

    assert code_box(app).value == ""
    assert app.session_state["visual"].services == ["Client", "AuthService", "UserStore"]


# ==========================================================================
# Typing in the code editor
# ==========================================================================


def test_typing_in_the_code_editor_makes_it_the_active_input():
    app = fresh()

    code_box(app).set_value("service A\n\nflow F:\n  A -> A : ping()\n").run()

    assert app.session_state["input_mode"] == "code"
    assert not app.exception
    assert analysed(app)


def test_a_syntax_error_typed_into_the_editor_is_reported_not_raised():
    app = fresh()

    code_box(app).set_value("flow Broken:\n  A B : c()\n").run()

    assert not app.exception
    assert any("could not be parsed" in error.value for error in app.error)


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_whitespace_only_input_is_treated_as_empty(text):
    app = fresh()

    code_box(app).set_value(text).run()

    assert not analysed(app)
    assert not app.exception
