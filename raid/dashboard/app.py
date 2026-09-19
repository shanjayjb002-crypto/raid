"""Streamlit UI for RAID.

Rendering only - every analysis decision lives in
:mod:`raid.dashboard.pipeline`, which is unit tested separately.

Two Streamlit details worth knowing when reading this:

* Service filter buttons use ``on_click`` callbacks. A callback fires *before*
  the script re-runs, so the tables at the top of the page already see the new
  filter. Reading ``st.button``'s return value instead would apply the filter
  one interaction late, because the tables are rendered above the buttons.
* The analysis is cached on its inputs. Streamlit re-runs the whole script on
  every interaction, and without caching, clicking a service name would re-run
  the simulation and the fuzzer just to filter a table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run raid/dashboard/app.py` from a checkout without an install.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from raid.dsl import DslSyntaxError  # noqa: E402
from raid.dashboard.builder import (  # noqa: E402
    LOGIN_DSL,
    PARAMETER_TYPES,
    VisualDiagram,
    VisualFlow,
    VisualInteraction,
    VisualParameter,
    compile_dsl,
    login_example,
    validate,
)
from raid.dashboard.pipeline import Analysis, AnalysisOptions, analyse, rows_to_csv  # noqa: E402

#: Both input modes open showing this same diagram, and the visual builder
#: compiles to it byte for byte (asserted in raid/tests/test_builder.py).
EXAMPLE = LOGIN_DSL

_SCORE_ICON = {"high": "🟢", "medium": "🟡", "low": "🔴"}
_SEVERITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🔵"}


@st.cache_data(show_spinner=False)
def _run(source: str, runs: int, mutations: int, seed: int, flow: str | None, failure: str | None):
    return analyse(
        source,
        AnalysisOptions(
            runs=runs,
            mutations=mutations,
            seed=seed,
            flow_name=flow,
            failure_service=failure,
        ),
    )


def _set_focus(service: str | None) -> None:
    st.session_state["focus"] = service


# --------------------------------------------------------------------------
# Visual builder
# --------------------------------------------------------------------------
#
# Every mutation below runs in an `on_click` callback and marks the visual
# model as the active input. `st.tabs` renders both tabs on every run, so the
# app cannot tell which one the user is looking at; tracking the last mode
# *acted on* is what decides which text feeds the analysis.


def _visual() -> VisualDiagram:
    return st.session_state["visual"]


def _use_visual() -> None:
    st.session_state["input_mode"] = "visual"


def _use_code() -> None:
    st.session_state["input_mode"] = "code"


def _add_service(name: str) -> None:
    _use_visual()
    name = name.strip()
    if name and name not in _visual().services:
        _visual().services.append(name)


def _delete_service(name: str) -> None:
    _use_visual()
    if name in _visual().services:
        _visual().services.remove(name)


def _add_flow(name: str) -> None:
    _use_visual()
    name = name.strip()
    if name and _visual().flow(name) is None:
        _visual().flows.append(VisualFlow(name=name))
        st.session_state["vb_flow"] = name


def _delete_flow(name: str) -> None:
    _use_visual()
    diagram = _visual()
    diagram.flows = [f for f in diagram.flows if f.name != name]


def _add_interaction(flow_name: str, interaction: VisualInteraction) -> None:
    _use_visual()
    flow = _visual().flow(flow_name)
    if flow is not None:
        flow.interactions.append(interaction)


def _delete_interaction(flow_name: str, index: int) -> None:
    _use_visual()
    flow = _visual().flow(flow_name)
    if flow is not None and 0 <= index < len(flow.interactions):
        flow.interactions.pop(index)


def _copy_to_code(text: str) -> None:
    st.session_state["code_text"] = text
    st.session_state["input_mode"] = "code"


def _render_service_builder() -> None:
    st.markdown("##### Services")
    with st.form("vb_add_service", clear_on_submit=True):
        columns = st.columns([3, 1])
        name = columns[0].text_input(
            "Service name", placeholder="PaymentService", label_visibility="collapsed"
        )
        added = columns[1].form_submit_button("Add service", width="stretch")
    if added:
        _add_service(name)
        st.rerun()

    services = _visual().services
    if not services:
        st.caption("No services yet.")
        return

    for service in list(services):
        row = st.columns([5, 1])
        row[0].markdown(f"`{service}`")
        row[1].button(
            "Delete",
            key=f"vb_del_svc_{service}",
            on_click=_delete_service,
            args=(service,),
            width="stretch",
        )


def _render_flow_builder() -> None:
    st.markdown("##### Flows")
    with st.form("vb_add_flow", clear_on_submit=True):
        columns = st.columns([3, 1])
        name = columns[0].text_input(
            "Flow name", placeholder="PlaceOrder", label_visibility="collapsed"
        )
        added = columns[1].form_submit_button("Add flow", width="stretch")
    if added:
        _add_flow(name)
        st.rerun()

    flows = _visual().flows
    if not flows:
        st.caption("No flows yet.")
        return

    names = [f.name for f in flows]
    selected = st.selectbox("Flow to edit", names, key="vb_flow")
    st.button(
        f"Delete flow '{selected}'",
        key="vb_del_flow",
        on_click=_delete_flow,
        args=(selected,),
    )
    _render_interaction_builder(selected)


def _render_interaction_builder(flow_name: str) -> None:
    flow = _visual().flow(flow_name)
    if flow is None:
        return

    st.markdown(f"##### Interactions in `{flow_name}`")

    for index, existing in enumerate(flow.interactions):
        row = st.columns([5, 1])
        arm = ""
        if existing.condition:
            word = "when" if existing.arm == "alt" else "otherwise, when"
            arm = f"  ·  _{word} **{existing.condition}**_"
        parameters = ", ".join(
            f"{p.name}:{p.type}" if p.type else p.name for p in existing.parameters
        )
        annotations = []
        if existing.timeout_ms is not None:
            annotations.append(f"timeout {existing.timeout_ms}ms")
        if existing.retry is not None:
            annotations.append(f"retry {existing.retry}")
        suffix = f"  ·  _{', '.join(annotations)}_" if annotations else ""
        row[0].markdown(
            f"{index + 1}. `{existing.source} → {existing.target} : "
            f"{existing.method}({parameters})`{suffix}{arm}"
        )
        row[1].button(
            "Delete",
            key=f"vb_del_int_{flow_name}_{index}",
            on_click=_delete_interaction,
            args=(flow_name, index),
            width="stretch",
        )

    services = _visual().services
    if len(services) < 1:
        st.info("Add at least one service before adding an interaction.")
        return

    st.markdown("**Add an interaction**")
    endpoints = st.columns(3)
    source = endpoints[0].selectbox("From", services, key="vb_src")
    target = endpoints[1].selectbox("To", services, key="vb_tgt")
    method = endpoints[2].text_input("Method name", key="vb_method", placeholder="charge")

    # The parameter-row count lives outside any form: a widget inside a
    # st.form does not trigger a rerun until submit, so rows added there
    # would not appear until after the interaction was saved.
    count = st.number_input(
        "How many parameters?", min_value=0, max_value=10, value=0, step=1, key="vb_nparams"
    )
    parameters: list[VisualParameter] = []
    for index in range(int(count)):
        columns = st.columns([3, 2])
        parameter_name = columns[0].text_input(
            f"Parameter {index + 1} name", key=f"vb_pname_{index}"
        )
        parameter_type = columns[1].selectbox(
            f"Parameter {index + 1} type",
            PARAMETER_TYPES,
            format_func=lambda t: t if t else "(untyped)",
            key=f"vb_ptype_{index}",
        )
        parameters.append(VisualParameter(parameter_name.strip(), parameter_type))

    annotations = st.columns(2)
    use_timeout = annotations[0].checkbox("Set a timeout", key="vb_use_timeout")
    timeout_ms = (
        annotations[0].number_input(
            "Timeout (ms)", min_value=1, max_value=600_000, value=1000, step=50, key="vb_timeout"
        )
        if use_timeout
        else None
    )
    use_retry = annotations[1].checkbox("Set a retry count", key="vb_use_retry")
    retry = (
        annotations[1].number_input(
            "Retries", min_value=0, max_value=20, value=3, step=1, key="vb_retry"
        )
        if use_retry
        else None
    )

    branched = st.checkbox("This interaction is inside a branch", key="vb_branched")
    condition = None
    arm = "alt"
    if branched:
        branch_columns = st.columns([2, 3])
        condition = branch_columns[0].text_input(
            "Condition name", key="vb_condition", placeholder="payment_success"
        ).strip()
        arm_label = branch_columns[1].radio(
            "Which side of the branch?",
            ["Main branch — runs when the condition holds", "Alternative branch — runs otherwise"],
            key="vb_arm",
        )
        arm = "alt" if arm_label.startswith("Main") else "else"
        st.caption(
            "Interactions tagged with the same condition, one after another, are "
            "grouped into a single branch for you."
        )

    if st.button("Add interaction", type="primary", key="vb_add_int"):
        _add_interaction(
            flow_name,
            VisualInteraction(
                source=source,
                target=target,
                method=method.strip(),
                parameters=parameters,
                timeout_ms=int(timeout_ms) if timeout_ms is not None else None,
                retry=int(retry) if retry is not None else None,
                condition=condition or None,
                arm=arm,
            ),
        )
        st.rerun()


def _render_visual_tab() -> str:
    """Render the builder and return the DSL it currently compiles to."""
    st.caption(
        "Build a diagram with forms. Everything you add is compiled to RAID's "
        "DSL below — that text, and nothing else, is what gets parsed."
    )

    left, right = st.columns(2)
    with left:
        _render_service_builder()
    with right:
        _render_flow_builder()

    text = compile_dsl(_visual())

    st.divider()
    st.markdown("##### Generated DSL (this is what RAID actually parses)")
    st.code(text or "# nothing to compile yet", language="text")

    for warning in validate(_visual()):
        st.warning(warning, icon="⚠️")

    st.button(
        "Copy into the code editor",
        key="vb_copy",
        on_click=_copy_to_code,
        args=(text,),
        help="Hand-tune the generated DSL — useful for nesting a branch inside a branch.",
    )
    return text


def _download(label: str, rows: list[dict], filename: str, key: str) -> None:
    st.download_button(
        label,
        data=rows_to_csv(rows),
        file_name=filename,
        mime="text/csv",
        disabled=not rows,
        key=key,
    )


def _render_sufficiency(analysis: Analysis) -> None:
    st.subheader("1 · Diagram sufficiency")
    st.caption(
        "Scored before anything is generated, so you can see which results "
        "rest on your diagram and which rest on RAID's defaults."
    )

    columns = st.columns(len(analysis.sufficiency.concerns) + 1)
    columns[0].metric("Overall", analysis.sufficiency.overall.upper())
    for column, concern in zip(columns[1:], analysis.sufficiency.concerns):
        column.metric(
            concern.concern.title(),
            f"{_SCORE_ICON[concern.score]} {concern.score}",
            help=concern.rationale,
        )

    suggestions = analysis.sufficiency.suggestions
    if not suggestions:
        st.success("Nothing blocking analysis — this diagram is fully specified.")
        return

    with st.expander(f"{len(suggestions)} suggestions to improve analysability", expanded=True):
        for suggestion in suggestions:
            st.markdown(
                f"{_SEVERITY_ICON[suggestion.severity]} **{suggestion.element}** — "
                f"{suggestion.message}"
            )


def _render_tests(analysis: Analysis, focus: str | None) -> None:
    selected = analysis.slice_for(focus)
    rows = analysis.test_case_rows(focus)

    st.subheader("2 · Generated test cases")
    st.caption(
        f"{len(selected.functional_tests)} functional · "
        f"{len(selected.boundary_tests)} boundary"
    )

    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No test cases for this selection.")
    _download("Download test cases (CSV)", rows, "raid_test_cases.csv", "dl_tests")


def _render_resilience(analysis: Analysis, focus: str | None) -> None:
    st.subheader("3 · Resilience simulation")
    report = analysis.report
    if report is None:
        st.info("No simulation was run.")
        return

    left, middle, right = st.columns(3)
    left.metric("p50 latency", f"{report.latency_p50:.0f} ms")
    middle.metric("p95 latency", f"{report.latency_p95:.0f} ms")
    right.metric("Runs completed", f"{report.completed_runs}/{report.runs}")

    st.markdown("**Critical path** — each service's share of end-to-end latency")
    st.dataframe(
        [
            {
                "service": c.service,
                "calls": c.calls,
                "mean latency (ms)": round(c.mean_latency, 1),
                "share": f"{c.share:.0%}",
            }
            for c in report.critical_path
        ],
        width="stretch",
        hide_index=True,
    )

    findings = analysis.slice_for(focus).resilience_findings
    if findings:
        for violation in findings:
            st.error(violation.describe())
    elif report.timed_out:
        st.info("No timeouts involve the selected service.")
    else:
        st.success("Every run completed inside its timeout budget.")


def _render_chaos(analysis: Analysis, focus: str | None) -> None:
    specs = analysis.slice_for(focus).chaos_specs

    st.subheader("4 · Auto-generated chaos test specs")
    st.caption(
        "Written only where a mutation made the system measurably worse than "
        "the healthy baseline."
    )

    if not specs:
        st.info("No chaos specs for this selection.")
        return

    for spec in specs:
        with st.expander(f"{spec.description}  ·  {', '.join(spec.regressions)}"):
            st.markdown(f"**Expected (baseline):** {spec.baseline_behaviour}")
            st.markdown(f"**Observed (mutated):** {spec.observed_behaviour}")
            st.caption(f"severity {spec.severity:.2f} · id `{spec.id}`")

    _download("Download chaos specs (CSV)", analysis.chaos_rows(focus), "raid_chaos_specs.csv", "dl_chaos")


def _graphviz(analysis: Analysis, focus: str | None) -> str:
    """Render the service graph as DOT, highlighting the focused service."""
    lines = [
        "digraph raid {",
        '  rankdir=LR;',
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fillcolor="#eef2ff"];',
        '  edge [fontname="Helvetica" fontsize=9 color="#64748b"];',
    ]
    for service in sorted(analysis.graph.graph.nodes):
        fill = "#fecaca" if service == focus else "#eef2ff"
        lines.append(f'  "{service}" [fillcolor="{fill}"];')
    for source, target, data in analysis.graph.graph.edges(data=True):
        lines.append(f'  "{source}" -> "{target}" [label="{data["method"]}"];')
    lines.append("}")
    return "\n".join(lines)


def _render_impact(analysis: Analysis, focus: str | None) -> None:
    st.subheader("5 · Impact graph")
    st.caption("Click a service to filter every table above to what it produced.")

    st.graphviz_chart(_graphviz(analysis, focus))

    services = analysis.focus_options()
    columns = st.columns(max(len(services) + 1, 2))
    columns[0].button("Show all", on_click=_set_focus, args=(None,), key="focus_all")
    for column, service in zip(columns[1:], services):
        column.button(
            service,
            on_click=_set_focus,
            args=(service,),
            key=f"focus_{service}",
            type="primary" if service == focus else "secondary",
        )

    st.markdown("**What each diagram element produced**")
    st.dataframe(analysis.impact.to_table(), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="RAID", page_icon="🛡️", layout="wide")
    st.title("🛡️ RAID")
    st.caption(
        "Resilience & Assertion Inference from Diagrams — "
        "everything below is derived locally from the diagram alone."
    )

    with st.sidebar:
        st.header("Settings")
        runs = st.slider("Simulation runs", 10, 500, 100, step=10)
        mutations = st.slider("Mutations to try", 1, 40, 15)
        seed = st.number_input("Random seed", value=0, step=1)
        st.caption("A fixed seed makes every result on this page reproducible.")

    st.session_state.setdefault("visual", login_example())
    st.session_state.setdefault("code_text", EXAMPLE)
    st.session_state.setdefault("input_mode", "visual")

    visual_tab, code_tab = st.tabs(["🧩 Build visually", "⌨️ Edit as code"])

    with visual_tab:
        visual_source = _render_visual_tab()

    with code_tab:
        st.caption("Type RAID DSL directly. Supports everything the builder does, plus nesting.")
        st.text_area(
            "Diagram (RAID DSL)",
            key="code_text",
            height=280,
            on_change=_use_code,
            help="Editing here makes the code the active input.",
        )
        st.button("Use this code", key="use_code", on_click=_use_code)

    mode = st.session_state["input_mode"]
    source = visual_source if mode == "visual" else st.session_state["code_text"]

    banner, switch = st.columns([5, 1])
    banner.caption(
        f"Analysing the diagram from **{'Build visually' if mode == 'visual' else 'Edit as code'}** "
        f"— whichever you touched last drives the results below."
    )
    switch.button(
        "Use the other",
        key="swap_mode",
        on_click=_use_code if mode == "visual" else _use_visual,
        width="stretch",
    )

    if not source.strip():
        st.info("Nothing to analyse yet — add a service and a flow to get started.")
        return

    try:
        preview = analyse(source, AnalysisOptions(runs=1, mutations=1))
    except DslSyntaxError as exc:
        st.error("That diagram could not be parsed.")
        st.code(str(exc), language="text")
        st.info(
            "Check that each interaction reads "
            "`ServiceA -> ServiceB : method(param:type)` and that the body of "
            "a `flow` or `alt` block is indented."
        )
        return
    except Exception as exc:  # noqa: BLE001 - a live demo must never show a traceback
        st.error(f"Could not analyse this diagram: {exc}")
        return

    with st.sidebar:
        flow = (
            st.selectbox("Flow", preview.flow_names) if len(preview.flow_names) > 1 else None
        )
        failure = st.selectbox(
            "Inject a failure on",
            ["(none — healthy baseline)", *preview.focus_options()],
        )
        failure_service = None if failure.startswith("(none") else failure

    try:
        analysis = _run(source, runs, mutations, int(seed), flow, failure_service)
    except Exception as exc:  # noqa: BLE001 - as above
        st.error(f"Could not analyse this diagram: {exc}")
        return

    for warning in analysis.warnings:
        st.warning(warning)

    focus = st.session_state.get("focus")
    if focus and focus not in analysis.focus_options():
        focus = None
        _set_focus(None)

    if focus:
        banner, clear = st.columns([6, 1])
        banner.info(f"Filtering every table by **{focus}**")
        clear.button("Clear", on_click=_set_focus, args=(None,), key="focus_clear")

    _render_sufficiency(analysis)
    st.divider()
    _render_tests(analysis, focus)
    st.divider()
    _render_resilience(analysis, focus)
    st.divider()
    _render_chaos(analysis, focus)
    st.divider()
    _render_impact(analysis, focus)


if __name__ == "__main__":
    main()
