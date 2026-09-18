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
from raid.dashboard.pipeline import Analysis, AnalysisOptions, analyse, rows_to_csv  # noqa: E402

EXAMPLE = """service PaymentService
service OrderService
service InventoryService

flow PlaceOrder:
  OrderService -> InventoryService : checkStock(itemId:int, qty:int)
  alt stock_available:
    OrderService -> PaymentService : charge(amount:float, cardId:string) [timeout=2s, retry=3]
    alt payment_success:
      OrderService -> InventoryService : reserve(itemId:int, qty:int)
  else out_of_stock:
    OrderService -> OrderService : rejectOrder()
"""

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

    source = st.text_area(
        "Diagram (RAID DSL)",
        value=EXAMPLE,
        height=280,
        help="Edit freely — the example is only a starting point.",
    )

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
