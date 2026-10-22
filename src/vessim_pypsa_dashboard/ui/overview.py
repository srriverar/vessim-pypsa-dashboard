"""Landing page: the five official Vessim examples, the roadmap case, and their views."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..vessim_runner import CASES, CASE_ORDER
from . import data, theme

VIEWS = [
    ("🔮", "Vessim only", "The official example running for real — actors, dispatchables, the policy, "
                        "the Recorder — with the policy laboratory and the PyPSA plan as sheets."),
    ("⚡", "LoadFlow (AdjieBrotot)", "The case's own system as a network: editable buses, lines, "
                              "transformers and meters + the single-line diagram, Newton-Raphson AC, "
                              "Y-bus, flows and losses, validated against pypsa.Network.pf()."),
    ("🏭", "ED-UC (AdjieBrotot)", "The case's own fleet: λ-iteration and a priority list on editable "
                            "unit sheets, compared against the PyPSA MILP."),
    ("🔗", "Vessim + PyPSA", "The Vessim scenario re-optimised on the case's own network — the ⚡ "
                        "buses — so congestion, nodal prices and curtailment are priced per bus."),
]

#: (label in the sidebar, the question it answers, what you do there)
ROUTES = [
    ("🏠 Overview", "what is in this app?", "the six cases and this route map"),
    ("🧭 Method", "how is it built, and how do the numbers connect?",
     "the pipeline, one MW traced through the four engines, the glossary"),
    ("01 … 06", "let me see one case end to end",
     "four tabs per case: Vessim only (run · policies · optimum · maths), LoadFlow (+ maths), "
     "ED-UC (+ maths), PyPSA (+ maths and the gap); 06 adds the roadmap tab"),
    ("🧩 When to use what", "Vessim or PyPSA — and when both?",
     "the decision guide, the fairness protocol checked on your run, the document"),
    ("🕶️ Blind spots & policies", "what does the LP miss, and how do I write my own rule?",
     "three live probes, the policy contract, and an editor that executes your class"),
    ("🔌 Hardware in the loop", "how would a real device join?",
     "the SiL monitor, Modbus/Prometheus recipes, the safety envelope"),
    ("📊 Studies · viewer", "what does a study look like?",
     "the Vessim viewer reproduced over live Optuna and dispatch studies"),
    ("📄 Papers · test systems", "what have people done with Vessim?",
     "five papers, their test systems recomputed here, paper numbers beside reproduced ones"),
]


def render() -> None:
    theme.hero(
        "00", "Vessim ↔ PyPSA: a microgrid laboratory",
        "TEACHING DASHBOARD AND REPRODUCIBLE RESEARCH",
        "Five official <b>Vessim</b> examples simulated for real and a sixth built on the "
        "tools ported to Python with their editable sheets and their single-line diagrams, and "
        "<b>Vessim roadmap</b>, with the same physics solved with <b>PyPSA</b>. The underlying "
        "question: how much is it worth "
        "planning with a network and a horizon instead of dispatching by rules?",
        [("Cases", str(len(CASE_ORDER))), ("Views per case", "4 + 1"), ("Pages", "6"),
         ("Solver", "HiGHS")],
    )

    st.markdown(
        '<div class="note">Every case opens <b>four tabs</b>: <b>Vessim only</b> — with the run, the '
        'policies and the optimum as three sheets — the load flow simulator, the dispatch/commitment '
        'simulator, and the PyPSA model fed by the signals Vessim measured. Pages that stand apart: '
        'The maths now sits inside each tab, in a <b>🧮</b> expander next to the number it '
        'explains; <b>🧩 When to use what</b> '
        'decides which tool answers which question (with the fairness protocol checked on your run), '
        '<b>🕶️ What PyPSA can\'t see</b> measures the blind spots of an LP, <b>🛠️ Change the '
        'policy</b> shows how to write and run your own rule in Vessim, <b>🔌 Hardware in the loop</b> '
        'puts a real device in the loop, and <b>📊 Studies</b> reproduces the Vessim viewer. Pick a '
        'case in the sidebar or with the buttons below.</div>',
        unsafe_allow_html=True)

    # ------------------------------------------------------------- case cards
    for row_start in range(0, len(CASE_ORDER), 2):
        columns = st.columns(2, gap="medium")
        for column, key in zip(columns, CASE_ORDER[row_start:row_start + 2]):
            spec = CASES[key]
            with column:
                st.markdown(
                    f"""
                    <div class="hero" style="padding:.95rem 1.1rem; margin-bottom:.2rem">
                      <div class="eyebrow">{spec.number} · {spec.source}</div>
                      <h1 style="font-size:1.22rem">{spec.title}</h1>
                      <p style="font-size:.85rem">{spec.description[:240]}…</p>
                      <div class="chips">
                        <span class="chip">steps <b>{spec.steps}</b></span>
                        <span class="chip">Δt <b>{spec.step_s/60:.0f} min</b></span>
                        <span class="chip">load <b>{spec.load_kw:,.0f} kW</b></span>
                        <span class="chip">solar <b>{spec.solar_kwp:,.0f} kWp</b></span>
                        <span class="chip">CO₂ <b>{spec.carbon_price:,.0f} USD/t</b></span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True)
                if st.button("Open case →", key=f"open_{key}", **theme.wide()):
                    st.session_state["pending_nav"] = key
                    st.rerun()

    # ----------------------------------------------------------------- views
    theme.section("The four tabs of every case", "same physics, several ways of looking at it — "
                  "case 06 adds a fifth, 🎲 Roadmap", "ARCHITECTURE")
    columns = st.columns(4)
    for column, (icon, title, text) in zip(columns, VIEWS):
        with column:
            st.markdown(
                f"""<div class="kpi" style="height:100%">
                      <div class="label">{icon} {title}</div>
                      <div class="sub" style="margin-top:.4rem; font-size:.8rem; line-height:1.35">{text}</div>
                    </div>""", unsafe_allow_html=True)

    theme.section("Where to go, and why", "every page answers one question", "ROUTE MAP")
    st.dataframe(pd.DataFrame(ROUTES, columns=["Page", "The question it answers",
                                               "What you do there"]),
                 hide_index=True, **theme.wide(), height=390)
    st.caption("The sidebar is this table. If a number in one page puzzles you, the row above it "
               "is usually where the explanation lives — and 🧭 Method traces a single megawatt "
               "through all four engines of a case so the correspondence is visible, not implied.")

    # ------------------------------------------------------------- comparison
    theme.section("Overview of every case", "Vessim in charge vs. PyPSA in charge", "COMPARISON")
    st.caption("All six examples run (including the software-in-the-loop one, which launches a real "
               "CPU process) and the same day is re-optimised with PyPSA. It takes a few seconds.")

    if st.button("▶ Run the full overview", type="primary", key="btn_overview"):
        st.session_state["overview_ready"] = True

    if not st.session_state.get("overview_ready"):
        st.markdown('<div class="small">Press the button to launch the twelve simulations.</div>',
                    unsafe_allow_html=True)
        return

    with st.spinner("Running every case and its OPF…"):
        frame = data.overview()

    st.dataframe(
        frame[["number", "title", "steps", "dt_min", "vessim_cost", "pypsa_cost",
               "vessim_emissions_kg", "pypsa_emissions_kg", "status"]]
        .rename(columns={"number": "#", "title": "Case", "steps": "Steps", "dt_min": "Δt (min)",
                         "vessim_cost": "Vessim (USD)", "pypsa_cost": "PyPSA (USD)",
                         "vessim_emissions_kg": "Vessim (kgCO₂)", "pypsa_emissions_kg": "PyPSA (kgCO₂)",
                         "status": "Status"})
        .style.format({"Vessim (USD)": "{:,.2f}", "PyPSA (USD)": "{:,.2f}",
                       "Vessim (kgCO₂)": "{:,.1f}", "PyPSA (kgCO₂)": "{:,.1f}",
                       "Δt (min)": "{:,.0f}"}, na_rep="—"),
        **theme.wide(), hide_index=True)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=frame["title"], y=frame["vessim_cost"], name="Vessim",
                             marker_color=theme.COLORS["vessim"]))
        fig.add_trace(go.Bar(x=frame["title"], y=frame["pypsa_cost"], name="PyPSA",
                             marker_color=theme.COLORS["pypsa"]))
        theme.figure(fig, height=330, y_title="USD", legend=True, unified=False)
        fig.update_layout(barmode="group")
        theme.show(fig)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=frame["title"], y=frame["vessim_emissions_kg"], name="Vessim",
                             marker_color=theme.COLORS["vessim"]))
        fig.add_trace(go.Bar(x=frame["title"], y=frame["pypsa_emissions_kg"], name="PyPSA",
                             marker_color=theme.COLORS["pypsa"]))
        theme.figure(fig, height=330, y_title="kg CO₂", legend=True, unified=False)
        fig.update_layout(barmode="group")
        theme.show(fig)

    total_v = float(frame["vessim_cost"].sum())
    total_p = float(frame["pypsa_cost"].sum())
    theme.kpis([
        ("Aggregate Vessim cost", data.fmt_usd(total_v), "", f"{len(CASE_ORDER)} cases, one day each", ""),
        ("Aggregate PyPSA cost", data.fmt_usd(total_p), "", "same data, optimal decision", "good"),
        ("Difference", data.fmt_usd(total_v - total_p), "",
         f"{100*(total_v-total_p)/max(total_v,1e-9):+.1f}%", "good" if total_v > total_p else "bad"),
        ("Vessim runtime", f"{frame['vessim_runtime_s'].sum():.2f}", "s", "real simulation", ""),
        ("PyPSA runtime", f"{frame['pypsa_runtime_s'].sum():.2f}", "s", "one per case", ""),
    ])
    theme.note("The first two cases are at watt scale: their costs are cents and percentage "
               "differences are amplified. Cases 03–05 are the ones where the gap means something "
               "economically.", "")
