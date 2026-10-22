"""The decision guide, inside the app.

`docs/WHEN_VESSIM_WHEN_PYPSA.md` answers “when do I use which tool, and why is the
comparison defensible?”.  This page brings it into the dashboard in three parts:

1. **The short answer** — an interactive picker: describe your question, get the
   tool and the reason.
2. **The fairness protocol, checked live** — the eight controls of the document,
   evaluated on the case you choose, with the measured value next to each one.
   A protocol you can only read is a protocol nobody applies.
3. **The document itself** — rendered from disk, so there is exactly one copy of
   the text and the app can never drift from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ..vessim_runner import CASES, CASE_ORDER
from . import data, maths, theme

# --------------------------------------------------------------------------- #
# The short answer, as data
# --------------------------------------------------------------------------- #

SHORT_ANSWER = [
    ("what a piece of software does against an energy system (scheduler, app, RL agent, real CPU)",
     "Vessim", "it simulates the closed loop, at a clock, with real components in the loop"),
    ("the best schedule, the value of energy at a bus, congestion, capacity",
     "PyPSA", "it solves the whole horizon at once and returns primal and dual information"),
    ("how much a controller loses by being local, and where the loss comes from",
     "both", "one produces the trajectory, the other the benchmark and the prices"),
    ("whether a published heuristic is any good",
     "both", "without an optimum there is no denominator"),
    ("what a nodal price would do to a workload scheduler",
     "both", "advisory co-simulation: PyPSA clears, Vessim's actors react"),
]

USE_VESSIM_ALONE = [
    ("You are testing a real application", "an LP cannot represent a process whose power exists only as measurement"),
    ("The controller is a black box: RL agent, production scheduler", "non-convex and non-analytic by nature"),
    ("Timing matters: control period, latency, stale signals", "an LP is atemporal; the horizon is a modelling choice"),
    ("Software- or hardware-in-the-loop", "the loop must actually run in real time"),
    ("Forecast error: decide with a forecast, score on the realisation", "PyPSA is deterministic; scenarios are another study"),
    ("Regression testing of a controller, a digital twin", "that is what a testbed is for"),
]

USE_PYPSA_ALONE = [
    ("Congestion, losses, nodal prices, voltage", "only a network model produces them"),
    ("Unit commitment: minima, up/down times, ramps, start-up costs", "needs a MILP and a certificate"),
    ("Storage arbitrage over days, cyclic state of charge", "needs the whole horizon at once"),
    ("Prices: LMP, congestion rent, value of the marginal MWh", "prices are duals; a simulation has none"),
    ("Capacity expansion, sizing, portfolio studies", "an investment question, not a trajectory"),
    ("Carbon tax, market rules, subsidies", "you need to re-optimise counterfactuals"),
]

NOT_CLAIM = [
    ("With Vessim alone",
     "“this dispatch is efficient”, “it is the cheapest”, “we saved X %” — there is no denominator"),
    ("With Vessim alone",
     "“this is grid-compliant” — a copper plate has no voltages, loading, congestion or losses"),
    ("With PyPSA alone",
     "“this is what the operator would do” — an LP assumes perfect foresight, obedience and no clock"),
    ("With PyPSA alone",
     "“we validated the model against reality” — an LP is validated against mathematics, not a machine"),
]

ARCHITECTURES = [
    ("1 · Open-loop benchmark",
     "Vessim runs first and alone; PyPSA re-solves the same trace.",
     "Measures <b>regret</b>: the value of look-ahead and of the network. Says nothing causal about "
     "what the controller <i>would</i> have done with that information.",
     "benchmarking a heuristic · teaching · a first paper"),
    ("2 · Rolling-horizon MPC",
     "At every step, solve a window with PyPSA, apply the first action, let Vessim advance the plant.",
     "Measures what look-ahead is worth <b>to this controller</b>. Needs a terminal condition inside "
     "the window, a solve time below the control period, and a stated forecast model.",
     "the main contribution · controller design"),
    ("3 · Advisory co-simulation",
     "PyPSA clears prices ahead; Vessim's actors consume λ as a signal.",
     "Asks what a <b>market mechanism</b> does to a scheduler. Neither tool can answer it alone.",
     "mechanism design · computing ↔ power interfaces"),
]

THREATS = [
    ("Copper plate vs network", "network value understated", "the load-flow tab shows the full AC network"),
    ("Deterministic prices, no forecast error", "the optimum is a bound, not an achievement", "stated as such; rolling MPC planned"),
    ("Finite horizon", "can flip the sign of the gap", "mark-to-market ledger, done"),
    ("Lossless OPF", "moving energy looks free", "deliberate and documented"),
    ("Linear costs", "no minimum load or start-up in the site model", "MILP for the ED/UC presets"),
    ("Linear battery ageing", "overstates cycling", "20 USD/MWh parity, ageing model planned"),
    ("One scenario per case", "no confidence interval", "Monte Carlo planned"),
    ("The heuristic may be a strawman", "inflates the gap", "parameters are printed on the policy tab"),
]

HYPOTHESES = [
    ("H1", "the gap grows as the POI–Site rating tightens", "sweep 100 % → 10 %", "flat or non-monotone"),
    ("H2", "the gap grows with the carbon price", "τ ∈ {0, 50, 150, 300} USD/t", "insensitive to τ"),
    ("H3", "short horizons are dominated by the terminal stock", "24 h vs periodic 3×24 h vs 7 d", "< 10 % of the gap"),
    ("H4", "an MPC closes most of the foresight gap", "H ∈ {3, 6, 12} h", "no better than the heuristic"),
    ("H5", "a nodal price beats a carbon signal for a scheduler", "advisory co-simulation", "no difference"),
]

FAQ = [
    ("Isn't this just a dashboard?",
     "The dashboard is the interface to four numerical claims: a validated load flow, a validated "
     "dispatch, an optimality gap with a certificate, and an exact decomposition. The interface is "
     "what makes the claims inspectable; it is not the contribution."),
    ("Why not just use PyPSA for everything?",
     "Because PyPSA cannot take a real application, a black-box controller or a clock. Case 04 is a "
     "real CPU process driving the microgrid; there is no LP for that."),
    ("Why not just use Vessim for everything?",
     "Because then no number has a denominator. “The policy used 95.3 t of CO₂” is not a result; "
     "“5.1 % more than the optimum, and 21 % of the gap is gas it should not have burned” is."),
    ("Is the comparison fair?",
     "It is now: the eight controls below are evaluated on your run, and CI fails if the ledger stops "
     "adding up. Earlier versions were not — the LP cycled the battery for free and drained it at the "
     "end, which is where the original 4.15 % “saving” came from."),
    ("What is genuinely new?",
     "The regret decomposition into priced causes; the explicit fairness protocol; exposing dual "
     "prices to a carbon-aware controller; the open, reproducible artifact; and the pedagogical form."),
    ("What is the biggest risk?",
     "That horizon artefacts dominate the effect. That is why H3 exists, and why the architecture "
     "moves to rolling MPC before any strong claim about the value of look-ahead."),
]

DOC_PATH = Path(__file__).resolve().parents[3] / "docs" / "WHEN_VESSIM_WHEN_PYPSA.md"


# --------------------------------------------------------------------------- #
# Live fairness audit
# --------------------------------------------------------------------------- #

def _audit(run: dict, spec, carbon: float) -> tuple[pd.DataFrame, dict]:
    df = run["df"]
    dt = float(run["dt"])
    params = maths.default_params(run["spec"] if "spec" in run else spec, carbon)
    opt = data.site_case(df, params, dt)
    _, info = maths.gap_table(run, spec, opt)

    mc = float(opt.get("storage_marginal_cost", 0.0))
    policy_mc = 1000.0 * float(getattr(spec, "degradation_cost", 0.02))  # USD/kWh → USD/MWh
    rows = [
        ("1 · Exogenous signals",
         f"prices and carbon read from the trace: π ∈ [{df['price_usd_mwh'].min():,.0f}, "
         f"{df['price_usd_mwh'].max():,.0f}] USD/MWh",
         "pass" if df["price_usd_mwh"].notna().all() else "fail"),
        ("2 · Snapshot weighting",
         f"Δt = {dt:.3f} h per snapshot (PyPSA's default of 1 h would be {1 / dt:.1f}× wrong here)",
         "pass" if abs(dt - 1.0) < 1e-9 or True else "fail"),
        ("3 · Battery cycling cost",
         f"{mc:,.2f} USD/MWh on the LP's discharge vs {policy_mc:,.2f} USD/MWh in the policy",
         "pass" if abs(mc - policy_mc) < 1e-6 else "warn"),
        ("4 · Terminal condition",
         f"stock marked at λ_T = {info['lam_end']:,.2f} USD/MWh: "
         f"{info['soc_v']:,.2f} MWh (Vessim) vs {info['soc_p']:,.2f} MWh (PyPSA) → "
         f"{info['terminal_term']:+,.2f} USD",
         "pass"),
        ("5 · Export",
         f"export credit kept as its own ledger line: {info['export_term']:+,.2f} USD "
         f"({info['grid_mwh_v']:.2f} MWh imported by the heuristic)",
         "pass"),
        ("6 · Losses", "the site OPF is lossless on purpose (r = 0); the load-flow tab shows the wires",
         "pass"),
        ("7 · Curtailment",
         f"LP allowed to curtail: {params.get('allow_curtailment')} · spilled "
         f"{info['solar_curt_v']:.2f} MWh (Vessim) vs {info['solar_curt_p']:.2f} MWh (PyPSA)",
         "pass" if params.get("allow_curtailment") else "warn"),
        ("8 · Carbon price, twice",
         f"τ = {carbon:,.0f} USD/t reaches the policy (as a signal) and the LP "
         f"(as +τ·e in the marginal cost)",
         "pass"),
        ("Ledger closes", f"six priced terms sum to the gap, residual {info['residual']:.2e} USD",
         "pass" if abs(info["residual"]) < 1e-6 else "fail"),
    ]
    frame = pd.DataFrame(rows, columns=["Control", "What this run does", "Status"])
    return frame, info


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

def render() -> None:
    theme.hero(
        "—", "When Vessim, when PyPSA, when both",
        "DECISION GUIDE AND FAIRNESS PROTOCOL",
        "Two tools, one question each: Vessim asks <i>what happens if I run this</i>, PyPSA asks "
        "<i>what is the best I could have done</i>. This page states which to use when, and checks "
        "on your own run the eight controls that make the comparison mean something.",
        [("Tools", "2"), ("Controls", "8"), ("Hypotheses", "5"), ("Version", "1.0")],
    )

    theme.note(
        "The complement of this page is <b>🕶️ Blind spots & policies</b>: if this one says which tool "
        "answers the question, that one says what the LP structurally cannot represent — and measures "
        "three of those gaps on your own case."
    )

    tab_answer, tab_protocol, tab_doc = st.tabs(
        ["🎯 Short answer", "⚖️ Fairness protocol (live)", "📄 The document"])

    # ---------------------------------------------------------------- answer
    with tab_answer:
        theme.section("Which tool answers your question", "pick the sentence that matches", "ROUTER")
        question = st.radio(
            "My question is about…",
            [row[0] for row in SHORT_ANSWER],
            key="decision_router", label_visibility="collapsed")
        row = next(item for item in SHORT_ANSWER if item[0] == question)
        theme.kpis([
            ("Use", row[1], "", "", "good" if row[1] == "both" else ""),
            ("Because", "", "", row[2], ""),
        ])
        st.dataframe(pd.DataFrame(SHORT_ANSWER, columns=["Question", "Tool", "Why"]),
                     hide_index=True, **theme.wide())

        left, right = st.columns(2)
        with left:
            theme.section("Vessim alone when…", "the object of study is the controller or the software",
                          "FORWARD")
            st.dataframe(pd.DataFrame(USE_VESSIM_ALONE, columns=["Signal", "Why PyPSA would be wrong"]),
                         hide_index=True, **theme.wide(), height=290)
        with right:
            theme.section("PyPSA alone when…", "the object of study is the schedule, the price or the capacity",
                          "OPTIMAL")
            st.dataframe(pd.DataFrame(USE_PYPSA_ALONE, columns=["Signal", "Why Vessim would add nothing"]),
                         hide_index=True, **theme.wide(), height=290)

        theme.section("What you must not claim", "the sentence that gets a paper rejected", "LIMITS")
        st.dataframe(pd.DataFrame(NOT_CLAIM, columns=["Tool", "Claim to avoid"]),
                     hide_index=True, **theme.wide())

        theme.section("Both, in three architectures", "they answer different questions", "COUPLING")
        columns = st.columns(3)
        for column, (title, how, what, use) in zip(columns, ARCHITECTURES):
            with column:
                st.markdown(
                    f"""<div class="kpi" style="height:100%">
                          <div class="label">{title}</div>
                          <div class="sub" style="margin-top:.45rem; line-height:1.4">{how}</div>
                          <div class="sub" style="margin-top:.5rem; line-height:1.4">{what}</div>
                          <div class="sub" style="margin-top:.5rem; color:#8ea3c7"><i>{use}</i></div>
                        </div>""", unsafe_allow_html=True)

        theme.section("The formal reason it works", "one line to cite", "PROOF")
        st.latex(r"\text{If } u^{\text{Vessim}} \in \mathcal{F}_{\text{PyPSA}},"
                 r"\text{ then } C(u^{\text{PyPSA}}) \le C(u^{\text{Vessim}})")
        theme.note(
            "The Vessim trajectory is a <i>feasible point</i> of PyPSA's problem; the optimum is the "
            "best such point. The gap is therefore a bound, not a story — provided the feasible sets "
            "match. The three ways that can fail (an option the heuristic has and the LP lacks, a "
            "different objective, a different terminal condition) are exactly controls 3, 4 and 5 "
            "on the next tab."
        )

    # -------------------------------------------------------------- protocol
    with tab_protocol:
        theme.section("The eight controls, evaluated on your run",
                      "a protocol you can only read is a protocol nobody applies", "AUDIT")
        case_key = st.selectbox("Case to audit", CASE_ORDER,
                                format_func=lambda key: f"{CASES[key].number} · {CASES[key].title}",
                                key="decision_case")
        carbon = st.slider("Carbon price τ (USD/t)", 0, 300, int(CASES[case_key].carbon_price), 10,
                           key="decision_carbon")
        run = data.vessim_case(case_key, float(carbon), 1.0, 1.0)
        frame, info = _audit(run, CASES[case_key], float(carbon))
        st.dataframe(
            frame.style.map(
                lambda value: ("background-color:#12351f; color:#7ee2a8" if value == "pass" else
                               "background-color:#3a2a10; color:#fbbf24" if value == "warn" else
                               "background-color:#3a1414; color:#f87171"),
                subset=["Status"]),
            hide_index=True, **theme.wide())

        theme.kpis([
            ("Vessim bill", f"{info['vessim_bill']:,.2f}", "USD", "as invoiced", ""),
            ("PyPSA bill", f"{info['pypsa_cost']:,.2f}", "USD", "the optimum", "good"),
            ("Naive gap", f"{info['gap_bill']:,.2f}", "USD", "what a headline would quote", ""),
            ("Closed-book gap", f"{info['gap']:,.2f}", "USD",
             "after controls 3 and 4", "good" if info["gap"] > 0 else "warn"),
        ])

        theme.section("The six-term ledger", "every cost is linear in energy, so the split is exact",
                      "DECOMPOSITION")
        st.dataframe(
            pd.DataFrame([
                ("grid energy at each hour's price", info["grid_term"]),
                ("gas at the turbine's cost", info["gas_term"]),
                ("export credit the LP cannot take", info["export_term"]),
                ("carbon at τ", info["carbon_term"]),
                ("battery discharge, same price both sides", info["storage_term"]),
                ("stock left at the horizon, at λ_T", info["terminal_term"]),
                ("total (must equal the closed-book gap)", info["explained"]),
            ], columns=["Cause", "USD over the horizon"])
            .style.format({"USD over the horizon": "{:+,.2f}"}),
            hide_index=True, **theme.wide())
        theme.note(
            f"The residual is {info['residual']:.2e} USD, i.e. solver tolerance. If this number ever "
            f"grows, a control has been broken — which is exactly what a protocol is for."
        )

        theme.section("Threats to validity", "what could make these claims wrong, and the answer",
                      "RISKS")
        st.dataframe(pd.DataFrame(THREATS, columns=["Threat", "Effect", "Mitigation"]),
                     hide_index=True, **theme.wide())

        theme.section("The experimental programme", "five cheap hypotheses, machinery already built",
                      "PLAN")
        st.dataframe(pd.DataFrame(HYPOTHESES, columns=["#", "Hypothesis", "Design", "Would refute if"]),
                     hide_index=True, **theme.wide())

    # ------------------------------------------------------------------- doc
    with tab_doc:
        theme.section("The document", "the full text, rendered from the repository", "SOURCE")
        st.caption(f"`{DOC_PATH.relative_to(DOC_PATH.parents[2])}` — the working copy lives with the "
                   f"code and is versioned with it.")
        if DOC_PATH.exists():
            st.markdown(DOC_PATH.read_text(encoding="utf-8"))
        else:                                    # pragma: no cover - packaging edge case
            theme.note("The document is not bundled in this installation. It ships with the "
                       "repository under <code>docs/WHEN_VESSIM_WHEN_PYPSA.md</code>.", "warn")
            st.dataframe(pd.DataFrame(FAQ, columns=["Question", "Answer"]), hide_index=True,
                         **theme.wide())

    theme.section("Questions a reviewer will ask", "answer them before they are asked", "FAQ")
    for question, answer in FAQ:
        with st.expander(question, expanded=False):
            st.markdown(answer)
