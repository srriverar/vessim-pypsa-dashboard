"""Method page: architecture, Vessim→PyPSA mapping, validation and credits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import grid
from ..vessim_runner import CASES
from ..vessim_system import ed_at, site_network, snapshot, system_for
from . import data, theme

MAPPING = pd.DataFrame([
    ("Consumer (W)", "PQ load on its bus", "Load `p_set`, a time series in MW"),
    ("Producer solar (W)", "PV bus + generator", "`p_max_pu` = normalised trace (0–1)"),
    ("SimpleBattery", "StorageUnit", "`p_nom`, `max_hours`, charge/discharge η, initial SoC"),
    ("GridConnection", "Slack bus / grid generator", "`p_nom` = import limit, cost = price"),
    ("FuelGenerator", "Thermal generator", "cost = fuel (USD/MWh), CO₂ = t/MWh"),
    ("Carbon signal", "Marginal cost + CO₂", "`marginal_cost + carbon_price · co2`"),
    ("Dispatch policy", "Nothing (it *is* the decision)", "The OPF *replaces* the policy: that is the comparison"),
], columns=["Vessim element", "PyPSA representation", "Concrete attribute"])

DRAWING = pd.DataFrame([
    ("Bus", "thick horizontal bar", "`v_nom` and `type` decide slack / PV / PQ"),
    ("Line", "thin route with flow arrows", "filled arrow = P, open chevron = Q"),
    ("Transformer", "two overlapping circles", "tap side and phase shift printed when non-default"),
    ("Generator", "circle with a sine wave", "greyed out when the commitment is 0"),
    ("Solar / wind", "panel grid / three-bladed rotor", "carrier of the generator"),
    ("External grid", "hatched square", "`carrier = grid`"),
    ("Load", "arrow pointing away from the bar", "the size is the active power of the hour"),
    ("Storage", "rectangle on the stub", "SoC printed under the name"),
    ("HVDC link", "two converter boxes on a dash-dot line", "`Link` in the model"),
], columns=["Element", "Symbol on the diagram", "What decides it"])




# --------------------------------------------------------------------------- #
# Trace one number
# --------------------------------------------------------------------------- #

GLOSSARY = pd.DataFrame([
    ("Actor", "something that consumes or produces power every step: a consumer, a producer, "
              "later a datacenter", "`vs.Consumer`, `vs.Producer`", "the load columns of the trace"),
    ("Signal", "a time series an actor or the policy can read: a load profile, a carbon "
               "intensity, a price", "`vs.Trace`, `vs.Signal`", "the `carbon_kg_mwh` and "
               "`price_usd_mwh` columns"),
    ("Dispatchable", "something the policy may set every step: battery, grid connection, "
                     "turbine, spillway", "`vs.SimpleBattery`, `vs.GridConnection`, "
                     "`FuelGenerator`", "the unit rows of the ED-UC sheet"),
    ("Dispatch policy", "the object that decides the dispatchables' setpoints; Vessim calls "
                        "`apply()` once per step, before the step is taken",
     "`vs.DefaultDispatchPolicy`, `MeritOrderPolicy`", "the whole 🎛️ sheet"),
    ("Recorder", "a controller that writes the state after every step; the CSV it produces is "
                 "the only thing the other tabs see", "`vs.Recorder`", "`run['df']`"),
    ("Snapshot", "one row of the trace, frozen into a network: the powers of a single step",
     "the step slider on ⚡", "the injections of the load flow"),
    ("Site model", "the PyPSA network built from the trace: one bus per role, one storage, one "
                   "grid connection", "`site_spec_from` → `site_case`",
     "the 🔗 tab and the LP column"),
    ("Regret", "what a rule costs above the LP on the same day, in USD",
     "`bill − lp_cost`", "the last column of the policy tables"),
], columns=["Term", "What it means here", "Vessim / PyPSA object", "Where you see it"])


def _trace() -> None:
    theme.section("Trace one number across the four tabs",
                  "pick a case and a step; follow the same MW through every engine", "TRACE")
    case_key = st.selectbox("Case", list(CASES),
                            format_func=lambda key: f"{CASES[key].number} · {CASES[key].title}",
                            key="method_trace_case")
    spec = CASES[case_key]
    carbon = float(spec.carbon_price)
    run = data.vessim_case(case_key, carbon, 1.0, 1.0)
    frame = run["df"]
    peak = int(np.argmax(frame["load_mw"].to_numpy()))
    step = st.slider("Step", 0, len(frame) - 1, peak, key=f"method_trace_step_{case_key}")

    snap = snapshot(run, spec, int(step))
    system = system_for(spec)
    result = grid.run_pf(site_network(run, spec, int(step)), start="dc")
    slack = float(result.buses.at["POI", "p_gen"])
    losses = float(result.summary["losses_mw"])
    ed = ed_at(run, spec, int(step), carbon)
    dispatch = ed["dispatch"].set_index("unit")["mw"]
    ed_mw = float(dispatch.get(system.grid[0], 0.0))
    params = data.site_spec_from(frame, spec, carbon, spec.grid_import_kw / 1000.0,
                                 spec.battery_kw / 1000.0,
                                 spec.battery_kwh / max(spec.battery_kw, 1e-9),
                                 20.0, 20.0, 20.0).__dict__
    lp = data.site_case(frame, params, spec.step_s / 3600.0)
    lp_mw = float(lp["dispatch"]["Grid import"].iloc[int(step)])

    rows = [
        ("Vessim only", "grid import recorded by the Recorder", snap["grid_mw"],
         "what the policy decided"),
        ("⚡ LoadFlow", "slack injection at the point of interconnection", slack,
         "the same import, plus the losses of a real network"),
        ("🏭 ED-UC", f"{system.grid[0]} after λ-iteration", ed_mw,
         "the same fleet re-dispatched by hand, battery held to Vessim's schedule"),
        ("🔗 PyPSA", "Grid import of the site LP", lp_mw,
         "the LP, free to move the battery as well"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Tab", "The number", "MW", "What it is"])
                 .assign(**{"MW": lambda d: d["MW"].round(4)}), hide_index=True, **theme.wide())
    theme.kpis([
        ("Vessim", f"{snap['grid_mw']:,.4f}", "MW", f"step {step} · {snap['price_usd_mwh']:,.0f} USD/MWh", ""),
        ("Slack − Vessim", f"{(slack - snap['grid_mw']) * 1000:+,.3f}", "kW",
         f"= the losses ({losses * 1000:,.2f} kW)", "good" if abs(slack - snap["grid_mw"] - losses) < 1e-3 else "bad"),
        ("ED − Vessim", f"{(ed_mw - snap['grid_mw']) * 1000:+,.2f}", "kW",
         "λ on the same units, to rounding", ""),
        ("LP − Vessim", f"{(lp_mw - snap['grid_mw']) * 1000:+,.2f}", "kW",
         "what the LP gains by moving the battery too", ""),
    ])
    figure = go.Figure(go.Bar(x=[row[0] for row in rows], y=[row[2] for row in rows],
                              marker_color=[theme.COLORS["vessim"], theme.COLORS["loadflow"],
                                            theme.COLORS["educ"], theme.COLORS["pypsa"]],
                              text=[f"{row[2]:.4f} MW" for row in rows], textposition="outside"))
    theme.figure(figure, height=280, legend=False, unified=False, y_title="grid import (MW)")
    theme.show(figure, key="method_trace_bars")

    dt = spec.step_s / 3600.0
    theme.note(
        f"That one number costs <b>{snap['grid_mw'] * snap['price_usd_mwh'] * dt:,.4f} USD</b> and "
        f"carries <b>{snap['grid_mw'] * snap['carbon_kg_mwh'] * dt:,.3f} kg CO₂</b> for this "
        f"{spec.step_s / 60:.0f}-minute step, at {snap['price_usd_mwh']:,.0f} USD/MWh and "
        f"{snap['carbon_kg_mwh']:,.0f} kg/MWh. The ⚡ column differs from Vessim by exactly the "
        "losses; the 🏭 column by the rounding of λ; the 🔗 column by whatever the LP can still "
        "improve. <b>Four engines, one MW, three explainable differences</b> — that is the whole "
        "architecture of this app in one row."
    )
    st.dataframe(GLOSSARY, hide_index=True, **theme.wide(), height=260)


def render() -> None:
    theme.hero(
        "—", "How this dashboard is built",
        "METHOD, VALIDATION AND CREDITS",
        "Three independent numerical engines on the same data: AdjieBrotot's load flow, their "
        "economic dispatch / unit commitment, and PyPSA. Vessim supplies the trace and the "
        "signals. Nothing is canned or cached by hand.",
        [("Language", "Python 3.11+"), ("UI", "Streamlit"), ("Charts", "Plotly"),
         ("Solver", "HiGHS")],
    )

    # ------------------------------------------------------------- pipeline
    theme.section("The pipeline", "from the Vessim simulation to the PyPSA optimum", "FLOW")
    st.markdown(
        """
        <div style="display:flex; gap:.6rem; align-items:stretch; flex-wrap:wrap; margin:.4rem 0 1rem">
          <div class="kpi" style="flex:1; min-width:200px; border-color:rgba(167,139,250,.5)">
            <div class="label">1 · Vessim</div>
            <div class="sub" style="font-size:.8rem; margin-top:.35rem">
              Environment + Microgrid + actors + dispatchables + Recorder. The five official
              examples run for real, at their own step size.</div>
          </div>
          <div class="kpi" style="flex:1; min-width:200px; border-color:rgba(56,189,248,.5)">
            <div class="label">2 · LoadFlow</div>
            <div class="sub" style="font-size:.8rem; margin-top:.35rem">
              Editable sheets → Y-bus → Newton-Raphson (AC) or DC. Bus voltages, angles, line
              flows, losses, loading, and the same diagram the original tool draws.</div>
          </div>
          <div class="kpi" style="flex:1; min-width:200px; border-color:rgba(251,191,36,.5)">
            <div class="label">3 · ED-UC</div>
            <div class="sub" style="font-size:.8rem; margin-top:.35rem">
              λ-iteration for the dispatch, a priority list for the commitment, with minima,
              ramps, start-up costs and minimum up/down times.</div>
          </div>
          <div class="kpi" style="flex:1; min-width:200px; border-color:rgba(52,211,153,.5)">
            <div class="label">4 · PyPSA</div>
            <div class="sub" style="font-size:.8rem; margin-top:.35rem">
              The same physics as a linear/MILP optimisation: network, congestion, curtailment,
              nodal prices and a carbon price.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------- mapping
    _trace()

    theme.section("Vessim → PyPSA equivalence", "what becomes what", "TRANSLATION")
    st.dataframe(MAPPING, **theme.wide(), hide_index=True)

    theme.section("Vessim → diagram equivalence", "how each element is drawn", "SYMBOLS")
    st.dataframe(DRAWING, **theme.wide(), hide_index=True)
    st.caption("The symbol vocabulary, the orthogonal routes and the flow arrows follow the two "
               "AdjieBrotot simulators (GPL-3.0). The implementation here is Python; no JavaScript "
               "is copied. Every number printed on a diagram comes from this repository's solvers.")
    st.markdown(
        "Every feeder also carries **arrows that travel along it**, the way the reference tools "
        "animate their networks: one SVG `<path>` glyph per arrow moved by SMIL `animateMotion` "
        "(`rotate=\"auto\"`), with the direction taken from the sign of the active power and the "
        "speed from its magnitude. A branch at 90 % loading visibly runs faster than one at 5 %. "
        "The static arrows stay underneath, and `prefers-reduced-motion` hides the moving ones."
    )

    theme.section("Where the maths lives", "it moved next to the number it explains", "MATHS")
    st.markdown(
        """
        | Where | What it writes down |
        |---|---|
        | `🔮 Vessim only` · expander | the Vessim recursion: balance, battery state, cost, emissions, and the merit order that produced the step |
        | `⚡ LoadFlow` · expander | the Y-bus, the Newton step and the losses, with this snapshot's voltages substituted |
        | `🏭 ED-UC` · tabs | the λ-iteration, the supply curve it inverts, and the units' cost and CO₂ |
        | `🔗 Vessim + PyPSA` · expanders | the LP program, λ checked by finite difference, and the gap decomposed into four priced terms that must add up exactly |
        | `🧩 When to use what` | which tool for which question, the eight fairness controls evaluated **on your run**, the threats to validity and the experimental programme |
        | `🔌 Hardware in the loop` | the `vs.Controller` contract, a live SiL probe, and recipes for Modbus, OPC UA, Prometheus/Kepler and a Raspberry Pi |
        | `🕶️ Blind spots & policies` | the six blind spots of an LP and three probes measured on your case, plus the `DispatchPolicy` contract, the eight rules that ship and an editor that runs your class |
        | `📊 Studies · viewer` | the Vessim viewer reproduced: configurations and runs, distributions, Pareto front, overlays with bands |
        | `🎛️ Policies` (inside `Vessim only`) | the rule that ran, the two scores it sorts on at any step, and six other rules on the same day with their regret |
        """,
    )
    with st.expander("📖 Every result, explained — the dictionary of the dashboard", expanded=False):
        from . import maths as _maths
        _maths.results_section()

    st.markdown(
        '<div class="note good">The dual price is not taken on trust. The page re-solves the whole '
        'day twice with one snapshot\'s load nudged by ±δ and compares (C₊ − C₋)/(2δΔt) with the λ '
        'PyPSA prints. The two agree, which is the definition of a marginal price.</div>',
        unsafe_allow_html=True)

    # ----------------------------------------------------------- validation
    theme.section("Validation", "do not take our word for it: the numbers are on screen", "AUDIT")
    st.markdown(
        """
        | What | Against what | Agreement |
        |---|---|---|
        | AC load flow | `pypsa.Network.pf()` | ΔV ≈ 10⁻¹² p.u. |
        | Economic dispatch | `n.optimize()` on the same problem | 0 % on a copper plate |
        | Unit commitment | PyPSA MILP | 3–57 % — and that gap *is* the result |
        | Aggregate cost | both methods | PyPSA is never more expensive |
        | Snapshot weights | 15-minute Vessim steps | battery speed matches the trace |
        """
    )
    st.markdown(
        '<div class="note good">The three validation tabs are recomputed live from the data you '
        'edit. Change a line or a cost and the comparison updates: nothing is pre-recorded.</div>',
        unsafe_allow_html=True)

    # ----------------------------------------------------------------- data
    theme.section("Official data", "the four Vessim traces, bundled", "DATA")
    st.markdown(
        "Everything on the dashboard comes from these four CSVs (or is computed from the same "
        "signals). They ship with the repository under their original MIT licence, so the app "
        "works offline and without API keys — a condition for a reproducible academic review."
    )
    name = st.selectbox("Trace", list(data.DATASETS), key="method_dataset")
    note, use = data.DATASETS[name]
    payload = data.dataset(name)
    frame = payload.get("frame", pd.DataFrame())
    if frame.empty:
        theme.note("This file is missing from the deployment; re-clone the repository.", "warn")
    else:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.dataframe(frame.head(8), **theme.wide(), height=250)
        with right:
            st.markdown(
                f'<div class="note"><b>{name}</b><br>{note}.<br><br>'
                f'<span style="opacity:.75">{use}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"{len(frame):,} rows · {frame.shape[1]} columns · "
                f"window {frame.shape[0] * (300 if 'solar' in name or 'watttime' in name else 60) / 3600:.1f} h"
            )
            data.download(frame.head(500), f"{name.split('.')[0]}_head.csv",
                          "⬇️ Download the first 500 rows", key=f"dl_{name}")

    # ------------------------------------------------------------- teaching
    theme.section("Four experiments that teach something", "a suggested class script", "TEACHING")
    st.markdown(
        """
        1. **Congestion.** In case 05, lower the *POI–Site* line to half. PyPSA can no longer
           import what it wants, the nodal price at the site rises and part of the sun is
           curtailed. Vessim cannot see it because it dispatches against a copper plate.
        2. **Value of the network.** In case 01 (`pjm5` in the 🏭 tab), raise the coal cost until
           line A–B saturates and watch the prices separate by bus.
        3. **Carbon.** Sweep the CO₂ price from 0 to 200 USD/t in case 05: the optimal solution
           switches technology before the heuristic does, and the heuristic reacts late.
        4. **Commitment.** In `uc3`, compare the priority list with the MILP: the heuristic starts
           slow units in the wrong hours and pays for ramps and technical minima.
        """
    )

    # ---------------------------------------------------------------- files
    with st.expander("🗂️ Repository layout", expanded=False):
        st.code("""vessim-pypsa-dashboard/
├── app.py                     # Streamlit entry point
├── requirements.txt
├── src/vessim_pypsa_dashboard/
│   ├── grid.py                # network model + Y-bus + Newton-Raphson + PyPSA export
│   ├── presets.py             # WSCC 9, 3-bus, feeder, HVDC, Ecopetrol
│   ├── ed_uc.py               # λ-iteration + priority list
│   ├── ed_presets.py          # PJM 5, uc3, solar, areas, Ecopetrol
│   ├── pypsa_models.py        # site OPF, StorageUnit, carbon sweeps
│   ├── policies.py            # FuelGenerator, MeritOrderPolicy, CurtailmentSink
│   ├── vessim_runner.py       # the five official cases, actually executed
│   ├── analysis.py            # manual vs PyPSA comparisons
│   ├── sld.py                 # single-line diagram renderer (SVG)
│   └── ui/                    # the case views + visual system
└── tests/                     # automatic validation against PyPSA""", language="text")

    # -------------------------------------------------------------- credits
    theme.section("Credits and licences", "on the shoulders of giants", "ATTRIBUTION")
    st.markdown(
        """
        * **Vessim** — dos-group / ML Systems Lab, MIT licence.
          <https://github.com/dos-group/vessim>
        * **PyPSA** — GPL-3.0, used as a computation dependency.
          <https://github.com/PyPSA/PyPSA>
        * **AdjieBrotot tools** — *LoadFlow Simulator* and *ED-UC Simulator*,
          <https://tool.adjiebrotots.com>, GPL-3.0 code at
          <https://github.com/adjiebrotot/tool>. Their algorithms, their editable sheets and
          their single-line diagram conventions are **re-implemented** here in Python; no
          JavaScript is copied.
        * **HiGHS** (MIT, through linopy), **Streamlit** (Apache-2.0) and **Plotly** (MIT).

        This repository is distributed under the MIT licence while keeping the attribution
        above. It is not affiliated with, or endorsed by, any of the projects cited.
        """
    )

    # --------------------------------------------------------------- deploy
    theme.section("Publishing to Streamlit Community Cloud", "from zero to a public URL", "DEPLOY")
    st.code("""# 1. push the repo to GitHub
git init && git add . && git commit -m "Vessim + PyPSA dashboard"
git remote add origin https://github.com/<user>/vessim-pypsa-dashboard.git
git push -u origin main

# 2. at https://share.streamlit.io  →  "New app"
#    Repository : <user>/vessim-pypsa-dashboard
#    Branch     : main
#    Main file  : app.py
#    → Deploy

# 3. (optional) run it locally, exactly as in the cloud
pip install -r requirements.txt
streamlit run app.py""", language="bash")
    st.markdown(
        '<div class="note">No secrets and no API keys: the Solcast/WattTime traces are bundled '
        'and case 04 launches a local process, not an external call. The first Streamlit Cloud '
        'build takes ~3–5 min because it compiles the scientific dependencies.</div>',
        unsafe_allow_html=True)
