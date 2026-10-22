"""Tab 3 — ED/UC (AdjieBrotot), applied to the case's own fleet.

No PJM 5-bus, no textbook unit table: the units are the ones Vessim just
dispatched — the on-site turbine, the grid connection, the battery and the
photovoltaic plant of *this* case — with the case's own prices, carbon intensity
and limits.  The method is the original tool's: λ-iteration for the economic
dispatch of one snapshot, a priority list with minimum up and down times for the
commitment of the day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..vessim_system import (
    carbon_cost_usd_per_mwh,
    demand_at,
    ed_at,
    fleet_table,
    snapshot,
    system_for,
    uc_day,
    units_at,
)
from . import data, sheets, theme


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _supply_curve(units: pd.DataFrame, demand: float, lam: float) -> go.Figure:
    figure = go.Figure()
    grid = np.linspace(0.0, float(units["marginal_cost"].max()) * 1.6 + 50.0, 400)
    total = np.zeros_like(grid)
    for _, unit in units.iterrows():
        b = float(unit["marginal_cost"])
        c = float(unit["marginal_cost_quadratic"])
        p = np.clip((grid - b) / (2 * c) if c > 0 else np.where(grid > b, unit["p_nom"], 0.0),
                    0.0, float(unit["p_nom"]))
        total = total + p
        figure.add_trace(go.Scatter(x=p, y=grid, mode="lines", name=unit.name,
                                    line=dict(width=0), fill="tozeroy", opacity=0.25))
    figure.add_trace(go.Scatter(x=total, y=grid, mode="lines", name="aggregate supply",
                                line=dict(color=theme.COLORS["sky"], width=3)))
    figure.add_vline(x=demand, line_dash="dash", line_color=theme.COLORS["amber"],
                     annotation_text=f"demand {demand:.2f} MW", annotation_position="top")
    figure.add_hline(y=lam, line_dash="dot", line_color=theme.COLORS["green"],
                     annotation_text=f"λ {lam:,.1f} USD/MWh", annotation_position="bottom right")
    theme.figure(figure, height=340, x_title="dispatch (MW)", y_title="marginal cost (USD/MWh)")
    return figure


def _iteration_figure(history: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=history["iteration"], y=history["error_mw"], mode="lines+markers",
                                line=dict(color=theme.COLORS["sky"], width=2)))
    theme.figure(figure, height=300, legend=False, unified=False,
                 x_title="iteration", y_title="supply − demand (MW)")
    figure.update_yaxes(type="log")
    return figure


def _day_figure(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Bar(x=frame.index, y=frame["gas_mw"], name="manual · gas",
                            marker_color="#f59e0b"))
    figure.add_trace(go.Bar(x=frame.index, y=frame["grid_mw"], name="manual · grid",
                            marker_color="#38bdf8"))
    figure.add_trace(go.Scatter(x=frame.index, y=frame["vessim_gas_mw"], name="Vessim · gas",
                                line=dict(color="#fbbf24", width=1.6, dash="dot")))
    figure.add_trace(go.Scatter(x=frame.index, y=frame["vessim_grid_mw"], name="Vessim · grid",
                                line=dict(color="#0ea5e9", width=1.6, dash="dot")))
    figure.add_trace(go.Scatter(x=frame.index, y=frame["demand_mw"], name="net demand",
                                line=dict(color="#94a3b8", width=1)))
    figure.update_layout(barmode="stack")
    theme.figure(figure, height=380, x_title="step", y_title="MW")
    return figure


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

PIPELINE = [
    ("1 · In", "the day's net load and the case's dispatchable fleet, priced with that hour's cost"),
    ("2 · Model", "λ-iteration for the economic dispatch; a priority list with up/down times for commitment"),
    ("3 · Out", "unit outputs and the marginal price λ, checked against PyPSA's MILP"),
    ("4 · Leaves for", "the Vessim + PyPSA sheet, which adds the network the manual method ignores"),
]


def render(case_key: str, run: dict, spec, carbon: float) -> None:
    theme.pipeline(PIPELINE)
    system = system_for(spec)
    theme.hero(
        "🏭", f"Dispatch and commitment on system {spec.number}",
        f"CASE {spec.number} · {spec.title.upper()} · {run.get('policy_kind', 'policy')}",
        "λ-iteration and the priority list, exactly as the AdjieBrotot tool implements them — but "
        f"the fleet is <b>the one case {spec.number} installed</b>: "
        + ", ".join(f"<b>{name}</b> ({rating})"
                    for kind, name, rating in system.equipment()
                    if kind in ("Generator", "Storage", "Connection"))
        + ". The fuel cost and CO₂, the grid's price signal and import limit, the battery's state "
        "of charge and the photovoltaic profile all come from the Vessim run, so every number on "
        "this page can be traced back to a step of the simulation.",
        [("Fleet", f"{len(system.equipment()) - len(system.groups)} units"),
         ("Method", "λ-iteration"), ("Commitment", "priority list")],
    )
    with st.expander("🔧 What this case installs — and what the policy actually ran",
                     expanded=theme.density() == "Full"):
        st.dataframe(pd.DataFrame(system.equipment(), columns=["Kind", "Equipment", "Rating"]),
                     hide_index=True, **theme.wide(), height=210)
        st.caption(
            f"The dispatch on this page is not a hypothetical: it is the fleet above, dispatched "
            f"by <b>{run.get('policy_kind', 'the case policy')}</b> in the run you are looking at, "
            "and re-solved here by hand. Same units, same prices, same carbon.")

    df = run["df"]
    steps = len(df)
    dt = spec.step_s / 3600.0

    # ------------------------------------------------------------- ED at a step
    theme.section("Economic dispatch of one snapshot",
                  "the marginal unit sets the price — and you can see which one it is", "ED")
    step = st.slider("Step", 0, steps - 1, int(np.argmax(df["load_mw"].to_numpy())),
                     key=f"ed_step_{case_key}")
    ed = ed_at(run, spec, int(step), carbon)
    snap = ed["snapshot"]
    units = ed["deltas"]

    left, right = st.columns([1.1, 1])
    with left:
        theme.show(_supply_curve(units, max(ed["demand_mw"], 0.0), ed["lam"]))
        st.caption("Each band is one unit's supply curve; the aggregate is what λ-iteration "
                   "inverts. The horizontal line is the λ the solver converged to, the vertical "
                   "line is the demand the fleet had to cover.")
    with right:
        table = ed["dispatch"].copy()
        sheets.tabular(table.set_index("unit"), height=170,
                       formats={"mw": "{:.3f}", "marginal USD/MWh": "{:.2f}",
                                "cost USD/h": "{:.2f}", "emissions kg/h": "{:.1f}",
                                "co2 kg/MWh": "{:.1f}"})
        theme.kpis([
            ("Demand", f"{ed['demand_mw']:,.3f}", "MW",
             "load − solar − battery (the battery follows Vessim)", ""),
            ("λ", f"{ed['lam']:,.2f}", "USD/MWh",
             f"{ed['iterations']} iterations · {'converged' if ed['converged'] else 'not converged'}",
             "good" if ed["converged"] else "bad"),
            ("Cost", f"{ed['cost_usd_h']:,.2f}", "USD/h",
             f"carbon included at {carbon:,.0f} USD/t", ""),
        ])

    theme.section("The fleet, and which units λ may actually move",
                  "exogeneity stated unit by unit, not assumed", "FLEET")
    fleet = fleet_table(run, spec, int(step), carbon)
    st.dataframe(fleet, hide_index=True, **theme.wide(), height=250)
    st.caption(
        "Two rows say *yes*: the turbine and the connection — those are what λ-iteration allocates, "
        "with carbon priced as τ·e. The sun is a profile, the battery follows the schedule the "
        "case's policy produced, and the loads are demand. A comparison that quietly moved the "
        "battery would be measuring the LP, not the manual method; a comparison that ignored the "
        "battery would be measuring a system this case does not have.")

    theme.section("The same step, three answers", "the manual method, the rule Vessim ran, the LP",
                  "COMPARE")
    lp = _lp_at_step(case_key, run, spec, carbon, int(step))
    manual = dict(zip(ed["dispatch"]["unit"], ed["dispatch"]["mw"]))
    vessim = dict(zip(ed["vessim"]["unit"], ed["vessim"]["mw"]))
    table_rows = []
    fleet_names = [system.gas[0]] if system.gas else []
    fleet_names.append(system.grid[0])
    for unit in [u for u in fleet_names if u in set(manual) | set(vessim)]:
        table_rows.append({
            "unit": unit,
            "manual MW": manual.get(unit, 0.0),
            "Vessim MW": vessim.get(unit, 0.0),
            "PyPSA MW": lp.get(unit, float("nan")) if lp else float("nan"),
        })
    sheets.tabular(pd.DataFrame(table_rows).set_index("unit"), height=140,
                   formats={"manual MW": "{:.3f}", "Vessim MW": "{:.3f}", "PyPSA MW": "{:.3f}"})
    st.caption("The manual method sees the *same* fleet as Vessim: when the turbine is the "
               "marginal unit, the two agree to rounding. The LP may disagree, because it can also "
               "move the battery — which is the next section's subject, not this one's.")

    tab_iter, tab_maths, tab_units = st.tabs(["Iterations", "The maths", "Units and limits"])
    with tab_iter:
        if len(ed["history"]):
            theme.show(_iteration_figure(ed["history"]))
            sheets.tabular(ed["history"].round(4).tail(8).set_index("iteration"), height=200)
        else:
            st.info("This snapshot has no deficit to dispatch: the sun and the battery cover the "
                    "load, so there is nothing for the units to do.")
    with tab_maths:
        st.latex(r"\min \sum_i C_i(P_i) \quad \text{s.t.} \quad \sum_i P_i = D, \;\; "
                 r"P_i^{\min} \le P_i \le P_i^{\max}")
        st.latex(r"C_i(P) = b_i P + c_i P^2, \qquad \lambda = b_i + 2 c_i P_i "
                 r"\;\;\text{for every marginal } i")
        st.markdown(
            f"At step {snap['step']} the fleet solves "
            f"`Σ P = {ed['demand_mw']:.3f} MW`. The bisection moves λ until the aggregate supply "
            f"equals the demand; at convergence λ = **{ed['lam']:,.2f} USD/MWh** and the cost is "
            f"**{ed['cost_usd_h']:,.2f} USD/h**. Carbon enters as a price per MWh: "
            f"`τ · e` with τ = {carbon:,.0f} USD/t and e the emission factor of each unit "
            f"(turbine {spec.gas_co2} kg/kWh, grid {spec.grid_carbon} kg/kWh).")
    with tab_units:
        view = units[["p_nom", "p_min_pu", "marginal_cost", "marginal_cost_quadratic", "co2"]] \
            .rename(columns={"p_nom": "p_nom MW", "marginal_cost": "b USD/MWh",
                             "marginal_cost_quadratic": "c USD/MWh²", "co2": "kg/kWh"})
        sheets.tabular(view, height=180, formats={"p_nom MW": "{:.3f}", "b USD/MWh": "{:.2f}",
                                                  "c USD/MWh²": "{:.5f}", "kg/kWh": "{:.3f}",
                                                  "p_min_pu": "{:.2f}"})
        st.caption("The convex term `c` is what makes the supply curve strictly increasing: "
                   "without it the marginal unit cannot be split and λ-iteration oscillates between "
                   "two commitments. It is a heat-rate curvature, not a fudge — and it is small "
                   "enough not to move the ranking.")

    # ------------------------------------------------------------- UC over the day
    theme.section("Commitment over the whole day",
                  "a priority list with minimum up and down times", "UC")
    uc = uc_day(run, spec, carbon)
    frame = uc["frame"]
    if not system.gas:
        theme.note(
            f"Case {spec.number} has no thermal fleet: the only dispatchable is "
            f"<b>{system.grid[0]}</b>, so the commitment is trivially on for the whole day. The "
            "sheet shows that instead of importing a turbine this system does not have — the "
            "priority list becomes interesting only where a second unit exists (case 05).")
    
    theme.kpis([
        ("Manual day cost", f"{uc['cost_usd']:,.2f}", "USD",
         f"{uc['emissions_t']:,.1f} t CO₂ · {uc['starts']} starts", ""),
        ("Vessim bill", f"{uc['vessim_cost_usd']:,.2f}", "USD",
         f"{uc['vessim_emissions_t']:,.1f} t CO₂",
         "good" if uc["cost_usd"] <= uc["vessim_cost_usd"] else "bad"),
        ("Difference", f"{uc['cost_usd'] - uc['vessim_cost_usd']:+,.2f}", "USD",
         f"{(uc['cost_usd'] / max(uc['vessim_cost_usd'], 1e-9) - 1):+.2%} against the rule", ""),
        ("Unserved", f"{uc['unserved_mwh']:,.3f}", "MWh",
         "should be zero: a commitment that cannot serve its load is a failure", 
         "bad" if uc["unserved_mwh"] > 1e-9 else "good"),
    ])
    theme.show(_day_figure(frame))
    st.caption("Bars: the manual commitment and dispatch. Dotted lines: what Vessim actually did. "
               "The manual method may start and stop the turbine at slightly different steps — that "
               "is the minimum up and down time doing its job, and it is visible here as the gap "
               "between the bars and the lines.")

    tab_day, tab_commit, tab_maths_uc = st.tabs(["Day table", "Commitment", "The rule"])
    with tab_day:
        view = frame[["time", "demand_mw", "gas_mw", "grid_mw", "vessim_gas_mw", "vessim_grid_mw",
                      "lam", "price_usd_mwh", "cost_usd"]].copy()
        sheets.tabular(view.tail(24).set_index("time"), height=320,
                       formats={c: "{:.3f}" for c in ("demand_mw", "gas_mw", "grid_mw",
                                                      "vessim_gas_mw", "vessim_grid_mw",
                                                      "price_usd_mwh")} | {"lam": "{:.1f}",
                                                                           "cost_usd": "{:.2f}"})
        st.caption("Last 24 steps. Every column is traceable: `vessim_*` is the Recorder, `gas_mw` "
                   "and `grid_mw` are the manual method, `lam` is the price the λ-iteration found.")
    with tab_commit:
        state = frame[["time", "committed", "demand_mw", "gas_mw"]].copy()
        sheets.tabular(state.set_index("time"), height=320,
                       formats={"demand_mw": "{:.2f}", "gas_mw": "{:.2f}"})
    with tab_maths_uc:
        st.latex(r"\text{commit } u_{i,t} \in \{0,1\}, \quad "
                 r"u_{i,t} - u_{i,t-1} \le u_{i,t+T_i^{\text{up}}}, \quad "
                 r"u_{i,t-1} - u_{i,t} \le 1 - u_{i,t+T_i^{\text{down}}}")
        st.latex(r"\text{then dispatch: } \min \sum_i C_i(P_{i,t}) \;\; \text{s.t.} \;\; "
                 r"\sum_i P_{i,t} = D_t, \;\; u_{i,t} P_i^{\min} \le P_{i,t} \le u_{i,t} P_i^{\max}")
        st.markdown(
            f"The priority list commits the cheapest unit that can serve the demand, honours "
            f"`min_up = min_down = {max(2, round(0.5 / dt))} steps` "
            f"({0.5:.0f} h at {dt * 60:.0f} min per step), and then runs λ-iteration inside the "
            "commitment. That is exactly the structure of the original tool; what changed is the "
            "fleet, which is now the case's own.")

    # ------------------------------------------------------------------ verdict
    theme.section("What the manual method buys, and what it costs", "", "VERDICT")
    st.markdown(
        f"""
        * **It is auditable.** Every step shows the λ, the commitment and the marginal unit; the
          LP gives you a price vector and a certificate, not a decision rule a plant operator can
          run on a whiteboard.
        * **It is myopic.** The dispatch at step `t` knows the price at step `t` and nothing else.
          The LP knows the entire day, which is why its cost ({_lp_cost(case_key, run, spec, carbon):,.2f} USD)
          can sit below both the manual method ({uc['cost_usd']:,.2f}) and the rule
          ({uc['vessim_cost_usd']:,.2f}).
        * **It is fair.** The manual method is given the same fleet, the same limits and the same
          prices as Vessim's policy; the difference is the method alone.
        """
    )


def _lp_at_step(case_key: str, run: dict, spec, carbon: float, step: int) -> dict:
    """PyPSA's dispatch at the same snapshot, from the site model."""
    try:
        params = data.site_spec_from(
            run["df"], spec, carbon, spec.grid_import_kw / 1000.0, spec.battery_kw / 1000.0,
            spec.battery_kwh / max(spec.battery_kw, 1e-9), 20.0, 20.0, 20.0).__dict__
        opt = data.site_case(run["df"], params, spec.step_s / 3600.0)
        dispatch = opt["dispatch"]
        # the LP names its columns after the bus roles; the sheet names them after
        # the equipment the case installed — say they are the same thing once
        aliases = {"Grid import": system_for(spec).grid[0], "Grid": system_for(spec).grid[0]}
        if system_for(spec).gas:
            aliases["Gas turbine"] = system_for(spec).gas[0]
        return {aliases.get(column, column): float(dispatch[column].iloc[int(step)])
                for column in dispatch.columns}
    except Exception:
        return {}


def _lp_cost(case_key: str, run: dict, spec, carbon: float) -> float:
    try:
        params = data.site_spec_from(
            run["df"], spec, carbon, spec.grid_import_kw / 1000.0, spec.battery_kw / 1000.0,
            spec.battery_kwh / max(spec.battery_kw, 1e-9), 20.0, 20.0, 20.0).__dict__
        opt = data.site_case(run["df"], params, spec.step_s / 3600.0)
        return float(opt["energy_cost"] + opt["carbon_cost"])
    except Exception:
        return float("nan")
