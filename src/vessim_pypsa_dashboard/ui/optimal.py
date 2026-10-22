"""Follow the optimum: does the LP's schedule survive contact with the plant?

The PyPSA tab answers *what is the best schedule*. This sheet answers the next
question, which an LP cannot answer about itself: **is that schedule
executable?** The optimum is converted into a `DispatchPolicy` — one setpoint
per device per step — and Vessim runs the same day again with it. If the model
and the plant agree, the closed-book gap between the executed plan and the
optimum is zero; if they do not, the residual is exactly the part of the plan
the plant could not do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import data, theme

CODE = '''from vessim_pypsa_dashboard.policies import OptimalPlanPolicy

plan = {                       # Vessim sign convention: positive = consumption
    "battery": -p_battery_mw * 1e6,
    "gas":     -p_gas_mw * 1e6,
    "grid":    -p_grid_mw * 1e6,
}
policy = OptimalPlanPolicy(plan, grid_name="grid")
env.add_microgrid(..., policy=policy)      # same actors, same day, new policy
env.run(until=24 * 3600)
'''


def _plan_figure(df: pd.DataFrame, result: dict, dt: float) -> go.Figure:
    optimum = result["optimum"]
    figure = go.Figure()
    index = df.index
    planned_battery = -np.asarray(optimum["battery_mw"], dtype=float)
    planned_grid = np.asarray(optimum["dispatch"].get("Grid import",
                                                      pd.Series(0.0, index=index)), dtype=float)
    figure.add_trace(go.Scatter(x=index, y=planned_battery, name="battery · plan",
                                line=dict(color="#a78bfa", width=2, dash="dot")))
    figure.add_trace(go.Scatter(x=index, y=df["battery_mw"], name="battery · executed",
                                line=dict(color="#a78bfa", width=2.4)))
    figure.add_trace(go.Scatter(x=index, y=planned_grid, name="grid · plan",
                                line=dict(color="#38bdf8", width=2, dash="dot")))
    figure.add_trace(go.Scatter(x=index, y=df["grid_mw"], name="grid · executed",
                                line=dict(color="#38bdf8", width=2.4)))
    theme.figure(figure, height=330, y_title="MW", x_title="")
    figure.update_yaxes(ticksuffix=" MW")
    return figure


def render(case_key: str, run: dict, spec, carbon_price: float) -> None:
    theme.section("Follow the optimum",
                  "PyPSA proposes a schedule; Vessim has to execute it, step by step",
                  "OPTIMAL POLICY")
    st.markdown(
        "<div class='small'>Every other policy in this dashboard <i>decides</i>. This one "
        "<b>obeys</b>: the LP's dispatch is turned into a setpoint per device per step and replayed "
        "through the same microgrid, with the same actors, the same sun and the same battery. "
        "It is the only way to ask an optimiser the question it cannot answer about itself — "
        "<i>is my plan implementable?</i></div>", unsafe_allow_html=True)

    try:
        result = data.optimal_case(case_key, float(carbon_price), 1.0, 1.0)
    except Exception as exc:                       # pragma: no cover - defensive
        theme.note(f"<b>The plan could not be executed:</b> {type(exc).__name__}: {exc}", "bad")
        return

    df = result["run"]["df"]
    dt = result["run"]["dt"]
    optimum = result["optimum"]
    track = result["ledger"]
    lp_cost = result["lp_cost"]
    closed = result["closed_gap"]

    theme.kpis([
        ("LP optimum", f"{lp_cost:,.2f}", "USD", "energy + carbon + cycling", "good"),
        ("Vessim bill", f"{result['tracking_bill']:,.2f}", "USD",
         "the same plan, executed by the plant", ""),
        ("Closed-book gap", f"{closed:+,.2f}", "USD",
         "stock at 24:00 marked at λ_T", "good" if abs(closed) < max(1.0, 1e-3 * lp_cost) else "warn"),
        ("Tracking error", f"{result['max_error_w']:.2e}", "W",
         f"{result['tracking_wh']:.3e} Wh moved off-plan", ""),
    ])

    theme.show(_plan_figure(df, result, dt))

    planned = pd.DataFrame({
        "Device": ["Battery", "Gas", "Grid import"],
        "Plan (MWh)": [
            float(np.abs(optimum["battery_mw"]).sum() * dt),
            float(optimum["dispatch"].get("Gas turbine", pd.Series(0.0)).sum() * dt),
            float(optimum["dispatch"].get("Grid import", pd.Series(0.0)).sum() * dt),
        ],
        "Executed (MWh)": [
            float(df["battery_mw"].abs().sum() * dt),
            float(df.get("gas_mw", pd.Series(0.0)).sum() * dt),
            float(df["grid_mw"].clip(lower=0).sum() * dt),
        ],
        "Difference (MWh)": [
            float((df["battery_mw"].abs() - np.abs(optimum["battery_mw"])).sum() * dt),
            float((df.get("gas_mw", pd.Series(0.0))
                   - optimum["dispatch"].get("Gas turbine", pd.Series(0.0))).sum() * dt),
            float((df["grid_mw"].clip(lower=0)
                   - optimum["dispatch"].get("Grid import", pd.Series(0.0))).sum() * dt),
        ],
    })
    st.dataframe(planned.round(3), hide_index=True, **theme.wide())

    tolerance = max(1.0, 1e-3 * abs(lp_cost))
    if abs(closed) <= tolerance:
        theme.note(
            f"The optimum is <b>executable</b>. Vessim replays the schedule and the closed-book gap "
            f"against the LP is <b>{closed:+,.2f} USD</b> — the plan survives contact with the plant, "
            "because the plant was described honestly and both models price the same battery "
            "cycling and the same terminal stock. That is the loop closing: PyPSA proposes, Vessim "
            "disposes, and the two books agree to the cent."
        )
    else:
        theme.note(
            f"The plan is <b>not</b> executable as written: after replaying it, the closed-book gap "
            f"is <b>{closed:+,.2f} USD</b>. The usual cause in this dashboard is that the LP's device "
            "is not the plant's device — a battery below the 1 kW floor of the site model, a rated "
            "power the hardware does not have, or an efficiency the model rounds. The gap is not the "
            "optimiser being wrong: it is the model being *coarser than the plant*, and this sheet is "
            "where that becomes a number instead of an opinion.", "warn")

    with st.expander("🧾 How the plan is executed", expanded=False):
        st.code(CODE, language="python")
        st.caption("`OptimalPlanPolicy` clips every planned setpoint to the device's feasible range, "
                   "lets the grid absorb whatever the plan could not deliver, and records the "
                   "mismatch as a tracking error. Nothing is re-optimised: this is the same plan.")

    with st.expander("🧮 The ledger of the executed plan", expanded=False):
        frame = track.get("frame") if isinstance(track, dict) else None
        if frame is not None:
            st.dataframe(frame.round(2), hide_index=True, **theme.wide())
        st.caption("Six priced terms, the same table the decision guide checks. A perfect tracker "
                   "has every difference term at zero and only the terminal stock left.")
