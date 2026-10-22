"""Tab 1 — Vessim only: the official examples, running for real."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import sld
from . import computing_panel, data, diagram, maths, theme


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _flows_figure(df: pd.DataFrame, dt: float) -> go.Figure:
    index = df.index
    fig = go.Figure()
    series = [
        ("Solar", df["solar_mw"].clip(lower=0), theme.COLORS["solar"], 1),
        ("Grid import", df["grid_mw"].clip(lower=0), theme.COLORS["grid"], 1),
        ("Gas turbine", df.get("gas_mw", pd.Series(0, index=index)), theme.COLORS["gas"], 1),
        ("Battery discharge", df["battery_mw"].clip(lower=0), theme.COLORS["battery"], 1),
        ("Battery charge", df["battery_mw"].clip(upper=0), theme.COLORS["battery"], -1),
        ("Grid export", df["grid_mw"].clip(upper=0), theme.COLORS["export"], -1),
    ]
    for name, values, color, sign in series:
        if float(np.abs(values).sum()) < 1e-12:
            continue
        fig.add_trace(go.Scatter(
            x=index, y=values, name=name, mode="lines", stackgroup="pos" if sign > 0 else "neg",
            line=dict(width=0.6, color=color), fillcolor=color, opacity=0.78,
            hovertemplate="%{y:.3f} MW",
        ))
    fig.add_trace(go.Scatter(x=index, y=df["load_mw"], name="Load", mode="lines",
                             line=dict(color="#f8fafc", width=2.2), hovertemplate="%{y:.3f} MW"))
    theme.figure(fig, height=350, y_title="MW", x_title="")
    fig.update_yaxes(ticksuffix=" MW")
    return fig


def _signals_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["carbon_kg_mwh"], name="Carbon intensity",
                             line=dict(color=theme.COLORS["solar"], width=2),
                             hovertemplate="%{y:.0f} kg/MWh"))
    fig.add_trace(go.Scatter(x=df.index, y=df["price_usd_mwh"], name="Grid price",
                             line=dict(color=theme.COLORS["grid"], width=2, dash="dot"),
                             yaxis="y2", hovertemplate="%{y:.0f} USD/MWh"))
    theme.figure(fig, height=240, legend=True)
    fig.update_layout(yaxis=dict(title="kg CO₂/MWh"),
                      yaxis2=dict(title="USD/MWh", overlaying="y", side="right", showgrid=False))
    return fig


def _soc_figure(df: pd.DataFrame, spec) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df.index, y=df["soc_mwh"], name="SoC",
                               line=dict(color=theme.COLORS["battery"], width=2),
                               fill="tozeroy", fillcolor="rgba(34,211,238,.15)",
                               hovertemplate="%{y:.3f} MWh"))
    theme.figure(fig, height=200, legend=False)
    fig.update_yaxes(title_text=f"SoC (capacity {spec.battery_kwh:,.0f} kWh)")
    return fig


# --------------------------------------------------------------------------- #
# Vessim microgrid diagram, at one recorded step
# --------------------------------------------------------------------------- #

def _microgrid_svg(run: dict, spec, step: int) -> tuple[str, dict]:
    df: pd.DataFrame = run["df"]
    row = df.iloc[int(step)]
    has_gas = float(np.abs(df.get("gas_mw", pd.Series(0.0)).to_numpy()).max()) > 1e-9
    net = sld.microgrid_network(with_gas=has_gas)

    load = float(row["load_mw"])
    solar = float(row["solar_mw"])
    gas = float(row.get("gas_mw", 0.0) or 0.0)
    battery = float(row["battery_mw"])          # + discharge, − charge
    grid = float(row["grid_mw"])                # + import, − export

    flows = sld.Flows()
    flows.bus["POI"] = {"v": 1.0, "price": float(row["price_usd_mwh"])}
    flows.bus["Site"] = {"v": 1.0}
    if has_gas:
        flows.bus["TG"] = {"v": 1.0}
    flows.device["Grid connection"] = {"p": grid, "pmax": spec.grid_import_kw / 1000.0}
    flows.device["Solar"] = {"p": max(solar, 0.0), "pmax": spec.solar_kwp / 1000.0}
    if has_gas:
        flows.device["Gas turbine"] = {"p": gas, "pmax": max(spec.gas_kw / 1000.0, 1e-3)}
    flows.device["Battery"] = {"p": battery, "soc": float(row["soc_mwh"]) / max(spec.battery_kwh / 1000.0, 1e-9)}
    flows.device["Load"] = {"p": load}
    flows.branch["POI–Site"] = {"p0": grid}
    if abs(grid) > 1e-9:                      # an idle tie line gets no "0 %" label
        flows.branch["POI–Site"]["loading"] = 100.0 * abs(grid) / max(spec.grid_import_kw / 1000.0, 1e-9)
    if has_gas:
        flows.branch["Turbine feeder"] = {"p0": gas}
        if abs(gas) > 1e-9:
            flows.branch["Turbine feeder"]["loading"] = 100.0 * gas / max(spec.gas_kw / 1000.0, 1e-9)

    t = df.index[int(step)]
    svg = sld.draw(net, flows, style=sld.Style(
        show_q=False, show_loading=True, colour_by="none", unit="MW",
        title=f"Microgrid at {pd.Timestamp(t).strftime('%H:%M')}",
        subtitle="what the Vessim Recorder saw at this step",
    ))
    balance = {"load": load, "solar": solar, "gas": gas, "battery": battery, "grid": grid,
               "residual": load - (solar + gas + battery + grid)}
    return svg, balance


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

PIPELINE = [
    ("1 · In", "the case's demand, weather, price and carbon signals, and a dispatch policy"),
    ("2 · Model", "Vessim steps the simulation: every actor, battery and connection, one step at a time"),
    ("3 · Out", "a recorded trace — load, PV, battery, grid, SoC, bill — step by step"),
    ("4 · Leaves for", "the other three sheets: this trace is their input, and its bill is their benchmark"),
]


def render(case_key: str, run: dict, spec, carbon_price: float) -> None:
    theme.pipeline(PIPELINE)
    df: pd.DataFrame = run["df"]
    base = run["baseline"]
    dt = run["dt"]
    metrics = run["metrics"]

    theme.section("The simulation", "actors, dispatchables and policy — exactly as Vessim ran them",
                  "01 · VESSIM ONLY")

    load_mwh = float(df["load_mw"].sum() * dt)
    self_suff = 100.0 * (1.0 - base["grid_mwh"] / load_mwh) if load_mwh > 0 else 0.0
    self_suff = float(np.clip(self_suff, -999, 100))

    theme.kpis([
        ("Operating cost", data.fmt_usd(base["total_cost_usd"]), "", "grid + gas − export", ""),
        ("Emissions", data.fmt(base["emissions_t"] * 1000, 1), "kg", f"{base['emissions_t']:.2f} tCO₂", ""),
        ("Energy from grid", data.fmt(base["grid_mwh"], 1), "MWh", f"peak {base['peak_grid_mw']:.2f} MW", "accent"),
        ("Self-sufficiency", data.fmt(self_suff, 1), "%", "1 − grid / load", "good" if self_suff > 50 else "warn"),
        ("Solar harvested", data.fmt(base["solar_mwh"], 1), "MWh", f"{len(df)} steps × {dt*60:.0f} min", ""),
        ("Runtime", f"{run['runtime_s']:.2f}", "s", "real Vessim, not a replay", ""),
    ])

    # --------------------------------------------------------- compute layer
    computing_panel.render(run)

    # -------------------------------------------------------- energy balance
    theme.section("Where the energy went", "the same day, in MWh and in dollars", "BALANCE")
    _energy_balance(df, base, spec, dt)

    # ---------------------------------------------------------------- diagram
    left, right = st.columns([1.05, 1], gap="medium")
    with left:
        theme.show(_flows_figure(df, dt))
    with right:
        theme.show(_signals_figure(df))
        theme.show(_soc_figure(df, spec), key="soc")

    theme.section("The microgrid, one step at a time",
                  "the same single-line diagram the two simulators use, fed by the Recorder", "DIAGRAM")
    step = st.slider("Step", 0, len(df) - 1, min(int(len(df) * 0.55), len(df) - 1),
                     key=f"vg_step_{case_key}",
                     help="Drag to move through the day; every number is a recorded value")
    svg, balance = _microgrid_svg(run, spec, step)
    diagram.render(svg, caption="Vessim microgrid",
                   hint=f"step {step} · {pd.Timestamp(df.index[step]).strftime('%d %b %H:%M')}")

    residual = balance["residual"]
    supply = balance["solar"] + balance["gas"] + balance["battery"] + balance["grid"]
    tone = "good" if abs(residual) < 1e-3 else "warn"
    theme.kpis([
        ("Load", f"{balance['load']:.3f}", "MW", "what the consumer asked for", ""),
        ("Solar", f"{balance['solar']:.3f}", "MW", f"{100*balance['solar']/max(balance['load'],1e-9):.0f} % of load", "accent"),
        ("Battery", f"{balance['battery']:+.3f}", "MW",
         "discharge" if balance["battery"] >= 0 else "charge", "" if abs(balance["battery"]) < 1e-6 else "warn"),
        ("Grid", f"{balance['grid']:+.3f}", "MW",
         "import" if balance["grid"] >= 0 else "export", "accent" if balance["grid"] > 0 else ""),
        ("Gas", f"{balance['gas']:.3f}", "MW", "captive turbine" if balance["gas"] else "none in this case", ""),
        ("Balance check", f"{residual:+.4f}", "MW", "supply − load", tone),
    ])
    with st.expander("🧮 Show the maths behind these numbers", expanded=False):
        maths.step_panel(run, spec, None, None, step)

    if abs(residual) > 1e-3:
        theme.note(
            f"The recorded sources add up to <b>{supply:,.3f} MW</b> against a load of "
            f"<b>{balance['load']:,.3f} MW</b> — a gap of <b>{residual:+,.3f} MW</b>. "
            "This is not a solver error: Vessim lets the grid connection go negative "
            "(export) and the policy may curtail surplus solar, so the residual is the "
            "unserved or spilled energy of this step.", "warn")

    # ------------------------------------------------------------------ policy
    theme.section("Policy inspector", "what Vessim decided at each interval", "DECISION")
    log = run.get("policy_log") or []
    if log:
        max_step = min(len(log) - 1, len(df) - 1)
        col1, col2 = st.columns([1, 1.4])
        with col1:
            index = st.slider("Interval", 0, max_step, min(18, max_step), key=f"pol_step_{case_key}")
            entry = log[index]
            price = entry.get("price", 0.0)
            carbon = entry.get("carbon", 0.0)
            st.markdown(
                f'<div class="note">At t = <b>{index * dt:.2f} h</b> the balance was '
                f'<b>{entry["p_delta"]/1e6:+.3f} MW</b> (positive = surplus), '
                f'with a price of <b>{price*1000:,.0f} USD/MWh</b> and '
                f'<b>{carbon*1000:,.0f} kg CO₂/MWh</b>.</div>',
                unsafe_allow_html=True)
        with col2:
            powers = entry.get("powers", {})
            names = list(powers)
            values = [powers[n] / 1e6 for n in names]
            fig = go.Figure(go.Bar(x=values, y=names, orientation="h",
                                   marker=dict(color=[theme.COLORS["green"] if v <= 0 else theme.COLORS["amber"]
                                                      for v in values]),
                                   hovertemplate="%{x:.3f} MW"))
            fig.add_vline(x=0, line_color="rgba(226,232,240,.5)")
            theme.figure(fig, height=220, legend=False, unified=False)
            fig.update_layout(xaxis_title="MW  (negative = delivers, positive = absorbs)")
            st.plotly_chart(fig, **theme.wide(), config={"displaylogo": False})
    else:
        st.markdown('<div class="small">This policy (DefaultDispatchPolicy) keeps no decision log; '
                    'open case 03 with the merit-order policy to inspect it.</div>',
                    unsafe_allow_html=True)

    # ------------------------------------------------------------- strategies
    if case_key == "solar_battery":
        _render_strategies(case_key, carbon_price)

    # ------------------------------------------------------------------- SiL
    if case_key == "sil":
        _render_sil(run)

    # ---------------------------------------------------------------- extras
    with st.expander("📋 Full time series (MW, MWh, USD)", expanded=False):
        st.dataframe(df.round(4), **theme.wide(), height=320)
        data.download(df, f"vessim_{case_key}.csv", key=f"dl_vessim_{case_key}")

    with st.expander("🧾 Vessim code for this case", expanded=False):
        st.code(spec.code or "# see vessim/examples", language="python")

    with st.expander("📈 Raw Recorder metrics", expanded=False):
        st.json(metrics, expanded=False)


def _energy_balance(df: pd.DataFrame, base: dict, spec, dt: float) -> None:
    """A plain table: every source and sink of the day, in MWh and in USD.

    The watt-scale cases are shown in kWh so that their numbers are readable;
    everything else stays in MWh.
    """
    load_mwh = float(df["load_mw"].sum() * dt)
    w_scale = load_mwh < 0.05                      # a few hundred watts for a day
    factor, unit = (1000.0, "kWh") if w_scale else (1.0, "MWh")

    solar = float(df["solar_mw"].sum() * dt)
    solar_used = float((df["solar_mw"] - df.get("curtailed_mw", pd.Series(0.0, index=df.index))).clip(lower=0).sum() * dt)
    grid_in = float(df["grid_mw"].clip(lower=0).sum() * dt)
    grid_out = float((-df["grid_mw"]).clip(lower=0).sum() * dt)
    gas = float(df.get("gas_mw", pd.Series(0.0, index=df.index)).sum() * dt)
    battery = float(df["battery_mw"].abs().sum() * dt / 2.0)

    rows = [
        ("Load served", load_mwh, "the job the microgrid had to do"),
        ("Solar produced", solar, "what the panels delivered"),
        ("Solar actually used", min(solar_used, load_mwh + battery), "the rest is curtailed or spilled"),
        ("Grid import", grid_in, "energy bought at the wholesale price"),
        ("Grid export", grid_out, "energy sold back"),
        ("Gas turbine", gas, "captive generation" if gas else "not present in this case"),
        ("Battery throughput", battery, "energy moved in and out (counted once)"),
    ]
    frame = pd.DataFrame(
        [(name, round(value * factor, 3), f"{100 * value / load_mwh:.1f} %" if load_mwh > 1e-9 else "—", note)
         for name, value, note in rows],
        columns=["Item", f"Energy ({unit})", "Share of load", "What it means"],
    )
    st.dataframe(frame, **theme.wide(), hide_index=True)

    energy = float(base["energy_cost_usd"])
    carbon = float(base["carbon_cost_usd"])
    theme.kpis([
        ("Energy cost", data.fmt_usd(energy), "", "grid import + gas − export credit", ""),
        ("Carbon cost", data.fmt_usd(carbon), "",
         f"{base['emissions_t']:,.2f} t × {spec.carbon_price:,.0f} USD/t", "warn" if carbon > 0 else ""),
        ("Total", data.fmt_usd(base["total_cost_usd"]), "", "energy + carbon", "accent"),
        ("Average price of load", f"{1000 * base['total_cost_usd'] / max(load_mwh, 1e-9):,.1f}", "USD/MWh",
         "total cost ÷ served energy", ""),
        ("Emission intensity", f"{1000 * base['emissions_t'] / max(load_mwh, 1e-9):,.0f}", "kg/MWh",
         "per unit of load served", ""),
    ])
    if w_scale:
        theme.note("This case runs at a few hundred watts, so its totals are fractions of a cent. "
                   f"The table above is in <b>kWh</b> for that reason; the intensity metrics are "
                   "the ones worth comparing across cases.", "")


def _render_strategies(case_key: str, carbon_price: float) -> None:
    theme.section("Three policies, the same traces", "self-consumption vs. peak shaving vs. carbon-aware",
                  "COMPARE")
    if st.button("⚖️ Run all three strategies", key="btn_strategies", type="primary"):
        st.session_state["strategies_result"] = data.strategy_comparison(case_key, carbon_price)
    result = st.session_state.get("strategies_result")
    if not result:
        st.caption("Press the button: Vessim runs its *strategy_comparison* example three times.")
        return

    rows = []
    for name, payload in result.items():
        b = payload["baseline"]
        rows.append({"Strategy": name.replace("_", " ").title(),
                     "USD": b["total_cost_usd"], "kg CO₂": b["emissions_t"] * 1000,
                     "Grid MWh": b["grid_mwh"], "Export MWh": b.get("export_mwh", 0.0)})
    frame = pd.DataFrame(rows)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=frame["Strategy"], y=frame["USD"], name="Cost (USD)",
                             marker_color=theme.COLORS["grid"], hovertemplate="%{y:,.2f} USD"))
        theme.figure(fig, height=300, legend=False, unified=False, y_title="USD")
        st.plotly_chart(fig, **theme.wide(), config={"displaylogo": False})
    with col2:
        st.dataframe(frame.style.format({"USD": "{:,.2f}", "kg CO₂": "{:,.1f}",
                                         "Grid MWh": "{:,.3f}", "Export MWh": "{:,.3f}"}),
                     **theme.wide(), hide_index=True)


def _render_sil(run: dict) -> None:
    theme.section("Software-in-the-loop", "a real CPU process drives the microgrid", "SiL")
    samples = run.get("extra", {}).get("measured_w") or []
    if not samples:
        st.caption("No SiL samples in this run.")
        return
    frame = pd.DataFrame(samples)
    theme.kpis([
        ("Samples", f"{len(frame)}", "", "1 s steps", ""),
        ("Mean CPU", f"{frame['cpu_seconds'].mean():.3f}", "s", "measured CPU time", "accent"),
        ("Mean power", f"{frame['watts'].mean():.1f}", "W", "wait4 + PUE", "warn"),
        ("Deviation", f"{frame['watts'].std():.2f}", "W", "process variability", ""),
    ])
    fig = go.Figure(go.Bar(x=frame["step"], y=frame["watts"], marker_color=theme.COLORS["vessim"],
                           hovertemplate="step %{x}: %{y:.2f} W"))
    theme.figure(fig, height=260, legend=False, unified=False, x_title="step", y_title="W")
    st.plotly_chart(fig, **theme.wide(), config={"displaylogo": False})
    st.dataframe(frame.round(4), **theme.wide(), hide_index=True)
    theme.note("This case runs at a few hundred watts, so its monetary totals round to "
               "zero. Read it in <b>W and kWh</b>, not in dollars — the point is that the "
               "load is a real process, not a synthetic trace.", "")
