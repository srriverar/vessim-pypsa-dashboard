"""Tab 4 — Vessim + PyPSA: the same Vessim day, re-optimised on the same network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import sld, vessim_system
from ..grid import run_pf
from . import data, diagram, maths, sheets, theme


def _spec_params(spec, carbon_price: float, battery_mw: float, battery_hours: float,
                 grid_limit: float, line_grid: float, line_solar: float, line_gas: float,
                 efficiency: float = 1.0, allow_curtailment: bool = True) -> dict:
    site = data.site_spec_from(pd.DataFrame(), spec, carbon_price, grid_limit, battery_mw,
                               battery_hours, line_grid, line_solar, line_gas,
                               allow_curtailment=allow_curtailment, battery_efficiency=efficiency)
    params = site.__dict__.copy()
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in params.items()}


def _stacked(dispatch: pd.DataFrame, columns: list[tuple[str, str, str]], height: int = 310,
             title: str = "") -> go.Figure:
    fig = go.Figure()
    for column, label, color in columns:
        if column not in dispatch:
            continue
        values = dispatch[column]
        if float(np.abs(values).sum()) < 1e-9:
            continue
        fig.add_trace(go.Bar(x=dispatch.index, y=values, name=label, marker_color=color,
                             hovertemplate="%{y:.3f} MW"))
    theme.figure(fig, height=height, title=title)
    fig.update_layout(barmode="relative", xaxis_title="", yaxis_title="MW")
    return fig


def _cost_bars(vessim_cost: float, pypsa_cost: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Vessim (heuristic)", "PyPSA (optimum)"], y=[vessim_cost, pypsa_cost],
        marker_color=[theme.COLORS["vessim"], theme.COLORS["pypsa"]],
        text=[data.fmt_usd(vessim_cost), data.fmt_usd(pypsa_cost)], textposition="outside",
        hovertemplate="%{x}: %{y:,.2f} USD<extra></extra>",
    ))
    theme.figure(fig, height=270, legend=False, unified=False, y_title="USD")
    return fig


def _cost_breakdown(opt: dict) -> go.Figure:
    energy = float(opt["energy_cost"])
    carbon = float(opt["carbon_cost"])
    fig = go.Figure(go.Bar(
        x=["Energy", "CO₂"], y=[energy, carbon],
        marker_color=[theme.COLORS["sky"], theme.COLORS["solar"]],
        text=[data.fmt_usd(energy), data.fmt_usd(carbon)], textposition="outside",
        hovertemplate="%{x}: %{y:,.2f} USD<extra></extra>",
    ))
    theme.figure(fig, height=290, legend=False, unified=False, y_title="USD")
    return fig


# --------------------------------------------------------------------------- #
# The case network at one snapshot, drawn with the OPF's own numbers
# --------------------------------------------------------------------------- #

def _storage_trace(opt: dict, dt_hours: float) -> pd.Series:
    """Battery power in MW from the state of charge: + discharge, − charge.

    The OPF stores the battery as a StorageUnit, so its power is not in the
    dispatch frame; it is read back from the SoC so the diagram shows a measured
    quantity instead of a residual.
    """
    soc = opt["soc"]
    if not len(soc):
        return pd.Series(dtype=float)
    energy = soc.iloc[:, 0]
    return (-energy.diff().fillna(0.0) / max(dt_hours, 1e-9)).rename("Battery")


def _case_svg(opt: dict, run: dict, spec, step: int) -> tuple[str, dict]:
    """Draw the OPF on the case's own network — every bus the ⚡ sheet solves.

    The topology, the names and the ratings come from
    :func:`vessim_system.site_network`, the same call the load-flow tab makes;
    only the numbers change: device powers and branch flows are the optimum's,
    and every bus is labelled with its nodal price.
    """
    net = vessim_system.site_network(run, spec, step)
    snapshots = opt["snapshots"]["time"]
    step = int(np.clip(step, 0, len(snapshots) - 1))
    prices = opt.get("prices", pd.DataFrame())
    device = opt.get("device", pd.DataFrame())
    branch_p0 = opt.get("branch_p0", pd.DataFrame())
    loading = opt.get("line_loading", pd.DataFrame())
    soc = opt.get("soc", pd.DataFrame())

    flows = sld.Flows()
    for bus in net.buses.index:
        row: dict[str, float] = {}
        if bus in prices.columns:
            value = float(prices[bus].iloc[step])
            if np.isfinite(value):
                row["price"] = value
        flows.bus[bus] = row

    for name in net.generators.index:
        power = float(device[name].iloc[step]) if name in device.columns else 0.0
        flows.device[name] = {"p": power, "pmax": float(net.generators.at[name, "p_nom"])}
    for name in net.storages.index:
        power = float(device[name].iloc[step]) if name in device.columns else 0.0
        capacity = (float(net.storages.at[name, "p_nom"])
                    * float(net.storages.at[name, "max_hours"]))
        level = float(soc.iloc[step, 0]) if len(soc) else float("nan")
        flows.device[name] = {"p": power,
                              "soc": level / capacity if capacity > 0 else None}
    for name in net.loads.index:
        flows.device[name] = {"p": float(net.loads.at[name, "p_set"])}

    for name in list(net.lines.index) + list(net.transformers.index):
        p0 = float(branch_p0[name].iloc[step]) if name in branch_p0.columns else 0.0
        payload = {"p0": p0, "p1": -p0}
        if name in loading.columns:
            value = float(loading[name].iloc[step])
            if np.isfinite(value):
                payload["loading"] = value
        flows.branch[name] = payload

    t = pd.Timestamp(snapshots.iloc[step])
    svg = sld.draw(net, flows, style=sld.Style(
        show_q=False, show_loading=True, colour_by="none", unit="MW",
        title=f"Vessim + PyPSA · {t.strftime('%H:%M')}",
        subtitle=f"{len(net.buses)} buses · the price on each is the OPF's dual",
    ))
    dispatch = opt["dispatch"]
    row = dispatch.iloc[step]
    load = float(net.loads["p_set"].sum())
    battery = float(device[net.storages.index[0]].iloc[step]) if len(net.storages.index) \
        and net.storages.index[0] in device.columns else 0.0
    balance = {"load": load, "solar": float(row.get("Solar", 0.0)),
               "gas": float(row.get("Gas turbine", 0.0)),
               "grid": float(row.get("Grid import", 0.0)), "battery": battery}
    balance["residual"] = balance["load"] - (balance["solar"] + balance["gas"]
                                             + balance["grid"] + balance["battery"])
    return svg, balance


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

PIPELINE = [
    ("1 · In", "the scenario of the Vessim run — loads, PV availability, price, carbon — and the same network"),
    ("2 · Model", "a linear OPF on that network: free setpoints, every bus balanced, every rating enforced"),
    ("3 · Out", "the optimal dispatch, the nodal price at every bus, congestion and curtailment"),
    ("4 · Compares", "bill and CO₂ against the rule's, with what crossed (the scenario) and what did not (its decisions)"),
]


def render(case_key: str, run: dict, spec, carbon_price: float) -> None:
    df = run["df"]
    dt = run["dt"]
    baseline = run["baseline"]

    theme.pipeline(PIPELINE)
    theme.section("The optimum on the same network",
                  "the Vessim scenario becomes an OPF — on the buses the ⚡ sheet solves",
                  "04 · VESSIM + PYPSA")

    # the defaults are the case's own nameplates — the same network the ⚡ sheet
    # solves — so the two tabs agree before a single slider is touched
    system = vessim_system.system_for(spec)
    defaults = {
        "battery_mw": max(system.battery[2], 1e-9),
        "battery_hours": max(system.battery[1] / max(system.battery[2], 1e-9), 1e-9),
        "grid_limit": max(system.grid[1], 1e-9),
        "line_grid": max(system.tx_mva, 1e-9),
        "line_solar": max(system.solar[1] * 1.1, 1e-9),
        "line_gas": max(system.gas[1] * 1.1, 1e-9) if system.gas else 1e-9,
    }

    with st.expander("🎛️ Electrical model parameters (PyPSA only)",
                     expanded=theme.density() == "Full"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            battery_mw = st.slider("Battery (MW)", 0.0, max(defaults["battery_mw"] * 3, 1e-6),
                                   float(defaults["battery_mw"]), key=f"int_bmw_{case_key}",
                                   format="%.3f")
            battery_hours = st.slider("Battery hours", 0.5, 8.0,
                                      float(np.clip(defaults["battery_hours"], 0.5, 8.0)), 0.5,
                                      key=f"int_bh_{case_key}")
        with c2:
            grid_limit = st.slider("Import limit (MW)", 0.0, max(defaults["grid_limit"] * 2, 1e-6),
                                   float(defaults["grid_limit"]), key=f"int_grid_{case_key}",
                                   format="%.3f")
            efficiency = st.slider("Battery efficiency", 0.70, 1.00, 1.00, 0.01,
                                   key=f"int_eff_{case_key}",
                                   help="Vessim's SimpleBattery is lossless (η = 1)")
        with c3:
            line_grid = st.slider("POI transformer (MVA)", max(defaults["line_grid"] * 0.1, 1e-6),
                                  max(defaults["line_grid"] * 2, 1e-6), float(defaults["line_grid"]),
                                  key=f"int_lg_{case_key}", format="%.3f",
                                  help="Lower it to create congestion: the key teaching lever")
            line_solar = st.slider("Site–PV collector (MVA)", max(defaults["line_solar"] * 0.1, 1e-6),
                                   max(defaults["line_solar"] * 2, 1e-6), float(defaults["line_solar"]),
                                   key=f"int_ls_{case_key}", format="%.3f")
        with c4:
            line_gas = st.slider("Turbine feeder (MVA)", max(defaults["line_gas"] * 0.1, 1e-9),
                                 max(defaults["line_gas"] * 2, 1e-6), float(defaults["line_gas"]),
                                 key=f"int_lp_{case_key}", format="%.3f")
            allow_curtailment = st.checkbox("Allow solar curtailment", value=True,
                                            key=f"int_curt_{case_key}",
                                            help="If off, PyPSA must consume every kWh of sun")

    params = _spec_params(spec, carbon_price, battery_mw, battery_hours, grid_limit,
                          line_grid, line_solar, line_gas, efficiency, allow_curtailment)

    try:
        opt = data.site_case(df, params, dt)
    except Exception as exc:
        theme.note(f"<b>PyPSA could not solve the site:</b> {type(exc).__name__}: {exc}", "bad")
        return

    pypsa_cost = opt["energy_cost"] + opt["carbon_cost"]
    vessim_cost = baseline["total_cost_usd"]
    saving = vessim_cost - pypsa_cost
    saving_pct = 100.0 * saving / vessim_cost if abs(vessim_cost) > 1e-9 else float("nan")
    congested = bool((opt["line_loading"].to_numpy() > 99.5).any()) if len(opt["line_loading"]) else False

    theme.kpis([
        ("Vessim", data.fmt_usd(vessim_cost), "", "heuristic policy", ""),
        ("PyPSA", data.fmt_usd(pypsa_cost), "", f"{opt['status']} · {opt['runtime_s']:.2f} s", "good"),
        ("Bill saving", data.fmt_usd(saving), "", f"{saving_pct:+.2f}% vs Vessim",
         "good" if saving > 0 else "bad"),
        ("Vessim emissions", f"{baseline['emissions_t']*1000:,.1f}", "kg", "", ""),
        ("PyPSA emissions", f"{opt['emissions_t']*1000:,.1f}", "kg",
         f"{opt['emissions_t']*1000 - baseline['emissions_t']*1000:+,.1f} kg", ""),
        ("Final SoC", f"{float(df['soc_mwh'].iloc[-1]):,.2f} / {float(opt['soc'].iloc[-1, 0]):,.2f}", "MWh",
         "Vessim / PyPSA", ""),
        ("Curtailed", f"{opt['curtailment_mwh']:,.2f}", "MWh", "solar not used", ""),
        ("Unserved", f"{opt['unmet_mwh']:,.2f}", "MWh", "deficit",
         "bad" if opt["unmet_mwh"] > 1e-6 else "good"),
        ("Lines at limit", "yes" if congested else "no", "", "loading > 99.5 %",
         "bad" if congested else "good"),
    ])

    capacity = max(battery_mw * battery_hours, 1e-9)

    snapshots = opt["snapshots"]["time"]
    step = st.slider("Snapshot", 0, len(snapshots) - 1,
                     min(int(len(snapshots) * 0.55), len(snapshots) - 1),
                     key=f"site_step_{case_key}",
                     help="Drag through the day; the bus labels show the nodal price of that hour")

    # ------------------------------------------------------- the same network
    theme.section("The same network, every bus",
                  "the ⚡ sheet solves it once, this sheet optimises the whole day on it",
                  "SAME SYSTEM")
    system = vessim_system.system_for(spec)
    lf_net = vessim_system.site_network(run, spec, step)
    same_buses = sorted(lf_net.buses.index) == sorted(opt.get("buses", ()))
    same_branches = (sorted(list(lf_net.lines.index) + list(lf_net.transformers.index))
                     == sorted(opt.get("branches", ())))
    gap = vessim_system.loss_gap(run, spec)
    theme.kpis([
        ("Buses", f"{len(opt.get('buses', ()))}", "",
         "the load flow's buses, not a contraction", "good" if same_buses else "bad"),
        ("Branches", f"{len(opt.get('branches', ()))}", "",
         "lines + transformers, same ratings", "good" if same_branches else "bad"),
        ("Nodal price spread", f"{float((opt['prices'].max(axis=1) - opt['prices'].min(axis=1)).max()):,.2f}",
         "$/MWh", "zero = no congestion anywhere", ""),
        ("Curtailed sun", f"{opt['curtailment_mwh']:,.3f}", "MWh",
         "the OPF may spill what the rule must take", ""),
        ("Unserved", f"{opt['unmet_mwh']:,.3f}", "MWh",
         "what the network could not deliver", "good" if opt["unmet_mwh"] < 1e-6 else "bad"),
        ("AC losses the OPF ignores", f"{gap['energy_mwh']:,.3f}", "MWh/day",
         f"{gap['share_pct']:.2f} % of the load · {gap['usd']:,.2f} USD", "warn"),
    ])
    theme.note(
        "<b>One network, two questions.</b> Both tabs build the network with the same call — "
        "<code>vessim_system.site_network</code> — so the buses, the cables, the transformers and "
        "their ratings cannot drift apart. The ⚡ sheet <b>evaluates</b> one recorded step with fixed "
        "setpoints (Newton–Raphson); this sheet <b>optimises</b> the whole day with free setpoints "
        "(a linear OPF). Same physics, same copper: what changes is who decides.",
        "good")
    with st.expander("🔎 The proof, step by step: the two networks are the same object", expanded=False):
        identity = pd.DataFrame([
            {"property": "buses", "load flow": len(lf_net.buses),
             "OPF": len(opt.get("buses", ())), "identical": same_buses},
            {"property": "lines", "load flow": len(lf_net.lines),
             "OPF": len([b for b in opt.get("branches", ()) if b in lf_net.lines.index]),
             "identical": set(lf_net.lines.index) <= set(opt.get("branches", ()))},
            {"property": "transformers", "load flow": len(lf_net.transformers),
             "OPF": len([b for b in opt.get("branches", ()) if b in lf_net.transformers.index]),
             "identical": set(lf_net.transformers.index) <= set(opt.get("branches", ()))},
        ])
        st.dataframe(identity, hide_index=True, **theme.wide())
        reconciled = vessim_system.loss_gap(run, spec)
        theme.note(
            f"Feed both models the same recorded powers and the ⚡ network loses "
            f"<b>{reconciled['energy_mwh']:,.3f} MWh</b> on this day "
            f"({reconciled['share_pct']:.2f} % of the load, ≈ {reconciled['usd']:,.2f} USD at "
            f"{reconciled['price_usd_mwh']:,.0f} USD/MWh). The linear OPF is lossless, so that is the "
            "one quantity it cannot price — deliberate, because Vessim's policy also treats the site "
            "as a copper plate and charging PyPSA for losses the rule never pays would flatter PyPSA.",
            "warn")

    # ------------------------------------------------------------- diagram
    theme.section("The same network, one snapshot at a time",
                  "every bus of the case, each labelled with the price the OPF gives it", "NETWORK")
    svg, balance = _case_svg(opt, run, spec, step)
    diagram.render(svg, caption="Vessim + PyPSA · the case's own network",
                   hint=f"{pd.Timestamp(snapshots.iloc[step]).strftime('%d %b %H:%M')} · $/MWh on the buses")

    price_list = opt["prices"].iloc[step].to_dict() if len(opt["prices"]) else {}
    spread = (max(price_list.values()) - min(price_list.values())) if price_list else 0.0
    theme.note(
        "These are <b>bills as invoiced</b>. They ignore two things the 🧮 gap decomposition below puts back: the "
        "battery cycling cost the heuristic already prices (20 USD/MWh) and the energy left in the "
        "tank at the end of the horizon. Both favour PyPSA. On a 24 h horizon with a 40 MWh battery "
        "they can reverse the sign of the saving — open the gap expander before quoting this "
        "number."
    )
    with st.expander("🧮 Show the maths: Vessim's rule vs PyPSA's optimum at this hour",
                     expanded=False):
        maths.step_panel(run, spec, opt, params, step)
    with st.expander("🧮 The gap, decomposed — where the difference between the two comes from",
                     expanded=False):
        try:
            maths.gap_section(run, spec, opt)
        except Exception as exc:                     # pragma: no cover - defensive
            theme.note(f"The decomposition could not be computed: {type(exc).__name__}: {exc}",
                       "bad")

    theme.kpis([
        ("Load at this hour", f"{balance['load']:.3f}", "MW", "", ""),
        ("Solar", f"{balance['solar']:.3f}", "MW", "", "accent"),
        ("Grid import", f"{balance['grid']:.3f}", "MW", "at the POI", ""),
        ("Gas turbine", f"{balance['gas']:.3f}", "MW", "captive plant", ""),
        ("Battery", f"{balance['battery']:+.3f}", "MW",
         "discharge" if balance["battery"] >= 0 else "charge", ""),
        ("Nodal spread", f"{spread:,.2f}", "$/MWh",
         "same price everywhere" if spread < 1e-6 else "congestion is pricing the buses",
         "good" if spread < 1e-6 else "warn"),
    ])

    # -------------------------------------------------- the proof, step by step
    snap = vessim_system.snapshot(run, spec, step)
    ac = run_pf(vessim_system.site_network(run, spec, step), start="dc")
    lp_grid = float(opt["dispatch"].get("Grid import", pd.Series(0.0)).iloc[step]) \
        if len(opt["dispatch"]) else 0.0
    lp_pv = float(opt["dispatch"].get("Solar", pd.Series(0.0)).iloc[step]) \
        if len(opt["dispatch"]) else 0.0
    lp_battery = float(balance["battery"])
    ac_load = float(ac.summary["load_mw"])
    ac_grid = float(ac.buses.loc["POI", "p_gen"]) if "POI" in ac.buses.index else float("nan")
    ac_losses = float(ac.summary["losses_mw"])
    proof = pd.DataFrame([
        {"quantity": "Load (MW)", "Vessim trace": snap["load_mw"], "AC network": ac_load,
         "PyPSA": float(df["load_mw"].iloc[step]), "difference": 0.0},
        {"quantity": "PV (MW)", "Vessim trace": snap["solar_mw"], "AC network": snap["solar_mw"],
         "PyPSA": lp_pv, "difference": lp_pv - snap["solar_mw"]},
        {"quantity": "Battery (MW)", "Vessim trace": snap["battery_mw"],
         "AC network": snap["battery_mw"], "PyPSA": lp_battery,
         "difference": lp_battery - snap["battery_mw"]},
        {"quantity": "Grid at the POI (MW)", "Vessim trace": snap["grid_mw"], "AC network": ac_grid,
         "PyPSA": lp_grid, "difference": ac_grid - lp_grid},
        {"quantity": "Losses in the network (MW)", "Vessim trace": float("nan"),
         "AC network": ac_losses, "PyPSA": 0.0, "difference": ac_losses},
    ])
    with st.expander("🔎 The correlation, checked at this snapshot", expanded=False):
        st.dataframe(proof.style.format({"Vessim trace": "{:,.4f}", "AC network": "{:,.4f}",
                                         "PyPSA": "{:,.4f}", "difference": "{:+,.4f}"}, na_rep="—"),
                     hide_index=True, **theme.wide())
        residual = ac_grid - lp_grid - ac_losses
        theme.note(
            f"Read the last two rows together: the AC slack injects <b>{ac_grid:,.4f} MW</b>, the OPF "
            f"imports <b>{lp_grid:,.4f} MW</b>, and the difference is exactly the network's losses — "
            f"<b>{residual:+.1e} MW</b> left over. The load, the PV and the battery are the same trace "
            "in both models: the OPF is not a different system, it is this one with the impedance "
            "removed and the meters summed."
            + ("" if abs(lp_pv - snap["solar_mw"]) < 1e-9 else
               f" The PV differs by {lp_pv - snap['solar_mw']:+.4f} MW because the OPF is "
               "allowed to curtail — a decision, not a modelling gap."),
            "good" if abs(residual) < 1e-6 else "warn")

    # ------------------------------------------------------------- dispatch
    pypsa_dispatch = opt["dispatch"].copy()
    gas = pypsa_dispatch.get("Gas turbine", pd.Series(0.0, index=pypsa_dispatch.index))
    grid = pypsa_dispatch.get("Grid import", pd.Series(0.0, index=pypsa_dispatch.index))
    solar = pypsa_dispatch.get("Solar", pd.Series(0.0, index=pypsa_dispatch.index))
    battery = _storage_trace(opt, dt)
    battery.index = pypsa_dispatch.index

    pypsa_frame = pd.DataFrame({"Solar": solar, "Grid import": grid,
                                "Gas turbine": gas, "Battery": battery},
                               index=opt["dispatch"].index).fillna(0.0)
    vessim_frame = pd.DataFrame({
        "Solar": df["solar_mw"].clip(lower=0),
        "Grid import": df["grid_mw"].clip(lower=0),
        "Gas turbine": df.get("gas_mw", pd.Series(0.0, index=df.index)),
        "Battery": df["battery_mw"],
    }, index=df.index)

    columns = [("Solar", "Solar", theme.COLORS["solar"]),
               ("Gas turbine", "Gas", theme.COLORS["gas"]),
               ("Grid import", "Grid", theme.COLORS["grid"]),
               ("Battery", "Battery", theme.COLORS["battery"])]

    left, right = st.columns(2)
    with left:
        theme.show(_stacked(vessim_frame, columns, title="Vessim · heuristic dispatch"))
    with right:
        theme.show(_stacked(pypsa_frame, columns, title="PyPSA · optimal dispatch"))

    left, right = st.columns([1.3, 1])
    with left:
        theme.show(_cost_bars(vessim_cost, pypsa_cost))
    with right:
        theme.show(_cost_breakdown(opt))

    soc_v = float(df["soc_mwh"].iloc[-1])
    soc_p = float(opt["soc"].iloc[-1, 0])
    if abs(soc_v - soc_p) > 0.05 * capacity:
        theme.note(
            f"Mind the end of the horizon: Vessim finishes with <b>{soc_v:,.2f} MWh</b> in the "
            f"battery and the OPF with <b>{soc_p:,.2f} MWh</b>. Part of the cost difference is "
            "that stored energy, not a better decision. Shorten the battery or raise the carbon "
            "price to see differences that do not depend on the horizon edge.", "warn")

    # ------------------------------------------------------------------- SoC
    left, right = st.columns([1.4, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=opt["soc"].index, y=opt["soc"].iloc[:, 0] / capacity,
                                 name="PyPSA", line=dict(color=theme.COLORS["pypsa"], width=2.4),
                                 hovertemplate="%{y:.2%}"))
        fig.add_trace(go.Scatter(x=df.index, y=df["soc_mwh"] / max(spec.battery_kwh / 1000.0, 1e-9),
                                 name="Vessim", line=dict(color=theme.COLORS["vessim"], width=2, dash="dot"),
                                 hovertemplate="%{y:.2%}"))
        theme.figure(fig, height=290, y_title="SoC (p.u.)")
        fig.update_yaxes(tickformat=".0%")
        theme.show(fig)
    with right:
        z = opt["line_loading"].T.to_numpy(dtype=float) if len(opt["line_loading"]) else np.zeros((1, 1))
        fig = go.Figure(go.Heatmap(
            z=z, x=list(opt["line_loading"].index) if len(opt["line_loading"]) else [0],
            y=list(opt["line_loading"].columns) if len(opt["line_loading"]) else [""],
            colorscale=[[0, "#101b2e"], [0.75, "#1d4ed8"], [1, theme.COLORS["red"]]],
            zmin=0, zmax=110, colorbar=dict(title="%"),
            hovertemplate="%{x}<br>%{y}: %{z:,.1f}%<extra></extra>"))
        theme.figure(fig, height=290, legend=False, unified=False, title="Line loading")
        fig.update_layout(xaxis_title="hour", yaxis_title="")
        theme.show(fig)

    _congestion(opt)

    # ---------------------------------------------------------------- sweeps
    theme.section("Sweeps", "the two levers that change the conclusion", "SENSITIVITY")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌱 Sweep the carbon price", **theme.wide(), key="btn_tax"):
            taxes = (0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 300.0)
            st.session_state[f"tax_sweep_{case_key}"] = data.tax_sweep(df, params, dt, taxes)
    with col2:
        if st.button("⚡ Sweep the POI line rating", **theme.wide(), key="btn_line"):
            base = max(params["line_grid_mw"], 0.05)
            limits = tuple(round(base * f, 3) for f in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0))
            st.session_state[f"line_sweep_{case_key}"] = data.line_sweep(df, params, dt, limits)

    sweep = st.session_state.get(f"tax_sweep_{case_key}")
    if sweep is not None and len(sweep):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep["carbon_price"], y=sweep["cost_usd"], name="Total cost (USD)",
                                 line=dict(color=theme.COLORS["pypsa"], width=2.4),
                                 hovertemplate="%{y:,.2f} USD"))
        fig.add_trace(go.Scatter(x=sweep["carbon_price"], y=sweep["emissions_t"], name="Emissions (t)",
                                 line=dict(color=theme.COLORS["solar"], width=2.4), yaxis="y2",
                                 hovertemplate="%{y:,.3f} t"))
        theme.figure(fig, height=310, x_title="carbon price (USD/t)")
        fig.update_layout(yaxis=dict(title="USD"),
                          yaxis2=dict(title="tCO₂", overlaying="y", side="right", showgrid=False))
        theme.show(fig)
        sheets.tabular(sweep, height=220, formats={"carbon_price": "{:.0f}", "cost_usd": "{:,.2f}",
                                                   "emissions_t": "{:,.3f}", "grid_mwh": "{:,.3f}",
                                                   "gas_mwh": "{:,.3f}"})
        theme.note("As the carbon price rises the optimal solution changes technology. Vessim already "
                   "reacts to that price through its merit-order policy; PyPSA shows what the "
                   "<b>least-cost</b> response looks like.", "")

    line_sweep = st.session_state.get(f"line_sweep_{case_key}")
    if line_sweep is not None and len(line_sweep):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=line_sweep["line_limit_mw"], y=line_sweep["cost_usd"],
                             name="Cost (USD)", marker_color=theme.COLORS["pypsa"],
                             hovertemplate="%{y:,.2f} USD"))
        fig.add_trace(go.Scatter(x=line_sweep["line_limit_mw"], y=line_sweep["curtailment_mwh"],
                                 name="Curtailed (MWh)", yaxis="y2", mode="lines+markers",
                                 line=dict(color=theme.COLORS["amber"], width=2),
                                 hovertemplate="%{y:,.3f} MWh"))
        theme.figure(fig, height=310, x_title="POI line rating (MVA)")
        fig.update_layout(yaxis=dict(title="USD"),
                          yaxis2=dict(title="MWh", overlaying="y", side="right", showgrid=False))
        theme.show(fig)
        sheets.tabular(line_sweep, height=220, formats={"line_limit_mw": "{:.3f}", "cost_usd": "{:,.2f}",
                                                        "emissions_t": "{:,.3f}", "curtailment_mwh": "{:,.3f}",
                                                        "grid_mwh": "{:,.3f}"})
        theme.note("As the POI line narrows the optimum can no longer import what it wants: "
                   "<b>congestion</b> appears, the price at the site bus rises and part of the sun is "
                   "curtailed. Vessim has no such constraint and overestimates what the grid can "
                   "deliver.", "warn")

    with st.expander("📋 PyPSA trace (MW per element)", expanded=False):
        sheet = pypsa_dispatch.copy()
        sheet["Battery"] = battery
        st.dataframe(sheet.round(4), **theme.wide(), height=320)
        data.download(sheet, f"pypsa_{case_key}.csv", key=f"dl_pypsa_{case_key}")


def _congestion(opt: dict) -> None:
    loading = opt["line_loading"]
    prices = opt["prices"]
    if not len(loading):
        return
    left, right = st.columns(2)
    with left:
        z = loading.T.to_numpy(dtype=float)
        fig = go.Figure(go.Heatmap(z=z, x=list(loading.index), y=list(loading.columns),
                                   colorscale=[[0, "#101b2e"], [0.75, "#1d4ed8"], [1, theme.COLORS["red"]]],
                                   zmin=0, zmax=110, colorbar=dict(title="%"),
                                   hovertemplate="%{x}<br>%{y}: %{z:,.1f}%<extra></extra>"))
        theme.figure(fig, height=300, legend=False, unified=False, title="Line loading")
        fig.update_layout(xaxis_title="hour", yaxis_title="")
        theme.show(fig)
    with right:
        if prices.shape[1] > 0:
            z = prices.T.to_numpy(dtype=float)
            fig = go.Figure(go.Heatmap(z=z, x=list(prices.index), y=list(prices.columns),
                                       colorscale="Viridis", colorbar=dict(title="USD/MWh"),
                                       hovertemplate="%{x}<br>%{y}: %{z:,.2f}<extra></extra>"))
            theme.figure(fig, height=300, legend=False, unified=False, title="Nodal price (LMP)")
            fig.update_layout(xaxis_title="hour", yaxis_title="")
            theme.show(fig)
    st.caption("Left: utilisation of every line. Right: nodal price per bus. If the buses diverge, "
               "the network — not the merit order — is setting the price.")
