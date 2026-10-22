"""Tab 2 — LoadFlow (AdjieBrotot), applied to the Vessim case itself.

The network is not a textbook system any more: it is the electrical envelope of
the microgrid that the case just simulated — the point of interconnection, the
step-down transformer, the photovoltaic collector, the battery converter and one
load per actor — loaded with the case's own powers at the snapshot you pick.
Newton-Raphson, the Y-bus and the single-line diagram are the original tool's;
the system is Vessim's.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import sld
from ..vessim_system import (
    actor_columns,
    layout_for,
    site_network,
    snapshot,
    system_for,
)
from . import data, diagram, maths, sheets, theme

def _installed(system) -> pd.DataFrame:
    """The case's equipment, as installed — this is what makes it *this* system."""
    return pd.DataFrame(system.equipment(), columns=["Kind", "Equipment", "Rating"])


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

def _slot(case_key: str, step: int) -> str:
    return f"lf::site::{case_key}::{step}"


def _payload(case_key: str, run: dict, spec, step: int) -> dict:
    slot = _slot(case_key, step)
    if slot not in st.session_state:
        st.session_state[slot] = data.net_payload(site_network(run, spec, step))
    return st.session_state[slot]


def _reset(case_key: str, run: dict, spec, step: int) -> None:
    slot = _slot(case_key, step)
    st.session_state[slot] = data.net_payload(site_network(run, spec, step))
    for part in data.PARTS:
        st.session_state.pop(f"lf::{slot}::{part}", None)


def _stressed(payload: dict, line: str | None, factor: float, tripped: list[str]) -> dict:
    stressed = copy.deepcopy(payload)
    lines = stressed["parts"]["lines"]
    if line and line in lines.index and factor != 1.0:
        lines.loc[line, "s_nom"] = float(lines.loc[line, "s_nom"]) * float(factor)
    for name in tripped:
        if name in lines.index:
            lines = lines.drop(index=name)
    stressed["parts"]["lines"] = lines
    return stressed


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _loss_pair(losses_mw: float) -> tuple[str, str]:
    """Losses as (value, unit): watts below 1 kW, kW below 1 MW, else MW."""
    watts = float(losses_mw) * 1e6
    if watts < 1000.0:
        return f"{watts:,.1f}", "W"
    if losses_mw < 1.0:
        return f"{losses_mw * 1e3:,.3f}", "kW"
    return f"{losses_mw:,.3f}", "MW"


def _voltage_figure(buses: pd.DataFrame, v_min: float, v_max: float) -> go.Figure:
    frame = buses.sort_values("v_pu")
    colours = [theme.COLORS["red"] if (v < v_min or v > v_max) else theme.COLORS["sky"]
               for v in frame["v_pu"]]
    figure = go.Figure(go.Bar(
        x=frame.index, y=frame["v_pu"], marker_color=colours,
        text=[f"{v:.3f}" for v in frame["v_pu"]], textposition="outside",
        hovertemplate="%{x}<br>V = %{y:.4f} p.u.<extra></extra>"))
    figure.add_hline(y=v_min, line_dash="dot", line_color=theme.COLORS["amber"],
                     annotation_text=f"v_min {v_min:.2f}", annotation_position="bottom left")
    figure.add_hline(y=v_max, line_dash="dot", line_color=theme.COLORS["amber"],
                     annotation_text=f"v_max {v_max:.2f}", annotation_position="top left")
    figure.add_hline(y=1.0, line_color="rgba(148,163,184,.35)")
    theme.figure(figure, height=310, legend=False, unified=False,
                 y_title="Voltage (p.u.)", x_title="")
    figure.update_yaxes(range=[max(0.0, float(frame["v_pu"].min()) - 0.05),
                               float(frame["v_pu"].max()) + 0.045])
    return figure


def _loading_figure(branches: pd.DataFrame) -> go.Figure:
    frame = branches.copy()
    frame["loading"] = frame["loading_pct"].astype(float)
    frame = frame.sort_values("loading")
    colours = [theme.COLORS["red"] if v > 100 else theme.COLORS["green"] for v in frame["loading"]]
    figure = go.Figure(go.Bar(x=frame["loading"], y=frame.index, orientation="h",
                              marker_color=colours,
                              text=[f"{v:.0f}%" for v in frame["loading"]],
                              textposition="outside"))
    figure.add_vline(x=100, line_dash="dash", line_color=theme.COLORS["red"],
                     annotation_text="rating", annotation_position="top")
    theme.figure(figure, height=310, legend=False, unified=False,
                 x_title="Loading (% of s_nom)", y_title="")
    return figure


def _ybus_figure(ybus, order) -> go.Figure:
    magnitude = np.abs(np.asarray(ybus, dtype=complex))
    figure = go.Figure(go.Heatmap(z=magnitude, x=list(order), y=list(order),
                                  colorscale="Blues", colorbar=dict(title="|Y| S")))
    theme.figure(figure, height=330, legend=False, unified=False)
    return figure


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #

PIPELINE = [
    ("1 · In", "one step of the Vessim trace, injected as fixed setpoints into the case's own network"),
    ("2 · Model", "Newton–Raphson on the Y-bus: voltages, angles, branch flows, losses"),
    ("3 · Out", "a solved snapshot — the slack equals the trace's import + the losses"),
    ("4 · Leaves for", "the Vessim + PyPSA sheet, which optimises on this same network"),
]


def render(case_key: str, run: dict, spec, carbon: float) -> None:
    theme.pipeline(PIPELINE)
    system = system_for(spec)
    theme.hero(
        f"⚡", f"Load flow on system {spec.number}",
        f"CASE {spec.number} · {spec.title.upper()} · {len(system.groups)} SWITCHBOARDS",
        f"This is the same Newton-Raphson solver, the same Y-bus and the same single-line diagram "
        f"as the AdjieBrotot tool, applied to <b>the system case {spec.number} installed</b> — "
        "not to a textbook network and not to a generic envelope: "
        + ", ".join(f"<b>{name}</b> ({rating})"
                    for kind, name, rating in system.equipment() if kind != "Switchboard")
        + ", with one meter per actor and the switchboards the case is wired on. Vessim does not "
        "model voltage and PyPSA's LP does not either; this tab is what the electrical layer adds.",
        [("System", f"{spec.number} · {spec.title}"), ("Switchboards", str(len(system.groups))),
         ("Method", "Newton-Raphson"), ("Source", "Vessim trace")],
    )
    with st.expander("🔧 What this case installs", expanded=theme.density() == "Full"):
        st.dataframe(_installed(system), hide_index=True, **theme.wide(), height=210)
        st.caption("Nameplates, not peaks: the ratings are what the case says it built; the next "
                   "slider picks the step whose *recorded* powers are frozen into the network.")

    df = run["df"]
    steps = len(df)
    net_load = (df["load_mw"] - df["solar_mw"]).to_numpy()
    peak = int(np.argmax(net_load + np.maximum(df["battery_mw"].to_numpy(), 0.0)))

    head1, head2, head3 = st.columns([2, 1, 1])
    with head1:
        step = st.slider("Snapshot (the step whose powers are frozen into the network)",
                         0, steps - 1, peak, key=f"lf_step_{case_key}",
                         format="%d")
    with head2:
        theme.spacer(28)
        if st.button("↺ Restore the case's network", **theme.wide(), key=f"lf_reset_{case_key}"):
            _reset(case_key, run, spec, int(step))
            st.rerun()
    with head3:
        theme.spacer(28)
        start = st.selectbox("Start", ["dc", "flat"], key=f"lf_start_{case_key}",
                             help="DC = linear seed (more robust); flat = flat voltage start")

    snap = snapshot(run, spec, int(step))
    layout = layout_for(spec)
    theme.kpis([
        ("Snapshot", f"{snap['time']:%H:%M}" if hasattr(snap["time"], "strftime") else str(step),
         "", f"step {snap['step']} of {steps}", ""),
        ("Site load", f"{snap['load_mw']:.3f}", "MW",
         f"{len(snap['actors'])} actors: " + ", ".join(list(snap["actors"])[:3]) +
         ("…" if len(snap["actors"]) > 3 else ""), ""),
        ("Solar", f"{snap['solar_mw']:.3f}", "MW", "must-take, unity power factor", "good"),
        ("Battery", f"{snap['battery_mw']:+.3f}", "MW",
         f"positive = discharging · SoC {snap['soc_mwh']:.2f} MWh", ""),
        ("Gas", f"{snap['gas_mw']:.3f}", "MW" if spec.gas_kw > 0 else "",
         "on-site turbine" if spec.gas_kw > 0 else "no turbine in this case", ""),
        ("Grid", f"{snap['grid_mw']:+.3f}", "MW",
         f"{snap['price_usd_mwh']:,.0f} USD/MWh · {snap['carbon_kg_mwh']:,.0f} kg/MWh", ""),
    ])

    theme.note(
        f"These injections are not a scenario: they are what <b>{run.get('policy_kind', 'the case policy')}"
        f"</b> decided at step {snap['step']} of the simulation, frozen into the network. Change the "
        "policy in the 🎛️ sheet and this diagram changes with it. "
        f"The envelope is stated, not hidden: <b>{layout.mv_kv:g}/{layout.site_kv:g} kV</b>, a "
        f"<b>{layout.tx_mva:g} MVA</b> transformer at the point of interconnection, "
        f"<b>{layout.cable_km:g} km</b> of collector cable, loads at power factor "
        f"{layout.power_factor:.2f}. Cable impedances are per kilometre <i>times</i> the length, "
        "by voltage class — the LV tails are resistive and lossy, the MV runs are reactance. "
        "Vessim simulates a copper plate (one bus, no voltage), so the reconstruction is the only "
        "place where voltage, loading and losses exist. Edit any cell below and the solution "
        "follows."
    )

    payload = _payload(case_key, run, spec, int(step))
    with st.expander("🗂️ Data sheets — the case's network, editable", expanded=False):
        payload = sheets.net_editor(payload, key_prefix=f"lf::{_slot(case_key, int(step))}")

    tables = payload["parts"]
    slack = tables["buses"]["type"].astype(str).str.lower().eq("slack").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tol = st.select_slider("Tolerance", options=[1e-4, 1e-6, 1e-8, 1e-10], value=1e-8,
                               format_func=lambda v: f"{v:.0e}", key=f"lf_tol_{case_key}")
    with c2:
        max_iter = st.slider("Max iterations", 3, 60, 30, key=f"lf_iter_{case_key}")
    with c3:
        v_min = st.slider("v_min (p.u.)", 0.90, 1.00, 0.95, 0.005, key=f"lf_vmin_{case_key}")
    with c4:
        v_max = st.slider("v_max (p.u.)", 1.00, 1.10, 1.05, 0.005, key=f"lf_vmax_{case_key}")

    if slack == 0:
        theme.note("Without a <b>Slack</b> bus there is no angular reference. Add one, or set a "
                   "generator's control to slack.", "bad")

    try:
        result = data.power_flow(payload, start, float(tol), int(max_iter))
    except Exception as exc:
        theme.note(f"<b>The case could not be solved:</b> {type(exc).__name__}: {exc}", "bad")
        return

    summary = result["summary"]
    net = data.net_from(payload)
    flows = sld.flows_from_pf(result)
    svg = sld.draw(net, flows, style=sld.Style(
        show_q=True, show_loading=True, show_loss=True, colour_by="voltage",
        v_min=v_min, v_max=v_max, title=f"{spec.title} · {snap['step']}",
        subtitle=f"Newton-Raphson {'converged' if result['converged'] else 'did not converge'}",
    ))
    diagram.render(svg, caption="Bus voltages, branch flows and loading — the case's own powers",
                   hint="thick bar = past a limit · arrows follow the active power")

    theme.kpis([
        ("Status", "Converged" if result["converged"] else "Did not converge", "",
         f"{result['iterations']} iterations", "good" if result["converged"] else "bad"),
        ("Max mismatch", f"{result['max_mismatch']:.2e}", "p.u.", f"tolerance {tol:.0e}",
         "" if result["converged"] else "bad"),
        ("P losses", *_loss_pair(summary["losses_mw"]),
         f"{summary.get('losses_pct', 0):.4f}% of generation", "warn"),
        ("Q losses", f"{summary['losses_mvar'] * 1000:,.2f}", "kVAr", "", ""),
        ("Min voltage", f"{_v_min_nonslack(result):.4f}", "p.u.",
         "lowest busbar other than the slack", "bad" if _v_min_nonslack(result) < v_min else "good"),
        ("Max branch loading", f"{summary['max_loading_pct']:.1f}", "%",
         "overloaded" if summary["max_loading_pct"] > 100 else "no overload",
         "bad" if summary["max_loading_pct"] > 100 else "good"),
    ])

    # ---------------------------------------------------------------- audit
    theme.section("The audit that ties the diagram to the simulation",
                  "the slack must reproduce Vessim's import — up to the losses", "AUDIT")
    slack_p = float(result["buses"]["p_gen"].reindex(["POI"]).fillna(0.0).iloc[0]) \
        if "POI" in result["buses"].index else float("nan")
    loss_mw = float(summary["losses_mw"])
    expected = snap["grid_mw"] + loss_mw
    theme.kpis([
        ("Slack injection", f"{slack_p:,.4f}", "MW", "solved by Newton-Raphson", ""),
        ("Vessim import", f"{snap['grid_mw']:,.4f}", "MW", "recorded by the Recorder", ""),
        ("Losses", f"{loss_mw * 1000:,.2f}", "kW", "the difference between the two", "warn"),
        ("Residual", f"{(slack_p - expected) * 1000:+.3f}", "kW",
         "zero means the network carries exactly the simulated powers",
         "good" if abs(slack_p - expected) < 1e-3 else "bad"),
    ])
    st.caption("Vessim's microgrid is a copper plate: it balances power, not voltage. Once the same "
               "powers are put on a network with impedance, the slack has to supply the load *and* "
               "the losses. That difference is the only number in the dashboard that Vessim cannot "
               "produce.")

    left, right = st.columns([1.15, 1])
    with left:
        theme.show(_voltage_figure(result["buses"], v_min, v_max))
    with right:
        theme.show(_loading_figure(result["branches"]))

    tab_bus, tab_branch, tab_conv, tab_ybus = st.tabs(["Buses", "Branches", "Convergence", "Y-bus"])
    with tab_bus:
        view = result["buses"][["v_nom", "type", "v_pu", "angle_deg", "p_gen", "q_gen",
                                "p_load", "q_load", "p_inj", "q_inj"]]
        sheets.tabular(view, height=300, formats={"v_pu": "{:.4f}", "angle_deg": "{:.3f}",
                                                  "p_gen": "{:.3f}", "q_gen": "{:.3f}",
                                                  "p_load": "{:.3f}", "q_load": "{:.3f}",
                                                  "p_inj": "{:.3f}", "q_inj": "{:.3f}"})
    with tab_branch:
        sheets.tabular(result["branches"], height=300,
                       formats={"p0": "{:.3f}", "q0": "{:.3f}", "p1": "{:.3f}", "q1": "{:.3f}",
                                "p_loss": "{:.4f}", "q_loss": "{:.4f}", "loading_pct": "{:.1f}"})
    with tab_conv:
        history = result["history"]
        figure = go.Figure(go.Scatter(x=history["iteration"], y=history["max_mismatch"],
                                      mode="lines+markers",
                                      line=dict(color=theme.COLORS["sky"], width=2),
                                      hovertemplate="iter %{x}: %{y:.3e}"))
        theme.figure(figure, height=280, legend=False, unified=False,
                     x_title="iteration", y_title="max |ΔP|, |ΔQ| (p.u.)")
        figure.update_yaxes(type="log")
        theme.show(figure)
        st.caption("Quadratic convergence: four or five iterations on this network, and the "
                   "mismatch drops by orders of magnitude each time.")
    with tab_ybus:
        theme.show(_ybus_figure(result["ybus"], result["bus_order"]))
        st.caption("Each cell is |Y_ij| in siemens: branch admittances on the diagonal, coupling "
                   "off it.")

    with st.expander("🧮 The maths of this tab — the load flow, with this snapshot's numbers",
                     expanded=False):
        maths.ac_sheet(net, result, snap, int(step))

    _stress(case_key, payload, spec, snap, v_min, v_max)
    _validation(payload, case_key)
    _teaching(spec, snap, layout)


def _v_min_nonslack(result: dict) -> float:
    buses = result["buses"]
    mask = ~buses["type"].astype(str).str.lower().eq("slack")
    return float(buses.loc[mask, "v_pu"].min()) if mask.any() else float(buses["v_pu"].min())


def _stress(case_key: str, payload: dict, spec, snap: dict, v_min: float, v_max: float) -> None:
    theme.section("Stress the case's own collector", "what an LP cannot see at all", "EXPERIMENT")
    lines = list(payload["parts"]["lines"].index)
    if not lines:
        return
    left, middle, right = st.columns([1.2, 1, 1.2])
    with left:
        line = st.selectbox("Cable", lines, index=0, key=f"lf_stress_line_{case_key}")
    with middle:
        factor = st.slider("Rating ×", 0.2, 1.5, 0.6, 0.05, key=f"lf_stress_factor_{case_key}")
    with right:
        tripped = st.multiselect("Take out of service", lines, default=[],
                                 key=f"lf_stress_trip_{case_key}")
    stressed = _stressed(payload, line, factor, tripped)
    try:
        result = data.power_flow(stressed, "dc", 1e-8, 40)
    except Exception as exc:
        theme.note(f"<b>No solution:</b> {type(exc).__name__}: {exc} — that is the answer: this "
                   "operating point is not reachable on this network.", "bad")
        return
    net = data.net_from(stressed)
    svg = sld.draw(net, sld.flows_from_pf(result), style=sld.Style(
        show_q=True, show_loading=True, show_loss=True, colour_by="voltage",
        v_min=v_min, v_max=v_max, title=f"{spec.title} under stress",
        subtitle=f"×{factor:.2f} on {line}" + (f" · out: {', '.join(tripped)}" if tripped else "")))
    diagram.render(svg, caption="The same snapshot, a weaker cable",
                   hint="the battery converter bus is the first to sag")
    summary = result["summary"]
    theme.kpis([
        ("Min voltage", f"{summary['min_v_pu']:.4f}", "p.u.",
         "limit " + f"{v_min:.2f}", "bad" if summary["min_v_pu"] < v_min else "good"),
        ("Max loading", f"{summary['max_loading_pct']:.1f}", "%", "of the stressed cable",
         "bad" if summary["max_loading_pct"] > 100 else ""),
        ("Losses", f"{summary['losses_mw'] * 1000:,.2f}", "kW",
         f"{summary.get('losses_pct', 0):.2f}% of generation", "warn"),
    ])
    st.caption("The optimisation in the PyPSA tab prices energy and carbon; it has no voltage and "
               "no cable rating unless you add one. Here the same snapshot fails a limit — and the "
               "battery converter bus is the one that sags first, because it is the end of the "
               "weakest cable.")


def _validation(payload: dict, case_key: str) -> None:
    theme.section("Cross-validation", "our Newton-Raphson against pypsa.Network.pf()", "AUDIT")
    check = data.power_flow_vs_pypsa(payload)
    if not check["ok"]:
        theme.note(f"PyPSA could not solve the same case: <code>{check['error']}</code>", "warn")
        return
    theme.kpis([
        ("Max ΔV", f"{check['max_dv']:.1e}", "p.u.", "same model, two engines", "good"),
        ("Max Δθ", f"{check['max_dangle']:.1e}", "°", "same model, two engines", "good"),
        ("Buses compared", f"{len(check['frame'])}", "", "one by one", ""),
    ])
    with st.expander("Bus-by-bus comparison", expanded=False):
        sheets.tabular(check["frame"], height=260,
                       formats={"v_ours": "{:.6f}", "v_pypsa": "{:.6f}", "dv": "{:.2e}",
                                "angle_ours": "{:.4f}", "angle_pypsa": "{:.4f}",
                                "dangle": "{:.2e}"})


def _teaching(spec, snap: dict, layout) -> None:
    theme.section("What this adds to the two other engines", "", "WHY")
    st.markdown(
        f"""
        | Question | Vessim | PyPSA (LP) | this tab |
        |---|---|---|---|
        | is there enough energy? | **yes**, exactly balanced every step | yes, by construction | yes — the slack closes it |
        | what does it cost? | **yes**, from the signals | yes, at the optimum | yes, from the case's own prices |
        | what is the voltage at the data centre? | no | no | **{snap['load_mw']:.2f} MW on a {layout.site_kv:g} kV bus** |
        | which cable is loaded, and how much? | no | only if you write the constraint | **yes, solved** |
        | how much energy is lost on the way? | no | only if lines are modelled | **{layout.cable_km:g} km of cable, solved** |
        """
    )
    st.caption("This is the honest division of labour: Vessim simulates *behaviour*, PyPSA "
               "optimises *decisions*, the load flow checks *feasibility*. A case that is optimal "
               "and still cooks the collector cable is a real outcome, and only this tab can see "
               "it.")
