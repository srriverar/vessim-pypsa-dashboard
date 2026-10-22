"""Policy lab: which rule is running, why it decided, and what other rules would do.

Three questions, in the order a reader asks them:

1. **Which policy is running in this example?** The class, its parameters, and
   its decision rule in words — plus, at any step, the two scores it sorts on
   and the numbers it actually saw.
2. **Why did it decide that?** The recorded decision log of the run, entry by
   entry, with the merit order of that step.
3. **What would another rule have done?** The same day, the same sun, the same
   prices, six other policies — and the regret of each one against the PyPSA
   optimum of the same day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import vessim as vs

from ..policies import (MeritOrderPolicy, POLICY_CATALOGUE, POLICY_ORDER, PolicySpec,
                        CurtailmentSink, FuelGenerator, build_policy, score_table)
from ..vessim_runner import CASES
from . import data, maths, theme

# --------------------------------------------------------------------------- #
# The policy that ran
# --------------------------------------------------------------------------- #

RULE_TEXT = {
    "DefaultDispatchPolicy": (
        "Allocate the deficit to the dispatchables **in the order they were declared**, one after "
        "the other until it is covered, and the surplus to the first one that can absorb it. No "
        "price, no carbon, no state of charge: the list order *is* the policy."
    ),
    "MeritOrderPolicy": (
        "Score every dispatchable in **USD/MWh with the carbon price folded in**, sort, and take "
        "the cheapest first until the deficit is covered (or the cheapest absorber first for a "
        "surplus). The score is recomputed at every step from the live grid signals, so the order "
        "can change during the day."
    ),
}


def _parameters(kind: str, state: dict, spec) -> pd.DataFrame:
    rows = []
    if kind == "MeritOrderPolicy":
        rows += [
            ("carbon_price_usd_per_t", "carbon price added to every fossil MWh", state.get("carbon_price_usd_per_t")),
            ("strategy", "cost | carbon_first | battery_first", state.get("strategy")),
            ("soc_target", "state of charge the battery is pushed towards", state.get("soc_target")),
            ("degradation_cost_usd_per_kwh", "what a discharged kWh costs (wear)", _policy_attr(spec, "degradation_cost", 0.02)),
            ("charge_bonus_usd_per_kwh", "discount for charging a nearly empty battery", _policy_attr(spec, "charge_bonus", 0.04)),
            ("export_price_factor", "share of the price paid on export", _policy_attr(spec, "export_price_factor", 0.35)),
        ]
    else:
        rows += [
            ("carbon_price_usd_per_t", "not used by this policy", "—"),
            ("strategy", "not used by this policy", "—"),
            ("soc_target", "not used by this policy", "—"),
            ("degradation_cost_usd_per_kwh", "not used by this policy", "—"),
            ("charge_bonus_usd_per_kwh", "not used by this policy", "—"),
            ("export_price_factor", "not used by this policy", "—"),
        ]
    frame = pd.DataFrame(rows, columns=["Parameter", "What it does", "Value"])
    # a mixed column of numbers and "—" cannot go through Arrow: show text
    frame["Value"] = [f"{v:,.3f}" if isinstance(v, (int, float)) else str(v) for v in frame["Value"]]
    return frame


def _policy_attr(spec, name: str, default):
    return getattr(spec, name, default) if spec is not None else default


def _dispatchables(spec, soc_fraction: float, policy_spec: PolicySpec | None = None):
    """Rebuild the dispatchable list of ``run_case`` for one step, for the scores.

    Nothing is simulated: the objects are constructed with the state the
    recorder reports at that instant, so the scores are the ones the policy
    would compute right now.
    """
    battery = vs.SimpleBattery("battery", capacity=max(spec.battery_kwh, 1e-9) * 1000,
                               initial_soc=float(np.clip(soc_fraction, 0.0, 1.0)),
                               c_rate=spec.battery_kw / max(spec.battery_kwh, 1e-9))
    items = []
    if policy_spec is not None and policy_spec.kind == "peak_shaving":
        items.append(vs.GridConnection("base_import", max_import=policy_spec.peak_cap_kw * 1000,
                                       max_export=0.0))
        items.append(battery)
        if spec.gas_kw > 0:
            items.append(FuelGenerator("gas", p_max=spec.gas_kw * 1000,
                                       fuel_cost=spec.gas_cost_usd_kwh,
                                       co2_kg_per_kwh=spec.gas_co2))
        items.append(vs.GridConnection("overflow", max_import=spec.grid_import_kw * 1000))
        items.append(CurtailmentSink("curtailment"))
        return items
    items.append(battery)
    if spec.gas_kw > 0:
        items.append(FuelGenerator("gas", p_max=spec.gas_kw * 1000, fuel_cost=spec.gas_cost_usd_kwh,
                                   co2_kg_per_kwh=spec.gas_co2))
    items.append(vs.GridConnection("grid", max_import=spec.grid_import_kw * 1000,
                                   max_export=spec.grid_export_kw * 1000))
    items.append(CurtailmentSink("curtailment"))
    return items


def current_policy_block(run: dict, spec, case_key: str, carbon: float, policy_spec=None) -> None:
    kind = run.get("policy_kind") or ("MeritOrderPolicy" if spec.policy == "merit"
                                      else "DefaultDispatchPolicy")
    state = run.get("policy_state") or {}
    declared = spec.policy

    left, right = st.columns([1.05, 1.0])
    with left:
        theme.section("The rule that actually ran", "one class, a handful of parameters", "POLICY")
        st.markdown(
            f"<div class='note'><b><code>vessim_pypsa_dashboard.policies.{kind}</code></b><br>"
            f"{RULE_TEXT.get(kind, '')}<br><br>"
            f"The case declares <code>policy=\"{declared}\"</code>"
            + (" — i.e. Vessim's own default." if declared == "default" else
               " — the merit-order policy from this repository.")
            + "</div>", unsafe_allow_html=True)
    with right:
        st.dataframe(_parameters(kind, state, spec), hide_index=True, **theme.wide(), height=250)

    df = run["df"]
    step = st.slider("Step to explain", 0, len(df) - 1, min(int(len(df) * 0.55), len(df) - 1),
                     key=f"pol_explain_{case_key}")
    row = df.iloc[step]
    soc = float(row.get("soc_mwh", 0.0)) / max(spec.battery_kwh / 1000.0, 1e-12)
    policy = build_policy(policy_spec or POLICY_CATALOGUE["official"], spec,
                          carbon_price=carbon, default_price=spec.price_base,
                          default_carbon=spec.carbon_base)
    items = _dispatchables(spec, soc, policy_spec)
    table = pd.DataFrame(score_table(policy, items, float(row["price_usd_mwh"]) / 1000.0,
                                     float(row["carbon_kg_mwh"]) / 1000.0,
                                     duration=spec.step_s))

    theme.section("Why it decided that", "the numbers the rule compares at this step", "SCORES")
    st.markdown(
        f"<div class='small'>At {pd.Timestamp(df.index[step]).strftime('%d %b %H:%M')} the grid "
        f"signal carried <b>{row['price_usd_mwh']:,.1f} USD/MWh</b> and "
        f"<b>{row['carbon_kg_mwh']:,.0f} kgCO₂/MWh</b>; the battery was at "
        f"<b>{100 * soc:,.1f} %</b>. The table below is computed from those three numbers, not "
        f"replayed from the recording.</div>", unsafe_allow_html=True)
    view = table.rename(columns={
        "order": "#", "name": "Dispatchable", "kind": "Class",
        "can_supply_mw": "Can supply (MW)", "can_absorb_mw": "Can absorb (MW)",
        "supply_score": "Supply score (USD/MWh)", "absorb_score": "Absorb score (USD/MWh)",
        "soc": "SoC",
    })
    st.dataframe(
        view.style.format({"Can supply (MW)": "{:,.4f}", "Can absorb (MW)": "{:,.4f}",
                           "Supply score (USD/MWh)": "{:,.2f}", "Absorb score (USD/MWh)": "{:,.2f}",
                           "SoC": "{:.1%}"}, na_rep="—"),
        hide_index=True, **theme.wide(), height=210)

    if isinstance(policy, MeritOrderPolicy):
        suppliers = table.dropna(subset=["supply_score"]).sort_values("supply_score")
        cheapest = suppliers.iloc[0]
        theme.note(
            f"The deficit of this step is covered cheapest-first: <b>{cheapest['name']}</b> at "
            f"<b>{cheapest['supply_score']:,.2f} USD/MWh</b> (carbon included), then the next one, "
            f"and so on until the balance closes. The battery's score moves with its state of "
            f"charge on purpose: at {100 * soc:,.0f} % it is cheap to use, near empty it becomes "
            f"expensive, which is how a one-step rule imitates a storage strategy."
        )
    else:
        theme.note(
            "Vessim's default policy has no scores: it walks the list <b>in declaration order</b> "
            "(battery, gas, grid, spillway) and stops when the balance closes. That is why case 01 "
            "charges the battery at midnight and why case 05 burns gas it did not need to burn — "
            "the list never looks at the signal."
        )

    # the logged decision, if the policy kept one
    log = run.get("policy_log") or []
    if log:
        entry = log[min(step, len(log) - 1)]
        powers = entry.get("powers", {})
        left, right = st.columns([1, 1])
        with left:
            theme.section("What it did", "the recorded decision, in MW", "LOG")
            frame = pd.DataFrame(
                [{"Dispatchable": name, "Power (MW)": float(value) / 1e6}
                 for name, value in powers.items()])
            st.dataframe(frame.style.format({"Power (MW)": "{:+,.4f}"}), hide_index=True,
                         **theme.wide(), height=190)
        with right:
            theme.section("What it saw", "the inputs of that decision", "INPUT")
            st.dataframe(pd.DataFrame([
                {"Input": "net deficit p_delta", "Value": f"{float(entry.get('p_delta', 0.0)) / 1e6:+,.4f} MW"},
                {"Input": "price signal", "Value": f"{1000 * float(entry.get('price', 0.0)):,.2f} USD/MWh"},
                {"Input": "carbon signal", "Value": f"{1000 * float(entry.get('carbon', 0.0)):,.0f} kgCO₂/MWh"},
            ]), hide_index=True, **theme.wide(), height=190)
        theme.note(
            "A positive <code>p_delta</code> would be a surplus (surplus handling), a negative one is "
            "a deficit (supply). Vessim's sign convention is consumption-positive, so the powers "
            "above are the raw dispatchable setpoints in W, divided by 10⁶."
        )


# --------------------------------------------------------------------------- #
# Several policies on the same day
# --------------------------------------------------------------------------- #

def _optimum(run: dict, spec, carbon: float) -> dict:
    params = maths.default_params(run["spec"] if "spec" in run else spec, carbon)
    return data.site_case(run["df"], params, run["dt"])


def _regret(run_like: dict, spec, opt: dict) -> dict:
    _, info = maths.gap_table(run_like, spec, opt)
    return info


def scenario_block(case_key: str, run: dict, spec, carbon: float,
                   solar_scale: float = 1.0, load_scale: float = 1.0) -> None:
    theme.section("The same day under six other rules",
                  "identical sun, load, prices and carbon — only the decision rule changes",
                  "SCENARIOS")
    st.markdown(
        "<div class='small'>Each row is a full Vessim run of this case. The benchmark is the PyPSA "
        "optimum of the same trace, closed-book: battery discharge priced at the same 20 USD/MWh "
        "and the energy left in the tank marked to market. <b>Regret</b> is what the rule leaves on "
        "the table, in USD and in percent of its own bill. To change the rule itself — the contract, "
        "the eight shipped classes and an editor that runs your own — see "
        "<b>🕶️ Blind spots & policies</b> → *How a policy is written*.</div>", unsafe_allow_html=True)

    keys = [key for key in POLICY_ORDER if key != "peak_shaving" or case_key == "solar_battery"]
    chosen = st.multiselect("Policies to run", keys,
                            default=[k for k in ("default", "merit_cost", "merit_carbon") if k in keys],
                            format_func=lambda key: POLICY_CATALOGUE[key].label,
                            key=f"pol_matrix_{case_key}")
    if not chosen:
        theme.note("Pick at least one policy to run the comparison.", "warn")
        return
    if st.button("▶ Run the policy matrix", type="primary", key=f"pol_run_{case_key}"):
        specs = tuple(POLICY_CATALOGUE[key] for key in chosen)
        with st.spinner(f"Running {len(specs)} policies over {spec.steps} steps…"):
            st.session_state[f"pol_result_{case_key}"] = {
                "specs": specs,
                "result": data.policy_matrix(case_key, float(carbon), float(solar_scale),
                                             float(load_scale), specs),
            }

    payload = st.session_state.get(f"pol_result_{case_key}")
    if not payload:
        st.markdown("<div class='small'>Press the button: this runs one complete Vessim simulation "
                    "per policy, then prices each one against the same optimum.</div>",
                    unsafe_allow_html=True)
        return

    opt = _optimum(run, spec, carbon)
    rows = []
    for policy_spec in payload["specs"]:
        result = payload["result"][policy_spec.key]
        baseline = result["baseline"]
        run_like = {"df": result["df"], "dt": result["dt"], "spec": spec, "baseline": baseline}
        info = _regret(run_like, spec, opt)
        rows.append({
            "Policy": policy_spec.label,
            "Class": result["policy_kind"].replace("DispatchPolicy", "").replace("Policy", ""),
            "Bill (USD)": info["vessim_bill"],
            "Closed-book cost (USD)": info["vessim_cost"] + info["terminal_term"],
            "Regret vs optimum (USD)": info["gap"],
            "Regret (%)": 100.0 * info["gap"] / max(info["vessim_bill"], 1e-9),
            "CO₂ (kg)": baseline["emissions_t"] * 1000.0,
            "Grid (MWh)": baseline["grid_mwh"],
            "Gas (MWh)": baseline["gas_mwh"],
            "Battery discharge (MWh)": info["discharge_v"],
            "Spilled sun (MWh)": baseline["curtailment_mwh"],
        })
    frame = pd.DataFrame(rows).sort_values("Regret vs optimum (USD)")
    st.dataframe(
        frame.style.format({"Bill (USD)": "{:,.2f}", "Closed-book cost (USD)": "{:,.2f}",
                            "Regret vs optimum (USD)": "{:+,.2f}",
                            "Regret (%)": "{:+.2f} %", "CO₂ (kg)": "{:,.1f}", "Grid (MWh)": "{:,.2f}",
                            "Gas (MWh)": "{:,.2f}", "Battery discharge (MWh)": "{:,.2f}",
                            "Spilled sun (MWh)": "{:,.2f}"})
        .background_gradient(subset=["Regret vs optimum (USD)"], cmap="RdYlGn_r"),
        hide_index=True, **theme.wide())

    best = frame.iloc[0]
    worst = frame.iloc[-1]
    greenest = frame.loc[frame["CO₂ (kg)"].idxmin()]
    optimum_cost = float(opt["energy_cost"] + opt["carbon_cost"])
    theme.kpis([
        ("Cheapest rule", f"{best['Policy'][:24]}", "",
         f"{best['Regret vs optimum (USD)']:+,.2f} USD vs the optimum",
         "good" if best["Regret vs optimum (USD)"] <= 1e-6 else ""),
        ("Most expensive", f"{worst['Policy'][:24]}", "",
         f"{worst['Regret vs optimum (USD)']:+,.2f} USD", "bad"),
        ("Greenest rule", f"{greenest['Policy'][:24]}", "",
         f"{greenest['CO₂ (kg)']:,.0f} kgCO₂", "good"),
        ("PyPSA optimum", f"{optimum_cost:,.2f}", "USD",
         f"it emits {opt['emissions_t'] * 1000:,.0f} kg", "good"),
    ])
    theme.note(
        "Want a different rule instead of a different row? <b>🕶️ Blind spots & policies</b> "
        "shows the <code>DispatchPolicy.apply()</code> contract, the one line where the policy plugs "
        "into <code>add_microgrid</code>, a playground with six sliders, and a complete custom policy."
    )
    theme.note(
        f"The cheapest rule and the greenest rule are <b>not the same rule</b>: "
        f"“{best['Policy']}” spends {best['Bill (USD)']:,.0f} USD and emits "
        f"{best['CO₂ (kg)']:,.0f} kg, while “{greenest['Policy']}” spends "
        f"{greenest['Bill (USD)']:,.0f} USD and emits {greenest['CO₂ (kg)']:,.0f} kg. That trade-off "
        f"is a menu, not a ranking — and the menu is what a carbon price is supposed to move."
    )

    # regret bars
    fig = go.Figure(go.Bar(
        x=frame["Policy"], y=frame["Regret vs optimum (USD)"],
        marker_color=[theme.COLORS["green"] if v <= 1e-6 else theme.COLORS["amber"]
                      if v > 0 else theme.COLORS["red"] for v in frame["Regret vs optimum (USD)"]],
        text=[f"{v:+,.2f}" for v in frame["Regret vs optimum (USD)"]], textposition="outside",
        hovertemplate="%{x}<br>regret %{y:+,.2f} USD<extra></extra>"))
    theme.figure(fig, height=320, legend=False, x_title="", y_title="regret vs the optimum (USD)")
    theme.note(
        "A <b>negative</b> regret means the rule beat the optimum on the closed book — which can only "
        "happen when the two are solving slightly different problems (a hard import cap, for "
        "instance, or a horizon artefact). It is a signal to check the protocol, not a discovery."
    )

    # dispatch comparison for two rules
    left, right = st.columns(2)
    options = [p.key for p in payload["specs"]]
    with left:
        first = st.selectbox("Compare rule A", options, index=0,
                             format_func=lambda k: POLICY_CATALOGUE[k].label,
                             key=f"pol_a_{case_key}")
    with right:
        second = st.selectbox("Compare rule B", options, index=min(1, len(options) - 1),
                              format_func=lambda k: POLICY_CATALOGUE[k].label,
                              key=f"pol_b_{case_key}")
    _dispatch_figure(payload["result"][first]["df"], payload["result"][second]["df"],
                     POLICY_CATALOGUE[first].label, POLICY_CATALOGUE[second].label, case_key)

    st.markdown(
        "<div class='note'>Read the table as an experiment, not a leaderboard: the rows differ in "
        "<i>one</i> thing (the policy), and the benchmark is the same closed-book optimum. That is "
        "the design the decision guide asks for before any claim of the form “policy X is better”."
        "</div>", unsafe_allow_html=True)


def _dispatch_figure(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str,
                     case_key: str) -> None:
    fig = go.Figure()
    for df, label, dash in ((df_a, label_a, None), (df_b, label_b, "dot")):
        fig.add_trace(go.Scatter(x=df.index, y=df["battery_mw"], name=f"battery · {label}",
                                 line=dict(color=theme.COLORS["battery"], width=2, dash=dash),
                                 hovertemplate="%{y:.3f} MW<extra></extra>"))
        fig.add_trace(go.Scatter(x=df.index, y=df["grid_mw"], name=f"grid · {label}",
                                 line=dict(color=theme.COLORS["grid"], width=1.6, dash=dash),
                                 hovertemplate="%{y:.3f} MW<extra></extra>"))
    theme.figure(fig, height=330, x_title="", y_title="MW")
    st.caption("Positive battery = discharge, negative = charge. Grid positive = import. "
               "Same axes, same day: the difference is the rule.")
