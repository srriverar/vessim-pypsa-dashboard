"""What PyPSA cannot see — and Vessim can.

An optimisation model is a *description* of the plant: a set of linear
constraints, a horizon, and an objective.  Everything that does not fit in that
description does not exist for the solver, however real it is:

* the fact that the controller is software, with CPU time, deadlines and bugs;
* the information it has at the moment it acts (which is not tomorrow's price);
* the device's efficiency curve, its derating, its discrete states;
* the rules that are not prices — a reserve floor, a contract, a compliance
  budget.

This page states each blind spot, shows the mechanism Vessim uses to represent
it, and **measures** it on the data of the case you pick: the cost of not
knowing, the energy a rule can move, the losses a constant efficiency hides.
The second half is the constructive one: a dispatch policy in Vessim is a class
with one method, and changing it is three lines.  The page lets you write one
and run it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..policies import MeritOrderPolicy, PolicySpec
from ..vessim_runner import CASES, CASE_ORDER
from . import data, maths, theme

# --------------------------------------------------------------------------- #
# The map
# --------------------------------------------------------------------------- #

BOUNDARY = [
    ("The future",
     "knows the whole horizon at once — perfect foresight is not an assumption, it is the input",
     "the controller is called at ``now`` and sees the signals of ``now``; the run is causal",
     "a plan that needs tomorrow is not a controller, it is a schedule"),
    ("The software",
     "a load is ``p_set``: a number per snapshot, fixed, with no reason behind it",
     "the load is produced by running code — threads, CPU time, deadlines, sandboxing",
     "if the workload can move or shrink, the profile the LP receives is already the answer to a "
     "question it never asked"),
    ("The device",
     "a storage unit with constant efficiency, continuous power, free terminal state",
     "``SimpleBattery`` with its own SoC and C-rate, or a device you write yourself",
     "the datasheet has a curve, a derating and a warranty; the LP has a scalar"),
    ("The rule",
     "a constraint you remember to add, or a price you remember to put in the objective",
     "any ``DispatchPolicy.apply()`` you can write, including rules no price can express",
     "an unmodelled constraint does not exist for the solver, but it does for the plant"),
    ("The failure",
     "solves a relaxation: if it is feasible, the plan is feasible; nothing else is checked",
     "every step must balance exactly, by construction, or the run raises",
     "Vessim certifies that the trajectory is *implementable step by step*; the LP certifies "
     "optimality of a model"),
    ("The coordination",
     "one central solver with all information from all sites at once",
     "geo-distributed actors and controllers, each with its own local view and latency",
     "a central optimum is an upper bound for a system that must coordinate over a network"),
]

MIRROR = [
    ("Network and losses", "no power flow at all in the microgrid model: one copper plate",
     "an OPF on the case's own buses, with line limits and nodal prices (🔗 tab)"),
    ("Price formation", "the policy reacts to a price it is given; it never *forms* one",
     "λ per bus and per hour, the dual of the balance, checkable by finite differences"),
    ("The best possible", "it can say what happened, never what should have happened",
     "the optimum, its certificate and the regret of every rule"),
    ("Feasibility proof", "it raises when the step cannot be balanced",
     "a solved LP proves that *some* schedule existed, and how good it was"),
]


def _section_map() -> None:
    theme.section("The boundary", "what each engine can and cannot represent", "MAP")
    st.dataframe(
        pd.DataFrame(BOUNDARY, columns=["What", "PyPSA · the model", "Vessim · the simulation",
                                        "Why the difference matters"]),
        hide_index=True, **theme.wide(), height=430)
    theme.note(
        "None of the rows on the left is a defect. An LP is <i>supposed</i> to be a model: that is "
        "why it has duals, certificates and can scale. The blind spots matter only when someone "
        "quotes the LP's plan as if it were a controller's behaviour — or the controller's bill as "
        "if it were a bound."
    )

    theme.section("And the other direction", "what Vessim cannot see, for the same reason", "MIRROR")
    st.dataframe(pd.DataFrame(MIRROR, columns=["What", "Vessim cannot see", "PyPSA does"]),
                 hide_index=True, **theme.wide(), height=250)


# --------------------------------------------------------------------------- #
# Probe 1 · the value of knowing the future
# --------------------------------------------------------------------------- #

def _probe_foresight(case_key: str, spec, carbon: float, solar: float, load: float,
                     opt: dict, policy_kind: str = "") -> None:
    st.markdown(
        "<div class='small'>PyPSA's LP is handed the whole day: prices, sun, load, carbon. The "
        "controller is not. This probe keeps the day identical and multiplies the price and carbon "
        "that <i>the policy sees</i> by <code>1 + σ·N(0,1)</code>; the bill is settled at the true "
        "prices, so every extra dollar is the price of the missing information. "
        "<b>σ = 0 is PyPSA's world</b>, and only PyPSA can live in it.</div>",
        unsafe_allow_html=True)

    if policy_kind == "DefaultDispatchPolicy":
        theme.note(
            "This case runs Vessim's <code>DefaultDispatchPolicy</code>, which never reads the price "
            "or the carbon signal — so σ cannot change anything, and it does not. A rule that "
            "ignores information cannot lose information; open <b>🎛️ Policies</b> and run it with a "
            "merit-order policy to see the cost appear."
        )

    sigmas = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
    sweep = data.noise_sweep(case_key, carbon, solar, load, sigmas, (1, 2, 3))
    stats = (sweep.groupby("sigma")
             .agg(bill=("bill", "mean"), bill_lo=("bill", "min"), bill_hi=("bill", "max"),
                  co2=("co2_t", "mean"), gas=("gas_mwh", "mean"))
             .reset_index())
    base = float(stats.loc[stats["sigma"] == 0.0, "bill"].iloc[0])
    chosen = st.select_slider("How wrong is the controller's signal?",
                              options=list(sigmas), value=0.15,
                              format_func=lambda s: f"{s:.0%}",
                              key=f"blind_sigma_{case_key}")
    row = stats.loc[stats["sigma"] == chosen].iloc[0]
    lp_bill = float(opt["energy_cost"] + opt["carbon_cost"])
    gap = float(base - lp_bill)

    theme.kpis([
        ("Perfect information", f"{base:,.2f}", "USD", "σ = 0, PyPSA's assumption", ""),
        (f"σ = {chosen:.0%}", f"{row['bill']:,.2f}", "USD",
         f"{row['bill'] - base:+,.2f} USD for the same day", "bad" if chosen else ""),
        ("Taken out of the gap",
         f"{(row['bill'] - base) / gap:.0%}" if abs(gap) > 0.05 else "—", "",
         f"the LP's own advantage is {gap:,.2f} USD", ""),
        ("CO₂", f"{row['co2'] * 1000:,.1f}", "kg",
         f"{(row['co2'] - stats.loc[0, 'co2']) * 1000:+,.1f} kg vs perfect information", ""),
    ])

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=stats["sigma"], y=stats["bill_hi"], mode="lines",
                                line=dict(width=0), showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=stats["sigma"], y=stats["bill_lo"], mode="lines",
                                line=dict(width=0), fill="tonexty",
                                fillcolor="rgba(56,189,248,.16)", showlegend=False,
                                hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=stats["sigma"], y=stats["bill"], mode="lines+markers",
                                name="mean over three seeds",
                                line=dict(color="#38bdf8", width=3)))
    figure.add_hline(y=lp_bill, line=dict(color="#34d399", dash="dash", width=2),
                     annotation_text=f"PyPSA optimum {lp_bill:,.0f}",
                     annotation_position="bottom right")
    figure.add_vline(x=chosen, line=dict(color="#fbbf24", dash="dot", width=2))
    theme.show(theme.figure(figure, height=330, x_title="σ · error on the signal the controller sees",
                            y_title="bill for the same day (USD)"))
    theme.note(
        f"At σ = 30 % the controller pays <b>{stats.loc[stats['sigma'] == 0.30, 'bill'].iloc[0] - base:+,.2f} "
        f"USD</b> for the same physical day — against an LP advantage of <b>{gap:,.2f} USD</b>. "
        "Read that twice: a large part of what is usually called <i>the value of optimisation</i> "
        "is in fact the value of <i>knowing the day</i>. PyPSA cannot separate the two, because it "
        "sees the day; Vessim can, precisely because it does not."
    )


# --------------------------------------------------------------------------- #
# Probe 2 · the software is a program
# --------------------------------------------------------------------------- #

def _probe_software(spec) -> None:
    st.markdown(
        "<div class='small'>PyPSA receives this row as <code>p_set</code>: "
        "<code>loads_t.p_set['Site load'] = df['load_mw']</code>. It cannot know whether the "
        "number is a factory, a heater or a program that decided to wait. Vessim runs the program."
        "</div>", unsafe_allow_html=True)

    probe = data.sil_probe(steps=6, step_s=0.4)
    samples = probe["samples"]
    cpu = float(samples["cpu_seconds"].mean())
    wall = float(samples["wall_seconds"].mean())
    watts = float(samples["watts"].mean())
    theme.kpis([
        ("CPU per step", f"{cpu:.3f}", "s", f"requested {probe['step_s']:.2f} s", ""),
        ("Wall per step", f"{wall:.3f}", "s", "the simulation was paced by it", ""),
        ("Draw", f"{watts:,.0f}", "W", "PUE · (base + core·CPU/step)", "good"),
        ("Load the LP sees", "1 number", "", "per snapshot, with no reason behind it", ""),
    ])
    st.caption("Measured now, in this container, by `sil.py` (`os.wait4` on a real child process). "
               "Nothing here is sampled from a table.")

    # the deferral the software performs before PyPSA ever sees the trace
    carbon_c = 120.0
    run = data.vessim_case("carbon_aware", carbon_c, 1.0, 1.0)
    case = CASES["carbon_aware"]
    dt = run["dt"]
    load_kw = run["df"]["load_mw"].to_numpy() * 1000.0
    intensity = run["df"]["carbon_kg_mwh"].to_numpy() / 1000.0
    flat_kw = case.load_kw
    actual_kwh = float(load_kw.sum() * dt)
    flat_kwh = float(flat_kw * len(load_kw) * dt)
    actual_kg = float((load_kw * intensity * dt).sum())
    flat_kg = float((flat_kw * intensity * dt).sum())
    avoided = flat_kg - actual_kg

    st.markdown(
        "<div class='small'>Case 03 is the clean example: 35 % of the load is flexible and the "
        "workload moves itself out of the dirty hours (a <code>DeferrableLoad</code> signal, coded, "
        "not optimised). The LP is then handed the <i>result</i>.</div>",
        unsafe_allow_html=True)
    theme.kpis([
        ("Work delivered", f"{actual_kwh:,.1f}", "kWh",
         f"flat reference {flat_kwh:,.1f} kWh — identical, by construction", "good"),
        ("CO₂ with the rule", f"{actual_kg:,.2f}", "kg",
         f"running flat: {flat_kg:,.2f} kg", ""),
        ("Avoided by the software", f"{avoided:,.2f}", "kg",
         f"{avoided / 1000 * carbon_c:,.2f} USD at τ = {carbon_c:.0f}", "good"),
        ("Who moved it", "the code", "", "PyPSA got the shifted profile as an input", ""),
    ])
    st.dataframe(pd.DataFrame({
        "hour": [f"{h:02d}:00" for h in range(24)][::3],
        "carbon (kg/kWh)": np.round(intensity.reshape(-1)[::12], 3),
        "load with the rule (kW)": np.round(load_kw.reshape(-1)[::12], 2),
        "flat workload (kW)": np.round(np.full(8, flat_kw), 2),
    }), hide_index=True, **theme.wide(), height=250)
    theme.note(
        "This is the blind spot in its purest form. The LP optimises the <i>plant</i> for a load it "
        "was given; whether that load could have run at 03:00 instead of 19:00 is a question about "
        "software, and it was already answered before the model started. A controller that shifts "
        "its own demand is doing optimisation too — just not in a solver."
    )


# --------------------------------------------------------------------------- #
# Probe 3 · the device is not a scalar
# --------------------------------------------------------------------------- #

def efficiency_curve(p_mw, p_rating_mw: float, eta0: float = 0.95, k: float = 0.10):
    """A real inverter's efficiency falls with load: η(x) = η₀ − k·x², x = p/P."""
    x = np.clip(np.abs(np.asarray(p_mw, dtype=float)) / max(float(p_rating_mw), 1e-12), 0.0, 1.2)
    return eta0 - k * x ** 2


def linearity_gap(opt: dict, dt: float, p_rating_mw: float, energy_mwh: float,
                  eta0: float = 0.95, k: float = 0.10) -> dict:
    """Energy the constant-efficiency model forgets, on the LP's own plan.

    Discharging ``p`` at AC draws ``p/η(p)`` from the store; charging stores
    ``|p|·η(p)``.  With a constant η₀ the LP books a different amount than the
    device will.  No re-dispatch is performed: this is the accounting error of
    the linearisation, a lower bound on what the curve costs.
    """
    p = np.asarray(opt["battery_mw"], dtype=float)
    eta = efficiency_curve(p, p_rating_mw, eta0, k)
    discharge = np.clip(p, 0.0, None)
    charge = np.clip(-p, 0.0, None)
    extra_store = (discharge * (1.0 / eta - 1.0 / eta0)
                   + charge * (eta0 - eta)).sum() * dt
    return {
        "extra_store_mwh": float(extra_store),
        "plan_peak_c": float(np.abs(p).max() / max(energy_mwh, 1e-12)),
        "eta_min": float(eta.min()),
        "throughput_mwh": float(np.abs(p).sum() * dt),
    }


def _probe_device(spec, opt: dict, dt: float) -> None:
    st.markdown(
        "<div class='small'>The LP carries <b>one</b> efficiency for the whole day. A datasheet "
        "carries a curve, a derating and a warranty. Here the LP's own battery plan is put through "
        "a load-dependent efficiency — no re-dispatch, just the energy the linear model forgot."
        "</div>", unsafe_allow_html=True)

    model_mw = max(spec.battery_kw / 1000.0, 1e-3)
    if model_mw > spec.battery_kw / 1000.0 * 1.0001:
        theme.note(
            f"This case's battery is <b>{spec.battery_kw:,.3f} kW</b>, below the 1 kW floor the site "
            "model uses to stay numerically sane, so the LP and the device are not the same machine "
            "and the probe would be comparing two different batteries. Run it on cases 02, 03 or 05."
        )
        return
    rating_mw = model_mw
    energy_mwh = max(spec.battery_kwh / 1000.0, 1e-9)
    device_c = rating_mw / energy_mwh
    eta0 = 0.95
    k = st.slider("How steep is the efficiency curve?", 0.0, 0.30, 0.10, 0.01,
                  key=f"blind_k_{spec.key}",
                  help="η(x) = η₀ − k·x², x = |p| / P. k = 0 is exactly what PyPSA assumes.")
    gap = linearity_gap(opt, dt, rating_mw, energy_mwh, eta0=eta0, k=k)
    price_mean = float((opt["dispatch"]["Grid import"].to_numpy()
                        * np.asarray(opt["prices"].iloc[:, 0], dtype=float)).sum()
                       / max(opt["dispatch"]["Grid import"].sum(), 1e-9)) \
        if "Grid import" in opt["dispatch"] else 0.0
    cost = gap["extra_store_mwh"] * price_mean
    lp_bill = float(opt["energy_cost"] + opt["carbon_cost"])

    theme.kpis([
        ("Plan peak rate", f"{gap['plan_peak_c']:.2f}", "C",
         f"the device is rated {device_c:.2f} C ({rating_mw:,.2f} MW / {energy_mwh:,.1f} MWh)", ""),
        ("Energy the LP forgets", f"{gap['extra_store_mwh']:,.3f}", "MWh",
         f"on {gap['throughput_mwh']:,.1f} MWh of throughput", "bad" if k else "good"),
        ("Worth", f"{cost:,.2f}", "USD",
         (f"{cost / lp_bill:.2%} of the LP bill, at λ̄ = {price_mean:,.0f} USD/MWh"
          if lp_bill > 1.0 else f"at λ̄ = {price_mean:,.0f} USD/MWh"), ""),
        ("Lowest efficiency seen", f"{gap['eta_min']:.3f}", "",
         "at the top of the power range" if k else "constant, by assumption", ""),
    ])

    x = np.linspace(0, 1.2, 80)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=efficiency_curve(x * rating_mw, rating_mw, eta0, k),
                                name=f"the device (k = {k:.2f})", line=dict(color="#f472b6", width=3)))
    figure.add_trace(go.Scatter(x=x, y=np.full_like(x, eta0), name="what the LP assumes",
                                line=dict(color="#34d399", width=2, dash="dash")))
    figure.add_vline(x=min(gap["plan_peak_c"], 1.2), line=dict(color="#fbbf24", dash="dot"),
                     annotation_text=f"peak of the plan ({gap['plan_peak_c']:.2f} C)",
                     annotation_position="top left")
    theme.show(theme.figure(figure, height=300, x_title="|p| / P · fraction of the rating",
                            y_title="round-trip efficiency"))

    theme.note(
        "PyPSA is not wrong here — it is <i>linear</i>. A linear program cannot represent a "
        "non-convex efficiency curve without integer variables, and with them the certificate and "
        "the speed go away. What the page shows is the size of the approximation: "
        f"<b>{gap['extra_store_mwh']:,.3f} MWh</b> of losses that exist in the plant and not in the "
        "model. If the number is big enough to matter, the model is missing a constraint you know "
        "how to write (a piecewise efficiency with a binary) — and that is a modelling decision, "
        "not a bug."
    )


# --------------------------------------------------------------------------- #
# Changing the policy
# --------------------------------------------------------------------------- #

CONTRACT = '''class DispatchPolicy(ABC):
    @abstractmethod
    def apply(self, p_delta: float, duration: float,
              dispatchables: list[Dispatchable],
              grid_signals: dict[str, float] | None = None) -> None:
        """Assign feasible setpoints that fully balance the power delta.

        p_delta > 0  excess power   ·   p_delta < 0  deficit
        No value is returned. Every resource must end with a feasible setpoint
        and the remaining delta must be exactly zero.
        """

    def state(self) -> dict:
        """Whatever you want the recorder to see. Default: {}."""
'''

SWAP = '''import vessim as vs

env = vs.Environment(sim_start="2026-01-01", step_size=900)
env.add_microgrid(
    actors=[vs.Consumer("planta", plant_trace),
            vs.Producer("solar", solar_trace)],
    dispatchables=[vs.SimpleBattery("bess", capacity=40_000, initial_soc=0.5),
                   FuelGenerator("gas", p_max=20_000, fuel_cost=0.105),
                   vs.GridConnection("grid", max_import=12_000, max_export=4_000)],
    policy=vs.DefaultDispatchPolicy(),         # ← the only line that changes
    grid_signals={"price": price_trace, "carbon": carbon_trace},
)
env.run(until=24 * 3600)
'''

SOCIETY = '''from vessim_pypsa_dashboard.policies import SocFloorPolicy

class SocFloorPolicy(MeritOrderPolicy):
    """A resilience rule: the battery is a reserve, not a trader."""

    def __init__(self, floor=0.4, ceiling=0.9, **kwargs):
        super().__init__(**kwargs)
        self.floor, self.ceiling = floor, ceiling

    def supply_score(self, d, price, carbon):
        if isinstance(d, vs.Storage) and _soc_of(d) <= self.floor:
            return 1e6                     # forbidden to discharge
        return super().supply_score(d, price, carbon)

    def absorb_score(self, d, price):
        if isinstance(d, vs.Storage) and _soc_of(d) >= self.ceiling:
            return 1e6                     # full enough
        return super().absorb_score(d, price)
'''


def _regret(run_like: dict, spec, opt: dict) -> float:
    _, info = maths.gap_table(run_like, spec, opt)
    return float(info["gap"])


def _policy_code(strategy: str, carbon: float, degradation: float, export_factor: float,
                 soc_target: float, charge_bonus: float) -> str:
    return (
        "from vessim_pypsa_dashboard.policies import MeritOrderPolicy\n\n"
        "policy = MeritOrderPolicy(\n"
        f"    carbon_price_usd_per_t={carbon:.0f},\n"
        f"    strategy={strategy!r},\n"
        f"    degradation_cost_usd_per_kwh={degradation:.3f},\n"
        f"    export_price_factor={export_factor:.2f},\n"
        f"    soc_target={soc_target:.2f},\n"
        f"    charge_bonus_usd_per_kwh={charge_bonus:.3f},\n"
        ")\n"
        "env.add_microgrid(..., policy=policy)\n"
    )


def _section_policies(case_key: str, spec, carbon: float, solar: float, load: float,
                      run: dict, opt: dict) -> None:
    theme.section("Change the policy", "a dispatch policy is a class with one method", "HOW TO")
    st.markdown(
        "<div class='small'>Vessim ships <b>one</b> policy, <code>DefaultDispatchPolicy</code>: "
        "allocate in list order. Everything else — merit order, carbon-first, peak shaving, your "
        "own rule — is a subclass of <code>DispatchPolicy</code> passed to "
        "<code>add_microgrid(policy=…)</code>. There is no registry, no configuration file: the "
        "policy is an object, so changing the policy is changing the argument. "
        "The full how-to — the contract, the eight shipped rules, an editor that compiles and runs "
        "your own class, and the five ways a policy breaks — is on <b>🛠️ Change the policy</b>. "
        "Below is the short version, measured.</div>",
        unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**1 · The contract you have to satisfy**")
        st.code(CONTRACT, language="python")
        st.caption("Four arguments, no return value, and the sum must be exactly zero. "
                   "`feasible_range(duration)` on every dispatchable tells you what it can do.")
    with right:
        st.markdown("**2 · Where the policy is plugged in**")
        st.code(SWAP, language="python")
        st.caption("This is the whole integration: `policy=` in `add_microgrid`.")

    theme.section("3 · Write one and run it here",
                  "six numbers, one class — the same door your own code will use", "PLAYGROUND")
    controls, result = st.columns([1, 1.25])
    with controls:
        strategy = st.selectbox("Strategy", ["cost", "carbon_first", "battery_first"],
                                key=f"blind_strategy_{case_key}",
                                help="cost: price + carbon; carbon_first: ×8 on carbon; "
                                     "battery_first: dump the battery")
        tau = st.slider("Carbon price τ (USD/t)", 0, 300, int(carbon), 10,
                        key=f"blind_tau_{case_key}")
        degradation = st.slider("Battery wear (USD/kWh discharged)", 0.0, 0.10, 0.05, 0.005,
                                key=f"blind_deg_{case_key}")
        export_factor = st.slider("Export earns (share of price)", 0.0, 1.0, 0.35, 0.05,
                                  key=f"blind_exp_{case_key}")
        soc_target = st.slider("SoC the rule defends", 0.1, 0.9, 0.65, 0.05,
                               key=f"blind_soc_{case_key}")
        charge_bonus = st.slider("Charge bonus (USD/kWh)", 0.0, 0.10, 0.02, 0.005,
                                 key=f"blind_bonus_{case_key}")
    params = (strategy, degradation, export_factor, soc_target, charge_bonus)
    mine = data.policy_variant(case_key, float(tau), solar, load, "custom_merit", params)
    my_bill = float(mine["baseline"]["total_cost_usd"])
    my_co2 = float(mine["baseline"]["emissions_t"])
    my_regret = _regret(mine, spec, opt)
    official_bill = float(run["baseline"]["total_cost_usd"])
    official_co2 = float(run["baseline"]["emissions_t"])
    lp_bill = float(opt["energy_cost"] + opt["carbon_cost"])

    with result:
        theme.kpis([
            ("Your rule", f"{my_bill:,.2f}", "USD", f"{my_co2:,.2f} t CO₂", ""),
            ("Official rule", f"{official_bill:,.2f}", "USD",
             f"{official_co2:,.2f} t CO₂", ""),
            ("Regret vs optimum", f"{my_regret:+,.2f}", "USD",
             f"the LP: {lp_bill:,.2f} USD", "good" if abs(my_regret) < 1e-6 else ""),
        ])
        st.code(_policy_code(strategy, float(tau), degradation, export_factor, soc_target,
                             charge_bonus), language="python")
        st.caption(f"This is running {mine['policy_kind']} on case {spec.number} right now — the "
                   "code above is exactly the object the page passed to `run_case`. Regret is "
                   "closed-book: the energy left in the battery at 24:00 is marked at λ_T, so a rule "
                   "that ends with a fuller tank can show a negative regret.")

    # --- the rule that is not a parameter -------------------------------- #
    theme.section("4 · A rule that no price can express",
                  "and what it costs to keep it", "RESILIENCE")
    st.markdown(
        "<div class='small'>A reserve floor is not a price. Below 40 % the battery may only charge: "
        "not because a number says so, but because the operator says so. In PyPSA that is one more "
        "constraint you must remember to write; in Vessim it is a subclass. Run it and look at the "
        "SoC column — and at the leak.</div>", unsafe_allow_html=True)

    floor = st.slider("Reserve floor (SoC)", 0.1, 0.7, 0.4, 0.05, key=f"blind_floor_{case_key}")
    guard = data.policy_variant(case_key, float(carbon), solar, load, "soc_floor", (floor, 0.9))
    guard_bill = float(guard["baseline"]["total_cost_usd"])
    guard_regret = _regret(guard, spec, opt)
    soc_min = float(guard["df"]["soc_mwh"].min())
    soc_max_capacity = max(float(spec.battery_kwh) / 1000.0, 1e-9)
    theme.kpis([
        ("Bill with the floor", f"{guard_bill:,.2f}", "USD",
         f"{guard_bill - official_bill:+,.2f} vs the official rule", ""),
        ("Regret vs the LP", f"{guard_regret:+,.2f}", "USD",
         "the price of the constraint, not inefficiency",
         "good" if abs(guard_regret) < 1e-6 else ""),
        ("Lowest SoC reached", f"{soc_min / soc_max_capacity:.0%}", "",
         f"floor set at {floor:.0%}", ""),
        ("What it buys", "a reserve", "", "energy the LP would have sold", "good"),
    ])

    st.code(SOCIETY, language="python")
    theme.note(
        "The regret is not inefficiency: it is the <b>price of the constraint</b>. The LP in the "
        "other tabs has no such constraint, so part of its advantage is simply that it was allowed "
        "to do something the operator forbids. And the leak below the floor — if you see "
        f"{soc_min / soc_max_capacity:.0%} instead of {floor:.0%} — is Vessim refusing to let the "
        "plant go unserved: a scoring rule can be overruled by the balance, exactly as a real BMS "
        "trips rather than black out the site. A model would have called it infeasible."
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

def case_controls() -> tuple:
    """The one case selector the whole page shares (one widget, one key)."""
    theme.section("Pick the case", "every probe runs on the trace of this case, at its own τ", "CASE")
    case_key = st.selectbox(
        "Case", CASE_ORDER, index=CASE_ORDER.index("ecopetrol"),
        format_func=lambda key: f"{CASES[key].number} · {CASES[key].title}",
        key="blind_case", label_visibility="collapsed")
    spec = CASES[case_key]
    carbon = float(spec.carbon_price)
    solar = load = 1.0
    run = data.vessim_case(case_key, carbon, solar, load)
    opt = data.site_case(run["df"], maths.default_params(run["spec"], carbon), run["dt"])
    return case_key, spec, carbon, solar, load, run, opt


def body(case: tuple) -> None:
    """The map and the three probes — everything the LP cannot represent."""
    case_key, spec, carbon, solar, load, run, opt = case
    _section_map()
    theme.section("Three probes", "measured on this case, not asserted", "EVIDENCE")
    tab_future, tab_code, tab_device = st.tabs(
        ["🕰️ Perfect foresight", "💻 The software", "🔋 The device"])
    with tab_future:
        _probe_foresight(case_key, spec, carbon, solar, load, opt, run["policy_kind"])
    with tab_code:
        _probe_software(spec)
    with tab_device:
        _probe_device(spec, opt, run["dt"])


def playground(case: tuple) -> None:
    """The short, measured version: tune a rule, run it, price the constraint."""
    case_key, spec, carbon, solar, load, run, opt = case
    _section_policies(case_key, spec, carbon, solar, load, run, opt)


def closing_note() -> None:
    theme.note(
        "<b>What this page does not claim.</b> It does not say Vessim is better: every probe above "
        "is a <i>cost</i> of not knowing, of being discrete, of being curved — a controller pays "
        "them, an LP does not because it does not run. It does not say the gaps are large: two of "
        "the three are under 1 % of the bill on this case, and that is a finding, not a failure. "
        "And it does not say the LP's plan is executable: that is precisely what the Vessim tab, "
        "the policies and the hardware-in-the-loop page exist to check."
    )


def render() -> None:
    """Standalone page (kept for links); the app uses :mod:`ui.explain`."""
    theme.hero(
        "🕶️", "What PyPSA cannot see", "THE OTHER HALF OF THE PICTURE",
        "An LP optimises a <i>model</i>: linear constraints, a horizon it already knows, a scalar "
        "efficiency. Vessim executes <i>software against a plant</i>. This page names what falls "
        "outside the model, measures three of those gaps on real runs, and then does the "
        "constructive half: it shows how to change the dispatch policy — and lets you write one.",
        [("Blind spots", "6"), ("Probes", "3 live"), ("Policies", "8 + yours")],
    )
    case = case_controls()
    body(case)
    playground(case)
    closing_note()
