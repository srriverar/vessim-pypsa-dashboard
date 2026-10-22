"""🛠️ Change the policy — the contract, the plug-in point, and your own class.

The blind-spots page argues *why* the controller is outside PyPSA's model.  This
page is the how-to: what a ``vessim.DispatchPolicy`` has to satisfy, where it is
attached, the eight rules this dashboard ships, and an editor that compiles and
runs the reader's own class on a real case.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..policies import POLICY_CATALOGUE, POLICY_ORDER
from ..vessim_runner import CASES
from . import data, theme

EXAMPLE_FILE = Path(__file__).resolve().parents[3] / "examples" / "my_policy.py"

CONTRACT = '''import vessim as vs

class MyPolicy(vs.DispatchPolicy):
    def apply(self, p_delta, duration, dispatchables, grid_signals=None):
        ...
        for d in dispatchables:
            d.set_power(watts, duration)   # positive = the site consumes
        # Vessim then checks:  p_delta - sum(powers) == 0
'''

PLUG_IN = '''# vanilla Vessim — the only line that changes
env.add_microgrid(
    actors=[...],
    dispatchables=[battery, grid],
    policy=MyPolicy(),                    # <-- here
    grid_signals={"price": price_trace, "carbon": carbon_trace},
)

# this dashboard — the same door, nothing else
from vessim_pypsa_dashboard.vessim_runner import run_case
run = run_case("ecopetrol", carbon_price=60, policy_override=MyPolicy())
print(run.df[["load_mw", "grid_mw", "battery_mw"]].tail())
'''

INVARIANTS = [
    ("the balance is exact",
     "`p_delta − Σ power == 0`, not ≈ 0. Vessim raises `Unmet demand of … W` on a float ulp."),
    ("every setpoint is feasible",
     "`d.feasible_range(duration)` is the interval the device can actually deliver *this* step."),
    ("no dispatchable is stale",
     "Write all of them, or the ones you skipped keep last step's power — and the balance drifts."),
    ("the sign convention is the site's",
     "Positive = the microgrid consumes; a generator or a discharging battery is negative."),
    ("you get the signal, not the future",
     "`grid_signals` is today's value; the LP is handed the whole day. That difference is the point."),
]

PITFALLS = [
    ("Any policy must close the balance itself",
     "`RuntimeError: Unmet demand of 761.029 W` — what Vessim says when `p_delta − Σ power ≠ 0`. "
     "The fix is to write the remainder into the device that can take it "
     "(`exact_balance` in `policies.py` does exactly this), never to clip a plan at `1 − 1e−9`."),
    ("Setting a power is not the same as logging it",
     "A refactor that computed the setpoints and logged them but forgot `set_power` produced the "
     "same `Unmet demand` message — the bug looked like physics and was a missing write."),
    ("The device's limits depend on the step",
     "`feasible_range(900)` is not `feasible_range(60)`: a battery can move 4× more energy in "
     "15 min than in 1 min. Asking for the wrong horizon silently disables the device."),
    ("Vessim subtracts in dispatch order",
     "The exact-zero test is left-to-right subtraction, so `0.1 + 0.2 − 0.3` style dust survives "
     "summing but not subtracting. Repair the last term, then *verify with the same order*."),
    ("An LP plan is not a policy",
     "`OptimalPlanPolicy` executes a schedule: it clips every setpoint to the device and reports "
     "the tracking error. That is how you learn whether the optimum is executable at all."),
]

DEFAULT_SOURCE = '''class MyPolicy(vs.DispatchPolicy):
    """Absorb your own surplus; spend it when the grid is expensive."""

    def __init__(self, expensive=0.20, default_price=0.18):
        super().__init__()
        self.expensive = float(expensive)
        self.default_price = float(default_price)
        self.log = []

    def apply(self, p_delta, duration, dispatchables, grid_signals=None):
        price = self.default_price
        if grid_signals and grid_signals.get("price") is not None:
            price = float(grid_signals["price"])

        battery = next(d for d in dispatchables if isinstance(d, vs.Storage))
        grid = next(d for d in dispatchables if isinstance(d, vs.GridConnection))
        low, high = battery.feasible_range(duration)
        g_low, g_high = grid.feasible_range(duration)

        # 1. the decision (positive = charge)
        if p_delta > 0:                  # surplus: absorb it, never more
            want = min(high, p_delta)
        elif price > self.expensive:     # expensive deficit: serve it, never more
            want = max(low, p_delta)
        else:
            want = 0.0
        # 2. leave the grid room to close the balance
        want = max(low, min(high, want, p_delta - g_low))

        # 3. write every dispatchable
        for d in dispatchables:
            if d is not battery and d is not grid:
                d.set_power(0.0, duration)
        battery.set_power(want, duration)

        # 4. the grid takes exactly the rest
        left = p_delta - want
        if not g_low <= left <= g_high:
            raise RuntimeError(f"the grid cannot absorb {left:,.1f} W at this step")
        grid.set_power(left, duration)
        self.log.append({"price": price, "battery_w": want, "grid_w": left})
'''


def _default_source() -> str:
    """The editable default is the class in ``examples/my_policy.py``, minus imports."""
    try:
        text = EXAMPLE_FILE.read_text()
        body = text[text.index("class ThriftyPolicy"):text.index("def day(")].rstrip()
        return body.replace("ThriftyPolicy", "MyPolicy")
    except Exception:                                  # pragma: no cover - defensive
        return DEFAULT_SOURCE


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def _contract() -> None:
    theme.section("The contract", "four arguments, one method, no registry", "WHERE IT LIVES")
    left, right = st.columns([1, 1])
    with left:
        st.code(CONTRACT, language="python")
        st.caption("Vessim ships exactly one implementation, `DefaultDispatchPolicy`, and it is "
                   "about as long as this. Everything else is your subclass.")
    with right:
        st.markdown("**What you have to satisfy**")
        st.dataframe(pd.DataFrame(INVARIANTS, columns=["Invariant", "Why it bites"]),
                     hide_index=True, **theme.wide(), height=300)
    theme.note(
        "A dispatch policy is called <i>every step</i>, before the step is taken, with the balance "
        "the actors have produced. It is not a schedule and it is not a controller: it is the "
        "plant's decision procedure, and it is an object — which is why changing it is an argument, "
        "not a configuration file."
    )


def _plug_in(spec) -> None:
    theme.section("Where it plugs in", "one keyword on add_microgrid", "INTEGRATION")
    left, right = st.columns([1, 1])
    with left:
        st.code(PLUG_IN, language="python")
        st.caption("Left: the library call. Right: this dashboard, which passes your object "
                   "through `policy_override` without knowing what it is.")
    with right:
        st.markdown("**The case you are looking at, as a Vessim script**")
        st.code(spec.code, language="python")
        st.caption("Simplified for reading: the dashboard also adds the computing actors, the "
                   "curtailment spillway and the `Recorder`.")

    theme.section("The eight rules that ship", "each one is a class, each one is a line", "CATALOGUE")
    rows = []
    for key in POLICY_ORDER:
        item = POLICY_CATALOGUE[key]
        rows.append({"key": key, "rule": item.label, "what it decides on": item.description,
                     "code": f"build_policy(POLICY_CATALOGUE['{key}'], spec)"})
    rows.append({"key": "official", "rule": "Official example (as shipped)",
                 "what it decides on": POLICY_CATALOGUE["official"].description,
                 "code": "run_case(...)  # no override"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **theme.wide(), height=290)

    st.markdown(
        "<div class='small'>Run all seven on <b>this</b> day — same load, same sun, same prices, "
        "same carbon, only the decision rule changes. The last column is the closed-book regret "
        "against the PyPSA optimum of the identical trace.</div>",
        unsafe_allow_html=True)
    if st.button("▶️ Run the seven rules on this case", **theme.wide(), key="guide_matrix"):
        st.session_state["guide_matrix_case"] = spec.key
    if st.session_state.get("guide_matrix_case") == spec.key:
        _matrix(spec)


def _matrix(spec) -> None:
    carbon = float(CASES[spec.key].carbon_price)
    with st.spinner("Seven rules, one day …"):
        runs = data.policy_matrix(spec.key, carbon, 1.0, 1.0,
                                  tuple(POLICY_CATALOGUE[k] for k in POLICY_ORDER))
        opt = data.optimal_case(spec.key, carbon, 1.0, 1.0)
    lp = float(opt["lp_cost"])
    rows = []
    for key, payload in runs.items():
        base = payload["baseline"]
        bill = float(base["total_cost_usd"])
        rows.append({"rule": POLICY_CATALOGUE[key].label, "class": payload["policy_kind"],
                     "bill USD": bill, "CO₂ kg": float(base["emissions_t"]) * 1000,
                     "peak MW": float(base["peak_grid_mw"]) * 1000.0,
                     "regret USD": bill - lp})
    table = pd.DataFrame(rows).sort_values("bill USD")
    st.dataframe(table.round({"bill USD": 2, "CO₂ kg": 1, "peak MW": 1, "regret USD": 2}),
                 hide_index=True, **theme.wide(), height=280)

    figure = go.Figure()
    figure.add_trace(go.Bar(x=table["rule"], y=table["bill USD"], name="bill",
                            marker_color="#38bdf8"))
    figure.add_hline(y=lp, line=dict(color="#34d399", dash="dash", width=2),
                     annotation_text=f"PyPSA optimum {lp:,.2f}",
                     annotation_position="top right")
    theme.show(theme.figure(figure, height=320, y_title="bill for the day (USD)"))
    spread = table["bill USD"].max() - table["bill USD"].min()
    st.caption(f"Spread across the seven rules: {spread:,.2f} USD "
               f"({spread / lp:.1%} of the optimum) for the *same* physical day. That number is "
               "what a policy decision is worth — and it moves with the case, not with the solver.")


def _editor(spec) -> None:
    theme.section("Write one and run it", "compiled in this page, executed by Vessim", "PLAYGROUND")
    st.markdown(
        "<div class='small'>The editor holds a complete policy. <code>vs</code> (vessim), "
        "<code>np</code> and <code>math</code> are already in scope: write no imports, define "
        "<code>class MyPolicy(vs.DispatchPolicy)</code>, press run. If it cannot balance the step, "
        "Vessim will say so and the page will show you the message — that is the lesson, not a "
        "crash.</div>", unsafe_allow_html=True)

    case_key = st.session_state.get("guide_case", "basic")
    chosen = st.selectbox(
        "Case to run it on", list(CASES),
        index=list(CASES).index(case_key),
        format_func=lambda k: f"{CASES[k].number} · {CASES[k].title}",
        key="guide_case_box")
    st.session_state["guide_case"] = chosen
    case = CASES[chosen]

    source = st.text_area("Your policy", value=st.session_state.get("guide_source", _default_source()),
                          height=420, key="guide_source_box")
    st.session_state["guide_source"] = source

    left, right = st.columns([1, 1.4])
    run_it = left.button("▶️ Compile and run", type="primary", **theme.wide())
    right.caption("The code runs in the app's process on this case's real trace — no sandbox, "
                  "so keep it to the policy class.")

    if not run_it and "guide_result" not in st.session_state:
        st.info("The button runs your class through `run_case(policy_override=…)` on the trace of "
                "the selected case, exactly as the library policies are run.")
        return

    if run_it:
        try:
            result = data.custom_policy_run(chosen, float(case.carbon_price), 1.0, 1.0, source)
            st.session_state["guide_result"] = {"case": chosen, "result": result,
                                                "source": source}
        except Exception as exc:
            st.session_state.pop("guide_result", None)
            theme.note(f"<b>{type(exc).__name__}:</b> {exc}", "bad")
            st.caption("This is Vessim refusing the policy, not the page: the residual was not "
                       "zero, or the device was asked for something it cannot deliver.")
            return

    cached = st.session_state.get("guide_result")
    if not cached or cached["case"] != chosen or cached["source"] != source:
        st.info("The code changed — press run to compile and execute it again.")
        return
    _show_result(cached["result"], case)


def _show_result(result: dict, case) -> None:
    from .maths import gap_table

    opt = data.optimal_case(case.key, float(case.carbon_price), 1.0, 1.0)
    base = result["baseline"]
    bill = float(base["total_cost_usd"])
    lp = float(opt["lp_cost"])
    official = data.vessim_case(case.key, float(case.carbon_price), 1.0, 1.0)
    reference = float(official["baseline"]["total_cost_usd"])
    theme.kpis([
        ("Your rule", f"{bill:,.2f}", "USD",
         f"{float(base['emissions_t']) * 1000:,.1f} kg CO₂ · peak "
         f"{float(base['peak_grid_mw']) * 1000:,.0f} kW", ""),
        ("Official rule", f"{reference:,.2f}", "USD",
         f"{bill - reference:+,.2f} USD against it",
         "good" if bill < reference else "bad"),
        ("PyPSA optimum", f"{lp:,.2f}", "USD", "the same day, decided with the whole trace", ""),
        ("Lines of policy", f"{result['source_lines']}", "",
         f"{result['policy_kind']} · {len(result['policy_log'])} decisions logged", "good"),
    ])
    st.dataframe(pd.DataFrame({
        "what": ["bill", "CO₂", "gas", "grid", "battery throughput", "peak import"],
        "your rule": [f"{bill:,.2f} USD", f"{float(base['emissions_t']) * 1000:,.1f} kg",
                      f"{float(base['gas_mwh']):.3f} MWh", f"{float(base['grid_mwh']):.3f} MWh",
                      f"{float(base['battery_throughput_mwh']):.3f} MWh",
                      f"{float(base['peak_grid_mw']) * 1000:,.0f} kW"],
        "official": [f"{reference:,.2f} USD",
                     f"{float(official['baseline']['emissions_t']) * 1000:,.1f} kg",
                     f"{float(official['baseline']['gas_mwh']):.3f} MWh",
                     f"{float(official['baseline']['grid_mwh']):.3f} MWh",
                     f"{float(official['baseline']['battery_throughput_mwh']):.3f} MWh",
                     f"{float(official['baseline']['peak_grid_mw']) * 1000:,.0f} kW"],
    }), hide_index=True, **theme.wide(), height=250)
    st.caption("Regret is deliberately not on this table: a rule can beat the LP's bill by ending "
               "the day with a fuller battery. The closed-book ledger on the `🏁 Follow the "
               "optimum` sheet is the version that survives that objection.")


def _pitfalls() -> None:
    theme.section("Five ways a policy breaks", "all five happened in this repository", "PITFALLS")
    for number, (title, body) in enumerate(PITFALLS, start=1):
        st.markdown(f"**{number}. {title}**  \n{body}")


def _take_away() -> None:
    theme.section("Take it away", "the same file, outside Streamlit", "FILE")
    try:
        source = EXAMPLE_FILE.read_text()
    except Exception:                                   # pragma: no cover - defensive
        source = ""
    if source:
        st.download_button("⬇️ Download examples/my_policy.py", source.encode(),
                           file_name="my_policy.py", mime="text/python", **theme.wide(),
                           key="guide_download")
        with st.expander("What the file does, and what it prints", expanded=False):
            st.markdown(
                "* one class, `ThriftyPolicy`, and one 24 h microgrid: 700 W load, a winter sun, a "
                "6 kWh battery, a 3 kW grid connection;\n"
                "* the policy absorbs its own surplus and serves the deficit when the price is "
                "above 0.20 USD/kWh — it never imports to charge, because the export tariff "
                "(35 % of the price) is below the import price;\n"
                "* output: charged/emptied kWh, imports and exports from Vessim's own metrics, the "
                "energy bill, and the same day with no battery and no rule.")
    st.code('''# in this repository
from vessim_pypsa_dashboard.vessim_runner import run_case
from my_policy import ThriftyPolicy

run = run_case("basic", carbon_price=60, policy_override=ThriftyPolicy())
print(run.policy_kind, run.baseline["total_cost_usd"])''', language="python")
    theme.note(
        "When you publish a result that depends on a policy, publish the class and the four "
        "invariants it satisfies. A prose description (“a price-aware heuristic”) is not "
        "reproducible; a subclass and its residual are."
    )


def body(spec=None) -> None:
    """Everything except the hero: contract, plug-in, catalogue, editor, pitfalls."""
    if spec is None:
        case_key = st.session_state.get("nav", "")
        spec = CASES[case_key] if case_key in CASES else CASES["ecopetrol"]
    _contract()
    _plug_in(spec)
    _editor(spec)
    _pitfalls()
    _take_away()
    theme.note(
        "The policy is also the honest boundary of the comparison: PyPSA and Vessim can be given "
        "the same day, the same prices and the same battery, but only one of them can be handed a "
        "<i>rule</i>. On the same page, the <b>🕶️ What the LP cannot see</b> tab comes back to "
        "these rules as probes: perfect foresight, the software itself, and the device's "
        "efficiency curve."
    )


def render() -> None:
    case_key = st.session_state.get("nav", "")
    spec = CASES[case_key] if case_key in CASES else CASES["ecopetrol"]
    theme.hero(
        "🛠️", "Change the policy", "ONE CLASS · ONE METHOD · FIVE WAYS IN",
        "Vessim's whole notion of “how the plant decides” is a subclass of `DispatchPolicy` with "
        "one method, attached with `policy=` when the microgrid is created. That is the API this "
        "dashboard exploits: the rules you saw in the tabs are ordinary classes, the optimum is "
        "another one, and the editor below is the same door — write your class, press run, and "
        "Vessim executes it on a real day.",
        [("The contract", "apply(p_delta, duration, dispatchables, signals)"),
         ("Shipped rules", f"{len(POLICY_ORDER)} + Vessim's own"),
         ("Your rule", "~25 lines, run here")],
    )
    body(spec)
