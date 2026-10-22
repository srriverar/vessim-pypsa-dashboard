"""The maths sheet — every number on the dashboard, written as an equation.

Each block follows the same three steps:

1. the equation, typeset;
2. the same equation with **the numbers of the run you are looking at**
   substituted into it, so it can be checked with a pocket calculator;
3. one sentence saying what the result *means* — the part a table of KPIs never
   tells you.

The Vessim half is a simulation: recursive rules applied one step at a time.
The PyPSA half is a linear program over the whole horizon.  Showing both on the
same page, with the same symbols, is the point of the whole application.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from . import data, theme
from ..vessim_runner import CASES

# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #


def _n(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:,.{digits}f}"


def _auto(value: float) -> str:
    """Digits that follow the magnitude: 0.0007 MW and 12 MW both read well."""
    if value is None or not math.isfinite(value):
        return "—"
    if abs(value) < 1e-9:
        value = 0.0                       # never print "-0.0000"
    size = abs(value)
    digits = 4 if size < 0.01 else (3 if size < 1 else (2 if size < 100 else 1))
    return f"{value:,.{digits}f}"


def _usd(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f} USD"


def _pct(value: float, digits: int = 1) -> str:
    return "—" if not math.isfinite(value) else f"{value:+.{digits}f} %"


def _block(latex: str, lines: list[str], meaning: str, tone: str = "") -> None:
    """Equation, numeric substitution, and the sentence that explains it."""
    st.latex(latex)
    if lines:
        st.code("\n".join(lines), language="text")
    if meaning:
        theme.note(meaning, tone)


def _subtitle(text: str) -> None:
    st.markdown(f'<div class="small" style="margin:.1rem 0 .55rem">{text}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# The numbers of one run, Vessim side
# --------------------------------------------------------------------------- #


def vessim_numbers(run: dict, spec, step: int) -> dict:
    """Everything the Vessim equations need, taken from the recorder itself."""
    spec = run.get("spec", spec)          # the run's own spec carries the live τ
    df: pd.DataFrame = run["df"]
    dt: float = float(run["dt"])
    step = int(np.clip(step, 0, len(df) - 1))
    row = df.iloc[step]

    load = float(row["load_mw"])
    solar = float(row["solar_mw"])
    gas = float(row.get("gas_mw", 0.0) or 0.0)
    battery = float(row["battery_mw"])          # + discharge, − charge
    grid = float(row["grid_mw"])                # + import, − export
    curtailed = float(row.get("curtailed_mw", 0.0) or 0.0)
    soc = float(row.get("soc_mwh", 0.0) or 0.0)

    price = float(row.get("price_usd_mwh", 0.0) or 0.0)
    carbon_intensity = float(row.get("carbon_kg_mwh", 0.0) or 0.0)  # kg/MWh
    carbon_t = carbon_intensity / 1000.0

    tau = float(spec.carbon_price)                       # USD/tCO2
    gas_cost = float(spec.gas_cost_usd_kwh) * 1000.0     # USD/MWh
    gas_co2 = float(spec.gas_co2)                        # tCO2/MWh
    capacity = max(float(spec.battery_kwh) / 1000.0, 1e-12)   # MWh
    power = max(float(spec.battery_kw) / 1000.0, 1e-12)       # MW

    grid_import = max(grid, 0.0)
    grid_export = max(-grid, 0.0)
    export_factor = 0.35                                  # Vessim pays 35 % on export

    # battery recursion, exactly as the recorder wrote it
    soc_next = float(df["soc_mwh"].iloc[step + 1]) if step + 1 < len(df) else float("nan")
    soc_delta = (soc_next - soc) if math.isfinite(soc_next) else float("nan")

    energy_cost = float(run["baseline"]["energy_cost_usd"])
    carbon_cost = float(run["baseline"]["carbon_cost_usd"])
    total_cost = float(run["baseline"]["total_cost_usd"])
    emissions = float(run["baseline"]["emissions_t"])

    grid_mwh = float(run["baseline"]["grid_mwh"])
    gas_mwh = float(run["baseline"]["gas_mwh"])
    export_mwh = float(run["baseline"]["export_mwh"])
    solar_mwh = float(run["baseline"]["solar_mwh"])

    return {
        "step": step, "dt": dt, "n": len(df), "row": row,
        "load": load, "solar": solar, "gas": gas, "battery": battery, "grid": grid,
        "curtailed": curtailed, "soc": soc, "soc_next": soc_next, "soc_delta": soc_delta,
        "price": price, "carbon_intensity": carbon_intensity, "carbon_t": carbon_t,
        "tau": tau, "gas_cost": gas_cost, "gas_co2": gas_co2,
        "capacity": capacity, "power": power,
        "grid_import": grid_import, "grid_export": grid_export,
        "export_factor": export_factor,
        "energy_cost": energy_cost, "carbon_cost": carbon_cost, "total_cost": total_cost,
        "emissions": emissions, "grid_mwh": grid_mwh, "gas_mwh": gas_mwh,
        "export_mwh": export_mwh, "solar_mwh": solar_mwh,
        "residual": load - (solar + gas + battery + grid),
        # what one step of each source costs *at this step*, USD
        "step_grid_cost": price * grid_import * dt,
        "step_gas_cost": gas_cost * gas * dt,
        "step_export_credit": export_factor * price * grid_export * dt,
        "step_emissions_kg": carbon_intensity * grid_import * dt + gas_co2 * gas * dt * 1000.0,
        # effective marginal cost of each source at this step, USD/MWh
        "grid_effective": price + tau * carbon_t,
        "gas_effective": gas_cost + tau * gas_co2,
        "solar_effective": 0.0,
    }


# --------------------------------------------------------------------------- #
# Section 1 — Vessim
# --------------------------------------------------------------------------- #


def vessim_sheet(run: dict, spec, step: int, *, compact: bool = False) -> dict:
    """Write down what Vessim did, with the arithmetic."""
    v = vessim_numbers(run, spec, step)
    t = pd.Timestamp(run["df"].index[v["step"]])

    _subtitle(
        f"Step {v['step']} of {v['n']} · {t.strftime('%d %b %H:%M')} · "
        f"Δt = {_n(v['dt'], 3)} h ({_n(v['dt'] * 60, 0)} min). "
        "Vessim writes one row per step; everything below is that row."
    )

    # (1) energy balance ---------------------------------------------------
    _block(
        r"L_t \;=\; S_t + G_t + B_t + I_t",
        [
            f"L_t  load ................ {_n(v['load'], 4)} MW",
            f"S_t  solar ............... {_n(v['solar'], 4)} MW",
            f"G_t  gas turbine ......... {_n(v['gas'], 4)} MW",
            f"B_t  battery (+dis/−chg) . {_n(v['battery'], 4)} MW",
            f"I_t  grid import ......... {_n(v['grid'], 4)} MW",
            f"residual L−(S+G+B+I) ..... {_n(v['residual'], 6)} MW",
        ],
        "This is an <b>accounting identity</b>, not an optimisation result: the recorder asks each "
        "actor what it did and sums it. A residual of zero means nothing was lost between the "
        "actors; any residual would be a bookkeeping bug, not physics.",
        "good" if abs(v["residual"]) < 1e-6 else "bad",
    )

    # (2) battery state ----------------------------------------------------
    if v["capacity"] > 0:
        delta_expected = -v["battery"] * v["dt"] / v["capacity"]
        _block(
            r"SoC_{t+1} \;=\; SoC_t \;-\; \frac{B_t\,\Delta t}{E_{nom}},"
            r"\qquad 0 \le SoC_t \le 1",
            [
                f"E_nom capacity ........... {_n(v['capacity'], 4)} MWh",
                f"P_nom power .............. {_n(v['power'], 4)} MW",
                f"SoC_t .................... {_n(v['soc'], 4)} MWh  "
                f"({_n(100 * v['soc'] / v['capacity'], 2)} %)",
                f"−B_t·Δt .................. {_n(-v['battery'] * v['dt'], 5)} MWh  (energy moved)",
                f"SoC_(t+1) expected ....... {_n(v['soc'] - v['battery'] * v['dt'], 4)} MWh",
                f"SoC_(t+1) recorded ....... {_n(v['soc_next'], 4)} MWh",
                f"rounding difference ...... {_n(v['soc_delta'] - delta_expected, 6)} MWh",
            ],
            "Vessim's <code>SimpleBattery</code> is lossless, so the state simply moves by the energy "
            "it exchanged. Charging (B&lt;0) <i>raises</i> the state, discharging lowers it. The "
            "clipping at 0 and E_nom is the only reason the equality can fail — the battery is "
            "never allowed to go outside its box.",
        )

    # (3) cost of the step -------------------------------------------------
    _block(
        r"C_t = \Delta t\left(\pi_t\,I_t \;-\; \phi\,\pi_t\,X_t \;+\; c_g\,G_t\right),"
        r"\qquad \phi = 0.35",
        [
            f"π_t  grid price ........... {_n(v['price'], 2)} USD/MWh",
            f"Δt·π_t·I_t ................ {_usd(v['step_grid_cost'], 4)}",
            f"−Δt·φ·π_t·X_t (export) .... {_usd(-v['step_export_credit'], 4)}",
            f"Δt·c_g·G_t (gas) .......... {_usd(v['step_gas_cost'], 4)}",
            f"c_g gas cost .............. {_n(v['gas_cost'], 2)} USD/MWh",
            f"C_t ....................... {_usd(v['step_grid_cost'] - v['step_export_credit'] + v['step_gas_cost'], 4)}",
        ],
        "Vessim charges <b>energy only</b>, with the price the signal carried at that instant. It has "
        "no notion of a network, so it never pays losses, congestion or a nodal price — that is "
        "exactly what the PyPSA half adds.",
    )

    # (4) emissions of the step and of the horizon -------------------------
    _block(
        r"E_t = \Delta t\left(e_t\,I_t + e_g\,G_t\right), \qquad "
        r"C^{CO_2} = \tau \sum_t E_t",
        [
            f"e_t grid intensity ....... {_n(v['carbon_intensity'], 2)} kg/MWh "
            f"= {_n(v['carbon_t'], 5)} t/MWh",
            f"e_g gas intensity ........ {_n(v['gas_co2'], 3)} t/MWh",
            f"E_t this step ............ {_n(v['step_emissions_kg'], 3)} kg",
            f"Σ_t E_t (horizon) ........ {_n(v['emissions'], 3)} t",
            f"τ carbon price ........... {_n(v['tau'], 2)} USD/t",
            f"τ·Σ_t E_t ................ {_usd(v['carbon_cost'], 2)}",
            f"energy + carbon .......... {_usd(v['total_cost'], 2)}",
        ],
        "The carbon price τ enters <b>twice</b> in this dashboard: as a cost added here, and as a "
        "signal Vessim's policy receives. Adding it to the bill without showing it to the policy "
        "would make the comparison unfair.",
    )

    # (5) why the step looks like that -------------------------------------
    order = sorted(
        [("solar", v["solar_effective"], v["solar"]),
         ("battery (if it has energy)", 0.0, abs(v["battery"])),
         ("gas turbine", v["gas_effective"], v["gas"]),
         ("grid import", v["grid_effective"], v["grid_import"])],
        key=lambda item: item[1],
    )
    lines = ["merit order at this step (USD/MWh, carbon included)"]
    for name, cost, served in order:
        lines.append(f"  {name:<28s} {cost:9.2f}   serving {_auto(served):>9s} MW")
    _block(
        r"\text{use the cheapest source first, subject to } "
        r"P^{min}_i \le p_i \le P^{max}_i,\; SoC \in [0, E_{nom}]",
        lines,
        f"The decision rule Vessim applies is a <b>priority list</b>: at this step the grid's "
        f"effective cost is {_n(v['grid_effective'], 2)} USD/MWh against {_n(v['gas_effective'], 2)} "
        f"for the gas turbine, so the turbine is asked to cover the residual and the grid is only "
        f"used if it still is not enough. The policy sees <i>this</i> step and nothing else — no "
        f"look-ahead, no network, no shadow price.",
    )

    if not compact:
        theme.note(
            "<b>What Vessim cannot see.</b> (a) <i>Time:</i> it decides step by step, so it cannot "
            "charge the battery at 03:00 to sell at 19:00. (b) <i>Space:</i> the site is one copper "
            "plate, so a congested line is invisible. (c) <i>Price:</i> it has a signal, not a "
            "marginal price, so it cannot value the last MWh. Each of the three becomes a term of "
            "the gap we compute in section 3."
        )
    return v


# --------------------------------------------------------------------------- #
# Section 2 — PyPSA
# --------------------------------------------------------------------------- #


def pypsa_numbers(opt: dict, params: dict, run: dict, step: int) -> dict:
    dispatch = opt["dispatch"]
    step = int(np.clip(step, 0, len(dispatch) - 1))
    dt = float(run["dt"])
    row = dispatch.iloc[step]
    battery = opt.get("battery_mw", pd.Series(0.0, index=dispatch.index))
    prices = opt["prices"]

    load = float(run["df"]["load_mw"].iloc[step])
    grid = float(row.get("Grid import", 0.0))
    solar = float(row.get("Solar", 0.0))
    gas = float(row.get("Gas turbine", 0.0))
    batt = float(battery.iloc[step]) if len(battery) > step else 0.0
    lamp = float(prices.iloc[step].get("Site", float("nan"))) if len(prices) else float("nan")
    soc = float(opt["soc"].iloc[step, 0]) if len(opt["soc"]) else float("nan")

    price = float(run["df"]["price_usd_mwh"].iloc[step])
    carbon_t = float(run["df"]["carbon_kg_mwh"].iloc[step]) / 1000.0
    tau = float(params.get("carbon_price", 0.0))

    return {
        "step": step, "dt": dt, "row": row,
        "load": load, "grid": grid, "solar": solar, "gas": gas, "battery": batt,
        "lambda": lamp, "soc": soc,
        "price": price, "carbon_t": carbon_t, "tau": tau,
        "grid_cost": price + tau * carbon_t,
        "gas_cost": float(params.get("gas_price", 0.0)) + tau * float(params.get("gas_carbon", 0.0)),
        "residual": load - (grid + solar + gas + batt),
        "battery_mw": float(params.get("battery_mw", 0.0)),
        "battery_mwh": float(params.get("battery_mw", 0.0)) * float(params.get("battery_hours", 0.0)),
        "line_grid": float(params.get("line_grid_mw", 0.0)),
        "line_solar": float(params.get("line_solar_mw") or 0.0),
        "line_gas": float(params.get("line_gas_mw") or 0.0),
        "efficiency": float(params.get("battery_efficiency", 1.0)),
        "total_cost": float(opt["energy_cost"] + opt["carbon_cost"]),
        "energy_cost": float(opt["energy_cost"]),
        "carbon_cost": float(opt["carbon_cost"]),
        "emissions": float(opt["emissions_t"]),
        "curtailment": float(opt["curtailment_mwh"]),
        "throughput": float(opt["battery_throughput_mwh"]),
        "unmet": float(opt["unmet_mwh"]),
        "runtime": float(opt["runtime_s"]),
        "status": str(opt["status"]),
    }


def pypsa_sheet(opt: dict, params: dict, run: dict, step: int, *,
                compact: bool = False, lam_check: tuple[float, float] | None = None) -> dict:
    p = pypsa_numbers(opt, params, run, step)

    _subtitle(
        "PyPSA does not simulate: it solves one linear program over all "
        f"{len(opt['dispatch'])} snapshots at once, on the case's own network ({len(opt.get('buses', ()))} buses)."
    )

    _block(
        r"\min_{g,p,c,d,e}\;\; \sum_{t}\Delta t\Big["
        r"\underbrace{(\pi_t+\tau e_t)\,g_t}_{\text{grid}} \;+\;"
        r"\underbrace{(c_g+\tau e_g)\,p_t}_{\text{gas}}\Big]",
        [
            f"π_t + τ·e_t  grid, all in .. {_n(p['grid_cost'], 2)} USD/MWh",
            f"c_g + τ·e_g  gas, all in ... {_n(p['gas_cost'], 2)} USD/MWh",
            f"solar marginal cost ........ 0.00 USD/MWh",
            f"battery marginal cost ...... 0.00 USD/MWh (lossless, no degradation term)",
            f"objective reached .......... {_usd(p['total_cost'], 2)}  ({p['status']}, {p['runtime']:.2f} s)",
        ],
        "The carbon price is folded straight into the marginal cost, so PyPSA sees one number per "
        "source. The battery and the sun are free at the margin — that is why an optimum can cycle "
        "the battery purely to move energy through time.",
    )

    _block(
        r"g_t + s_t + p_t - c_t + d_t = L_t \quad(\text{Site bus}),\qquad"
        r"|f_{\ell,t}| \le S_\ell,\qquad"
        r"e_{t+1} = e_t + \Delta t\left(\eta c_t - \tfrac{d_t}{\eta}\right)",
        [
            f"g_t grid ................... {_n(p['grid'], 4)} MW",
            f"s_t solar ................... {_n(p['solar'], 4)} MW",
            f"p_t gas ..................... {_n(p['gas'], 4)} MW",
            f"battery (+discharge) ........ {_n(p['battery'], 4)} MW",
            f"L_t load .................... {_n(p['load'], 4)} MW",
            f"residual .................... {_n(p['residual'], 6)} MW",
            f"|POI–Site| ≤ S .............. {_n(abs(p['grid']), 3)} ≤ {_n(p['line_grid'], 3)} MVA",
            f"|Site–PV collector| ≤ S ..... {_n(abs(p['solar']), 3)} ≤ {_n(p['line_solar'], 3)} MVA",
            f"battery energy .............. {_n(p['soc'], 3)} of {_n(p['battery_mwh'], 3)} MWh",
        ],
        "The same balance as Vessim's, but written <b>per bus</b> and enforced for every hour at "
        "once. That single change is what creates losses, congestion and a price at each node. The "
        "residual is zero to solver tolerance because a hard constraint says it must be.",
        "good" if abs(p["residual"]) < 1e-4 else "warn",
    )

    # the dual -------------------------------------------------------------
    lines = [
        "price at the site bus (dual of the balance constraint)",
        f"  λ_t reported by PyPSA ....... {_n(p['lambda'], 3)} USD/MWh",
    ]
    meaning = (
        "λ is the <b>marginal value of one more MWh of load</b> at that hour: the shadow price of "
        "the balance constraint. It is not an accounting average — it is a derivative."
    )
    if lam_check is not None:
        numerical, delta = lam_check
        lines += [
            f"  λ measured by finite difference {_n(numerical, 3)} USD/MWh",
            f"  δ used in the probe ......... ±{_n(delta, 3)} MW at this snapshot",
            f"  |λ_report − λ_measured| ..... {_n(abs(numerical - p['lambda']), 6)}",
        ]
        meaning += (
            " We re-solved the whole day twice, with one snapshot's load nudged by ±δ, and took "
            "(C₊ − C₋)/(2δΔt). The two numbers agreeing is the proof that the price PyPSA prints "
            "really is the derivative of the optimum — the Lagrangian multiplier of the problem."
        )
    _block(
        r"\lambda_t \;=\; \frac{\partial C^\star}{\partial L_t} \;=\; "
        r"\frac{\partial \mathcal{L}}{\partial L_t}",
        lines, meaning,
    )

    if not compact:
        state = ("charging" if p["battery"] < -1e-9 else
                 ("discharging" if p["battery"] > 1e-9 else "idle"))
        text = (f"<b>Reading the optimum.</b> At this step PyPSA takes "
                f"{_auto(max(p['grid'], 0.0))} MW from the grid and {_auto(max(p['gas'], 0.0))} MW "
                f"from the turbine, while the battery is {state} at "
                f"{_auto(abs(p['battery']))} MW. Every unit is used up to the point where its cost "
                f"equals λ = {_n(p['lambda'], 2)} USD/MWh, or up to its limit if the limit binds "
                f"(line {_auto(p['line_grid'])} MVA, battery {_auto(p['battery_mw'])} MW, "
                f"tank {_auto(p['battery_mwh'])} MWh).")
        if p["battery"] < -1e-9 and p["gas"] > 1e-9:
            text += (
                f" Charging <i>while the turbine runs</i> is not a mistake: the battery buys at "
                f"λ = {_n(p['lambda'], 2)} USD/MWh now to displace "
                f"{_auto(p['grid_cost'])} USD/MWh of grid energy later. A rule that only looks at "
                f"the present cannot make that trade — this is the gap in one sentence."
            )
        theme.note(text)
    return p


# --------------------------------------------------------------------------- #
# Section 3 — where the difference comes from
# --------------------------------------------------------------------------- #


def gap_table(run: dict, spec, opt: dict) -> tuple[pd.DataFrame, dict]:
    """Exact decomposition of ``C_Vessim − C_PyPSA`` into its priced causes.

    Two conventions keep the comparison honest, and both are needed:

    * **The same objective.** Vessim's policy prices every discharged MWh at
      ``degradation_cost_usd_per_kwh`` (0.02 USD/kWh).  PyPSA is given the same
      ``storage_marginal_cost``, so neither model is rewarded for cycling for
      free.  The cycling term below is the difference that remains.
    * **The same terminal condition.** A finite horizon lets an optimiser end
      with an empty battery and call it a saving.  Energy left in the tank at
      the last snapshot is credited at that hour's marginal price λ_T, so both
      books close at the same value.  The initial SoC is identical in both
      models, so it cancels.

    Every cost is linear in energy, so the six terms must add up to the gap.
    """
    spec = run.get("spec", spec)
    df = run["df"]
    dt = float(run["dt"])
    price = df["price_usd_mwh"].to_numpy(dtype=float)
    carbon_t = df["carbon_kg_mwh"].to_numpy(dtype=float) / 1000.0
    gas_cost = float(spec.gas_cost_usd_kwh) * 1000.0
    tau = float(spec.carbon_price)
    gas_co2 = float(spec.gas_co2)
    mc = float(opt.get("storage_marginal_cost", getattr(spec, "storage_marginal_cost", 0.0)))

    v_grid = df["grid_mw"].clip(lower=0).to_numpy(dtype=float)
    v_export = (-df["grid_mw"]).clip(lower=0).to_numpy(dtype=float)
    v_gas = df.get("gas_mw", pd.Series(0.0, index=df.index)).to_numpy(dtype=float)

    p_grid = opt["dispatch"].get("Grid import", pd.Series(0.0, index=df.index)).to_numpy(dtype=float)
    p_gas = opt["dispatch"].get("Gas turbine", pd.Series(0.0, index=df.index)).to_numpy(dtype=float)

    grid_term = float(((v_grid - p_grid) * price).sum() * dt)
    gas_term = float(((v_gas - p_gas) * gas_cost).sum() * dt)
    export_term = float(-(v_export * 0.35 * price).sum() * dt)
    carbon_term = float(((v_grid - p_grid) * carbon_t * tau).sum() * dt
                        + ((v_gas - p_gas) * gas_co2 * tau).sum() * dt)

    # PyPSA charges ``marginal_cost`` on *discharge*, not on the half-cycle
    # "throughput", so the term must use the same quantity the solver bills.
    discharge_v = float(df["battery_mw"].clip(lower=0).sum() * dt)
    discharge_p = float(opt.get("storage_discharge_mwh", 0.0))
    throughput_v = float(df["battery_mw"].abs().sum() * dt / 2.0)
    throughput_p = float(opt.get("battery_throughput_mwh", 0.0))
    storage_term = mc * (discharge_v - discharge_p)

    soc_v = float(df["soc_mwh"].iloc[-1])
    soc_p = float(opt["soc"].iloc[-1, 0]) if len(opt["soc"]) else 0.0
    lam_end = float(opt["prices"]["Site"].iloc[-1]) if len(opt["prices"]) else 0.0
    if not math.isfinite(lam_end):
        lam_end = float(price[-1])
    terminal_term = -lam_end * (soc_v - soc_p)

    # Fair objective: the bill *plus* the cycling cost the policy already
    # internalises.  PyPSA's bill already pays it, so both sides now price the
    # same things and the gap measures information, not preferences.
    vessim_bill = float(run["baseline"]["total_cost_usd"])
    cyc_v = mc * discharge_v
    vessim_cost = vessim_bill + cyc_v          # bill + the cycling the policy prices
    pypsa_cost = float(opt["energy_cost"] + opt["carbon_cost"])
    gap_fair = vessim_cost - pypsa_cost
    # Closed book: the stock left at the horizon is worth the same in both
    # models, so the optimiser cannot "save" by emptying the battery.
    gap = gap_fair + terminal_term
    explained = grid_term + gas_term + export_term + carbon_term + storage_term + terminal_term

    rows = [
        ("Grid energy, valued at each hour's price",
         "Σ Δt (I^V − I^P)·π_t", grid_term),
        ("Gas energy, valued at the turbine's cost",
         "Σ Δt (G^V − G^P)·c_g", gas_term),
        ("Export credit Vessim takes and PyPSA does not",
         "−Σ Δt X^V·φ·π_t", export_term),
        ("Carbon, valued at τ",
         "τ·Σ Δt (ΔI·e_t + ΔG·e_g)", carbon_term),
        ("Battery discharge, priced the same in both models",
         "c_batt·(D^V − D^P)", storage_term),
        ("Energy left in the tank at the horizon (mark-to-market)",
         "−λ_T·(SoC^V_T − SoC^P_T)", terminal_term),
    ]
    frame = pd.DataFrame(rows, columns=["Cause", "Term", "USD over the horizon"])
    frame["Share of the gap"] = (100.0 * frame["USD over the horizon"] / gap
                                 if abs(gap) > 1e-9 else np.nan)

    info = {
        "vessim_cost": vessim_cost, "pypsa_cost": pypsa_cost, "gap": gap,
        "vessim_bill": vessim_bill, "cycling_cost_v": cyc_v,
        "gap_bill": vessim_bill - float(opt["energy_cost"] + opt["carbon_cost"]),
        "gap_fair": gap_fair, "gap_closed": gap,
        "explained": explained, "residual": gap - explained,
        "grid_term": grid_term, "gas_term": gas_term, "export_term": export_term,
        "carbon_term": carbon_term, "storage_term": storage_term,
        "terminal_term": terminal_term,
        "grid_mwh_v": float(v_grid.sum() * dt), "grid_mwh_p": float(p_grid.sum() * dt),
        "gas_mwh_v": float(v_gas.sum() * dt), "gas_mwh_p": float(p_gas.sum() * dt),
        "solar_curt_v": float(df.get("curtailed_mw", pd.Series(0.0)).sum() * dt),
        "solar_curt_p": float(opt["curtailment_mwh"]),
        "throughput_v": throughput_v, "throughput_p": throughput_p,
        "discharge_v": discharge_v, "discharge_p": discharge_p,
        "soc_v": soc_v, "soc_p": soc_p, "lam_end": lam_end,
        "storage_marginal_cost": mc,
    }
    return frame, info


def gap_section(run: dict, spec, opt: dict) -> None:
    frame, info = gap_table(run, spec, opt)
    gap = info["gap"]                       # fair and closed book
    naive = info["gap_bill"]
    pct = 100.0 * naive / info["vessim_bill"] if abs(info["vessim_bill"]) > 1e-9 else float("nan")

    theme.kpis([
        ("Vessim bill", _usd(info["vessim_bill"]), "", "rules, one step at a time", ""),
        ("PyPSA bill", _usd(info["pypsa_cost"]), "", "the optimum, as solved", "good"),
        ("Naive gap", _usd(naive), "", _pct(pct) + " of the Vessim bill", "good"),
        ("Fair gap", _usd(gap), "",
         "same cycling cost, same terminal value",
         "good" if gap > 0 else "warn"),
    ])

    view = frame.copy()
    st.dataframe(
        view.style.format({"USD over the horizon": "{:,.2f}", "Share of the gap": "{:+.1f} %"}),
        hide_index=True, **theme.wide(),
    )

    sign = ("the heuristic is, on a closed book, the cheaper of the two"
            if gap < 0 else "the optimum is cheaper, as it must be")
    theme.note(
        f"The table decomposes the <b>fair</b> gap, {_usd(gap, 2)}: both models price battery "
        f"discharge at {_n(info['storage_marginal_cost'], 1)} USD/MWh and both close the horizon at "
        f"the same marginal value. Read against the naive gap of {_usd(naive, 2)}, the difference is "
        f"exactly the two conventions a headline comparison usually forgets — "
        f"{_usd(info['storage_term'], 2)} of cycling cost "
        f"({_n(info['discharge_v'], 2)} vs {_n(info['discharge_p'], 2)} MWh discharged) and "
        f"{_usd(info['terminal_term'], 2)} of energy left in the tank "
        f"({_n(info['soc_v'], 2)} vs {_n(info['soc_p'], 2)} MWh at λ_T = {_n(info['lam_end'], 2)} "
        f"USD/MWh). "
        + ("<b>On this horizon the optimum loses</b>: an LP with no terminal condition empties the "
           "battery and books the proceeds as a saving. Pricing the stock is what makes the "
           "comparison honest — and it is the strongest argument for a longer horizon or a rolling "
           "controller rather than a single 24 h solve."
           if gap < 0 else
           f"Here {sign}: the extra information is genuinely worth {_usd(gap, 2)}.")
    )
    theme.note(
        f"Quantities behind the terms: grid {_n(info['grid_mwh_v'], 2)} MWh in Vessim vs "
        f"{_n(info['grid_mwh_p'], 2)} in PyPSA; gas {_n(info['gas_mwh_v'], 2)} vs "
        f"{_n(info['gas_mwh_p'], 2)} MWh; solar spilled {_n(info['solar_curt_v'], 2)} vs "
        f"{_n(info['solar_curt_p'], 2)} MWh. The six terms add to {_usd(info['explained'], 2)} "
        f"against a gap of {_usd(gap, 2)}"
        + (" — exact, nothing is left unexplained."
           if abs(info["residual"]) < 1e-6 else
           f"; the {_usd(info['residual'], 4)} left over is solver tolerance.")
    )


# --------------------------------------------------------------------------- #
# Section 4 — every result, explained
# --------------------------------------------------------------------------- #

RESULTS = [
    ("Vessim cost", "Σ Δt (π_t·I_t − φ·π_t·X_t + c_g·G_t) + τ·Σ E_t",
     "What the rules actually spent over the horizon, carbon included. It is an outcome, not a "
     "lower bound: nothing in the simulation guarantees it is small."),
    ("PyPSA cost", "min Σ Δt [(π+τe)g + (c_g+τe_g)p]",
     "The cheapest feasible schedule under the same prices, load and equipment. By construction no "
     "other dispatch can beat it, so it is the benchmark."),
    ("Saving", "C_V − C_P",
     "The value of looking ahead and of knowing the network. Positive means the extra information "
     "PyPSA uses was worth money; it is the number a proposal should quote."),
    ("Emissions", "Σ Δt (e_t·I_t + e_g·G_t)",
     "Tonnes of CO₂ of the scheduled energy. It moves for two reasons: less grid energy, or grid "
     "energy moved to cleaner hours."),
    ("Curtailment", "Σ Δt (S_t^avail − S_t^used)",
     "Sunlight that was available and not used — usually because a line, a battery or a price "
     "blocked it. Vessim reports what its policy spilled; PyPSA curtails only when it is optimal."),
    ("Battery throughput", "½·Σ Δt |p_t^batt|",
     "Energy pushed in and out again, halved so a full cycle counts once. It measures how hard the "
     "asset works, and therefore its ageing."),
    ("λ, nodal price", "∂C*/∂L_t",
     "The marginal cost of one more MWh at that bus and hour. Equal to the cost of the marginal "
     "unit when nothing binds; higher when a line or a tank is the limit."),
    ("Line loading", "100·|f_ℓ,t| / S_ℓ",
     "Percentage of a branch's rating in use. At 100 % the line sets the price and congestion "
     "appears; above 100 % the case is infeasible, not just expensive."),
    ("Voltage (p.u.)", "|V_i| / V_nom",
     "How far the bus is from nominal. Outside 0.95–1.05 a real operator would act; in this model "
     "it is a warning from the load-flow solution."),
    ("Losses", "Σ_ℓ (p_ℓ,0 + p_ℓ,1)",
     "Energy converted to heat in the conductors. The OPF here is lossless on purpose, so that the "
     "comparison isolates scheduling; the load-flow tab shows what the real wires cost."),
    ("Balance residual", "L − (S + G + B + I)",
     "A self-check. Zero means the diagram, the table and the solver agree. Anything else is a bug "
     "and is printed instead of being hidden."),
    ("Unmet load", "Σ Δt max(0, L_t − served_t)",
     "Demand the model could not supply. It must be zero in every case here: an optimal schedule "
     "that abandons load is telling you the case is infeasible, not that it is cheap."),
]


def ac_sheet(net, result: dict, snap: dict, step: int) -> None:
    """The load flow, written out with this snapshot's numbers.

    Three equations: the complex power injected at a bus, the Newton step that
    makes the mismatch vanish, and the losses as the difference between what
    enters a branch and what leaves it.  Every symbol is substituted with a
    value from the run, so the reader can check the arithmetic by hand.
    """
    buses = result["buses"]
    bus = "Site" if "Site" in buses.index else buses.index[0]
    row = buses.loc[bus]
    v = float(row["v_pu"])
    angle = float(np.radians(row["angle_deg"]))
    p_inj = float(row.get("p_inj", 0.0))
    q_inj = float(row.get("q_inj", 0.0))

    theme.section("The load flow, with this snapshot's numbers",
                  "the same Newton-Raphson the AdjieBrotot tool runs", "Y-BUS · NEWTON")
    st.latex(r"S_i = V_i \sum_{j} Y_{ij}^{*} V_j^{*}, \qquad "
             r"P_i = V_i \sum_j V_j (G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij})")
    st.latex(r"\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = "
             r"J \begin{bmatrix} \Delta \theta \\ \Delta V \end{bmatrix}, \qquad "
             r"x^{(k+1)} = x^{(k)} + \Delta x")
    st.latex(r"P_{loss,k} = P_{k\to} + P_{k\leftarrow}, \qquad "
             r"\Delta P_{slack} = \sum_k P_{loss,k}")

    _block(
        r"S_{%s} = %.4f + j%.4f \text{ MVA}" % (bus, p_inj, q_inj),
        [
            f"V at **{bus}** = {v:.4f} p.u. · angle = {np.degrees(angle):+.3f}°",
            f"injected active power = {p_inj:+.4f} MW, reactive = {q_inj:+.4f} MVAr",
            f"mismatch after {result['iterations']} iterations = "
            f"{result['max_mismatch']:.2e} p.u.",
        ],
        "The solver stops when the largest nodal mismatch is below the tolerance: it solves "
        "Kirchhoff, not an objective.",
        "good" if result["converged"] else "bad",
    )
    _block(
        r"\sum_k P_{loss,k} = %.6f \text{ MW}" % float(result["summary"]["losses_mw"]),
        [
            f"losses = {float(result['summary']['losses_mw']):.6f} MW "
            f"({float(result['summary'].get('losses_pct', 0.0)):.4f} % of generation)",
            f"reactive losses = {float(result['summary']['losses_mvar']):.6f} MVAr",
            "the slack absorbs load **plus** losses: "
            f"{float(buses.at['POI', 'p_gen']):.4f} MW vs Vessim's "
            f"{float(snap['grid_mw']):.4f} MW",
        ],
        "This is the only number on the dashboard that Vessim cannot produce: a copper plate has "
        "no resistance.",
        "warn",
    )
    with st.expander("The Y-bus of this snapshot", expanded=False):
        st.dataframe(result["branches"].round(4), **theme.wide(), height=240)
        st.caption("One row per branch: the Y-bus is assembled from these impedances, then the "
                   "Newton step inverts the Jacobian built from them.")


def results_section() -> None:
    frame = pd.DataFrame(RESULTS, columns=["Result", "How it is computed", "What it means"])
    st.dataframe(frame, hide_index=True, **theme.wide(), height=430)


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #


def default_params(spec, carbon_price: float) -> dict:
    """Same defaults as the PyPSA tab, so the sheet matches what the tabs show."""
    from .view_integrated import _spec_params

    battery_mw = max(spec.battery_kw / 1000.0, 1e-3)
    battery_hours = max(spec.battery_kwh / max(spec.battery_kw, 1e-3), 1e-3)
    grid_limit = max(spec.grid_import_kw / 1000.0, 1e-3)
    return _spec_params(spec, carbon_price, battery_mw, battery_hours, grid_limit,
                        grid_limit, max(spec.solar_kwp / 1000.0 * 1.2, 1e-3),
                        max(spec.gas_kw / 1000.0 * 1.2, 1e-3))


def _lambda_probe(df: pd.DataFrame, params: dict, dt: float, step: int,
                  delta: float | None = None) -> tuple[float, float]:
    """Central finite difference of the optimum with respect to one snapshot.

    ``δ`` defaults to 0.5 % of the peak load: big enough to beat the solver's
    tolerance, small enough never to change which constraint binds.  A fixed
    0.05 MW would swamp a 700 W house and be noise on a 20 MW plant.
    """
    if delta is None:
        delta = max(1e-9, 0.005 * float(df["load_mw"].max()))

    # The case's network carries one load per bus (``load_<bus>_mw``), not the
    # single ``load_mw`` of the old four-node model, so the perturbation has to
    # be spread over every load bus in proportion to what it draws.  Perturbing
    # ``load_mw`` alone changed nothing the OPF could see and the derivative
    # came out as a flat zero.
    columns = [c for c in df.columns if c.startswith("load_") and c.endswith("_mw")
               and c != "load_mw"]
    if not columns:
        columns = ["load_mw"]

    def cost_at(offset: float) -> float:
        probe = df.copy()
        shares = probe.iloc[step][columns].astype(float)
        total = float(shares.sum())
        if total > 0:
            for column, share in shares.items():
                probe.iloc[step, probe.columns.get_loc(column)] = share + offset * share / total
        res = data.site_case(probe, params, dt)
        return float(res["energy_cost"] + res["carbon_cost"])

    plus, minus = cost_at(+delta), cost_at(-delta)
    return (plus - minus) / (2.0 * delta * dt), delta


def render(case_key: str, run: dict, spec, carbon_price: float) -> None:
    """The full maths page for one case."""
    df = run["df"]
    dt = float(run["dt"])

    theme.section("The maths, line by line",
                  "same equations, same numbers, both methods", "🧮 THE MATHS")
    theme.note(
        "Everything the other tabs print is computed by the equations on this page. "
        "Pick a step and the numbers below follow it; every line can be checked by hand. "
        "Sections 1 and 2 are the two methods, section 3 is the difference between them, "
        "and section 4 defines every result the dashboard shows."
    )

    step = st.slider("Step to inspect", 0, len(df) - 1, min(int(len(df) * 0.55), len(df) - 1),
                     key=f"math_step_{case_key}")
    t = pd.Timestamp(df.index[step])
    _subtitle(f"<b>{t.strftime('%d %b %Y · %H:%M')}</b> · "
              f"carbon price τ = {_n(carbon_price, 0)} USD/t")

    # ---- section 1
    theme.section("1 · Vessim: recursive rules",
                  "a simulation applies a policy at every step and never looks back",
                  "SIMULATION")
    vessim_sheet(run, spec, step)

    # ---- section 2
    theme.section("2 · PyPSA: one linear program",
                  "the same day, solved at once, with the site as a network", "OPTIMISATION")
    params = default_params(spec, carbon_price)
    try:
        opt = data.site_case(df, params, dt)
    except Exception as exc:  # pragma: no cover - solver failure
        theme.note(f"<b>PyPSA could not solve this case:</b> {type(exc).__name__}: {exc}", "bad")
        return

    lam_check = None
    if len(df) <= 200:                       # keep the page responsive on long traces
        try:
            lam_check = _lambda_probe(df, params, dt, step)
        except Exception:                    # pragma: no cover - infeasible probe
            lam_check = None
    pypsa_sheet(opt, params, run, step, lam_check=lam_check)

    # ---- section 3
    theme.section("3 · The gap, decomposed",
                  "no residual to hide behind: the terms must add up", "C_V − C_P")
    gap_section(run, spec, opt)

    # ---- section 4
    theme.section("4 · Every result, explained",
                  "the dictionary for the rest of the dashboard", "REFERENCE")
    results_section()

    theme.note(
        "<b>How to use this page in class.</b> Move the step to an hour where the battery is "
        "charging and read the SoC recursion again; then check that λ at that hour is low. Move it "
        "to the evening peak and see both flip. The two halves of the page are the two halves of "
        "the argument: rules are local, prices are global."
    )


def step_panel(run: dict, spec, opt: dict | None, params: dict | None, step: int) -> None:
    """Compact version used by the Vessim and PyPSA tabs, inside an expander."""
    vessim_sheet(run, spec, step, compact=True)
    if opt is not None and params is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        lam_check = None
        if len(run["df"]) <= 200:
            try:
                lam_check = _lambda_probe(run["df"], params, float(run["dt"]), step)
            except Exception:                # pragma: no cover
                lam_check = None
        pypsa_sheet(opt, params, run, step, compact=True, lam_check=lam_check)
