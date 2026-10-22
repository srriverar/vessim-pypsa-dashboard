"""Glue between the teaching algorithms and the PyPSA benchmark.

``prepare_manual_case`` turns a network + profiles into the *residual* problem
the textbook methods solve: total demand, must-take renewables (solar/wind), a
greedy battery schedule, and the thermal units that must cover what is left.
``run_manual_ed`` and ``run_manual_uc`` then call the lambda iteration and the
priority list, and ``compare_*`` puts their answers next to PyPSA's.

The manual methods deliberately know nothing about the network.  That is the
lesson: on a copper plate they match PyPSA almost exactly; on a meshed network
with a binding line they cannot, and the gap is the value of an OPF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ed_uc import CommitmentResult, DispatchResult, economic_dispatch, unit_commitment_priority
from .grid import PowerNetwork
from .profiles import shape
from .pypsa_models import OpfResult

CO2_BY_CARRIER = {
    "coal": 0.85, "gas": 0.35, "diesel": 0.70, "grid": 0.40,
    "solar": 0.0, "wind": 0.0, "hydro": 0.0, "battery": 0.0, "": 0.0,
}


def carrier_co2(net: PowerNetwork) -> pd.Series:
    return net.generators["carrier"].map(lambda c: CO2_BY_CARRIER.get(str(c).lower(), 0.0))


# --------------------------------------------------------------------------- #
# Residual problem
# --------------------------------------------------------------------------- #

@dataclass
class ManualCase:
    hours: int
    load: pd.Series
    vre: pd.Series
    battery_mw: pd.Series
    net_load: pd.Series
    units: pd.DataFrame
    curtailment_mwh: float = 0.0
    storage_throughput_mwh: float = 0.0
    load_ts: pd.DataFrame = field(default_factory=pd.DataFrame)
    vre_pu: dict[str, pd.Series] = field(default_factory=dict)
    load_profiles: dict[str, str] = field(default_factory=dict)
    gen_profiles: dict[str, str] = field(default_factory=dict)
    vre_names: list[str] = field(default_factory=list)


def _profile_for(profiles: dict[str, str], name: str, default: str) -> str:
    return profiles.get(name, default)


def prepare_manual_case(net: PowerNetwork, info: dict, hours: int | None = None) -> ManualCase:
    hours = int(hours or info.get("hours", 24))
    load_profiles = dict(info.get("load_profiles", {}))
    gen_profiles = dict(info.get("gen_profiles", {}))

    load_ts = pd.DataFrame(index=range(hours))
    for name, row in net.loads.iterrows():
        prof = _profile_for(load_profiles, name, "sys")
        load_ts[name] = [float(row["p_set"]) * m for m in shape(prof, hours)]
    total_load = load_ts.sum(axis=1) if len(load_ts.columns) else pd.Series(0.0, index=range(hours))

    vre_pu: dict[str, pd.Series] = {}
    vre_total = pd.Series(0.0, index=range(hours))
    vre_names: list[str] = []
    for name, row in net.generators.iterrows():
        carrier = str(row["carrier"] or "").lower()
        if carrier not in ("solar", "wind"):
            continue
        prof = gen_profiles.get(name, "pv" if carrier == "solar" else "wnd")
        values = pd.Series(shape(prof, hours), dtype=float).clip(0, 1)
        vre_pu[name] = values
        vre_total = vre_total + values * float(row["p_nom"])
        vre_names.append(name)

    battery = _greedy_storage(net, total_load, vre_total, hours)
    net_load = (total_load - vre_total - battery["battery_mw"]).clip(lower=0)
    curtail = float((-(total_load - vre_total - battery["battery_mw"])).clip(lower=0).sum())
    units = net.generators.loc[~net.generators.index.isin(vre_names)].copy()

    return ManualCase(
        hours=hours, load=total_load, vre=vre_total, battery_mw=battery["battery_mw"],
        net_load=net_load, units=units, curtailment_mwh=curtail,
        storage_throughput_mwh=float(battery["battery_mw"].abs().sum() / 2),
        load_ts=load_ts, vre_pu=vre_pu, load_profiles=load_profiles,
        gen_profiles=gen_profiles, vre_names=vre_names,
    )


def _greedy_storage(net: PowerNetwork, load: pd.Series, vre: pd.Series, hours: int) -> dict[str, pd.Series]:
    """Charge from surplus renewables, discharge into the peak (rule-based)."""
    battery = pd.Series(0.0, index=range(hours))
    if net.storages.empty:
        return {"battery_mw": battery}
    row = net.storages.iloc[0]
    p_max = float(row["p_nom"])
    e_max = p_max * float(row["max_hours"])
    eff = float(np.sqrt(max(float(row["efficiency_store"]) * float(row["efficiency_dispatch"]), 1e-6)))
    soc = float(row["soc0"]) * e_max
    for t in range(hours):
        residual = float(load.iloc[t] - vre.iloc[t])
        if residual > 0:
            discharge = min(p_max, soc, residual)
            soc -= discharge
            battery.iloc[t] = discharge
        else:
            charge = min(p_max, max(e_max - soc, 0.0), -residual)
            soc += charge * eff
            battery.iloc[t] = -charge
    return {"battery_mw": battery}


# --------------------------------------------------------------------------- #
# Manual dispatch
# --------------------------------------------------------------------------- #

def run_manual_ed(case: ManualCase, hour: int, voll: float = 10_000.0) -> DispatchResult:
    hour = int(np.clip(hour, 0, case.hours - 1))
    demand = float(case.net_load.iloc[hour])
    return economic_dispatch(case.units, demand, voll=voll)


def run_manual_uc(case: ManualCase, reserve_frac: float = 0.0, voll: float = 10_000.0) -> CommitmentResult:
    return unit_commitment_priority(case.units, case.net_load.to_numpy(), reserve_frac=reserve_frac,
                                    voll=voll)


# --------------------------------------------------------------------------- #
# PyPSA side
# --------------------------------------------------------------------------- #

def opf_cost_frame(opf: OpfResult, net: PowerNetwork, dt_hours: float = 1.0) -> pd.DataFrame:
    """Energy cost, start-up cost and emissions per snapshot from a PyPSA run."""
    dispatch = opf.dispatch
    base_mc = net.generators["marginal_cost"].reindex(dispatch.columns).fillna(0.0)
    co2 = carrier_co2(net).reindex(dispatch.columns).fillna(0.0)
    energy = dispatch.mul(base_mc, axis=1).sum(axis=1) * dt_hours
    emissions = dispatch.mul(co2, axis=1).sum(axis=1) * dt_hours
    startup = pd.Series(0.0, index=dispatch.index)
    if opf.commitment is not None and not opf.commitment.empty:
        status = opf.commitment.reindex(columns=dispatch.columns).fillna(0.0)
        previous = status.shift(1)
        init = net.generators.get("init", pd.Series(1, index=net.generators.index))
        init = init.reindex(dispatch.columns).fillna(1).astype(float)
        starting = (status > 0.5) & ((previous < 0.5) | previous.isna())
        # a unit that was already running before the horizon is not a start-up
        starting.iloc[0] = (status.iloc[0] > 0.5) & (init < 0.5)
        costs = net.generators["start_up_cost"].reindex(dispatch.columns).fillna(0.0)
        startup = starting.astype(float).mul(costs, axis=1).sum(axis=1)
    return pd.DataFrame({
        "energy_cost_usd": energy,
        "startup_cost_usd": startup,
        "emissions_t": emissions,
        "total_cost_usd": energy + startup,
    })


def opf_summary(opf: OpfResult, net: PowerNetwork, dt_hours: float = 1.0) -> dict:
    frame = opf_cost_frame(opf, net, dt_hours)
    prices = opf.prices
    demand = None
    try:
        demand = opf.network.loads_t.p_set.sum(axis=1)
    except Exception:
        pass
    avg_price = float("nan")
    if demand is not None and len(prices.columns):
        weights = demand.to_numpy(dtype=float)
        weighted = prices.mean(axis=1).to_numpy(dtype=float) * weights
        avg_price = float(weighted.sum() / weights.sum()) if weights.sum() else float("nan")
    return {
        "total_cost_usd": float(frame["total_cost_usd"].sum()),
        "energy_cost_usd": float(frame["energy_cost_usd"].sum()),
        "startup_cost_usd": float(frame["startup_cost_usd"].sum()),
        "emissions_t": float(frame["emissions_t"].sum()),
        "avg_price_usd_mwh": avg_price,
        "committed_hours": int(opf.commitment.to_numpy().sum()) if opf.commitment is not None and not opf.commitment.empty else 0,
        "objective": opf.objective,
    }


def manual_summary(uc: CommitmentResult, net: PowerNetwork) -> dict:
    co2 = carrier_co2(net).reindex(uc.dispatch.columns).fillna(0.0)
    emissions = (uc.dispatch.mul(co2, axis=1)).sum().sum()
    return {
        "total_cost_usd": uc.total_cost,
        "energy_cost_usd": uc.energy_cost,
        "startup_cost_usd": uc.startup_cost,
        "emissions_t": float(emissions),
        "avg_price_usd_mwh": float(uc.hourly["price_usd_mwh"].mean()),
        "committed_hours": uc.committed_hours,
        "shed_mwh": uc.shed_mwh,
        "spill_mwh": uc.spill_mwh,
    }


def comparison_table(manual: dict, optimal: dict) -> pd.DataFrame:
    rows = []
    labels = {
        "total_cost_usd": ("Total operating cost (USD)", "USD"),
        "energy_cost_usd": ("Energy cost (USD)", "USD"),
        "startup_cost_usd": ("Start-up cost (USD)", "USD"),
        "emissions_t": ("Emissions (tCO₂)", "t"),
        "avg_price_usd_mwh": ("Average price (USD/MWh)", "USD/MWh"),
        "committed_hours": ("Committed unit-hours", "h"),
    }
    for key, (label, unit) in labels.items():
        m = manual.get(key, float("nan"))
        o = optimal.get(key, float("nan"))
        delta = (m - o) if pd.notna(m) and pd.notna(o) else float("nan")
        pct = 100 * delta / o if o not in (0, float("nan")) and pd.notna(o) else float("nan")
        rows.append({"metric": label, "manual": m, "pypsa": o, "delta": delta,
                     "delta_pct": pct, "unit": unit})
    return pd.DataFrame(rows)


def dispatch_comparison(uc: CommitmentResult, opf: OpfResult) -> pd.DataFrame:
    """Hourly manual vs PyPSA total dispatch (thermal only, MW)."""
    manual = uc.dispatch.sum(axis=1)
    optimal = opf.dispatch.sum(axis=1)
    n = min(len(manual), len(optimal))
    return pd.DataFrame({
        "hour": np.arange(n),
        "manual_mw": manual.to_numpy()[:n],
        "pypsa_mw": optimal.to_numpy()[:n],
    })


def unit_comparison(uc: CommitmentResult, opf: OpfResult, hour: int) -> pd.DataFrame:
    manual = uc.dispatch.iloc[hour] if hour < len(uc.dispatch) else pd.Series(dtype=float)
    optimal = opf.dispatch.iloc[hour] if hour < len(opf.dispatch) else pd.Series(dtype=float)
    index = sorted(set(manual.index) | set(optimal.index))
    frame = pd.DataFrame({
        "manual_mw": manual.reindex(index).fillna(0.0),
        "pypsa_mw": optimal.reindex(index).fillna(0.0),
    })
    frame["delta_mw"] = frame["manual_mw"] - frame["pypsa_mw"]
    return frame


# --------------------------------------------------------------------------- #
# Vessim vs PyPSA baseline comparison
# --------------------------------------------------------------------------- #

def vessim_baseline_metrics(run, spec, dt_hours: float) -> dict:
    """Costs and energy of a raw Vessim run, in the same units as PyPSA's."""
    df = run.df
    grid = df["grid_mw"].clip(lower=0)
    export = (-df["grid_mw"]).clip(lower=0)
    gas = df.get("gas_mw", pd.Series(0.0, index=df.index))
    # CaseSpec.gas_cost_usd_kwh is USD/kWh -> USD/MWh for a MW trajectory
    energy_cost = float((grid * df["price_usd_mwh"] * dt_hours).sum()
                        + (gas * spec.gas_cost_usd_kwh * 1000.0 * dt_hours).sum()
                        - (export * df["price_usd_mwh"] * 0.35 * dt_hours).sum())
    emissions = float((grid * df["carbon_kg_mwh"] * dt_hours).sum() / 1000.0
                      + (gas * spec.gas_co2 * dt_hours).sum())
    return {
        "total_cost_usd": energy_cost + emissions * spec.carbon_price,
        "energy_cost_usd": energy_cost,
        "carbon_cost_usd": emissions * spec.carbon_price,
        "emissions_t": emissions,
        "grid_mwh": float(grid.sum() * dt_hours),
        "export_mwh": float(export.sum() * dt_hours),
        "gas_mwh": float(gas.sum() * dt_hours),
        "solar_mwh": float(df["solar_mw"].sum() * dt_hours),
        "peak_grid_mw": float(grid.max()),
        "battery_throughput_mwh": float(df["battery_mw"].abs().sum() * dt_hours / 2),
        "curtailment_mwh": 0.0,
    }


def site_metrics(result) -> dict:
    """Costs and energy of a PyPSA site optimisation (MW trajectories)."""
    snapshots = result.network.snapshots
    if len(snapshots) > 1:
        dt = float(pd.Timedelta(snapshots[1] - snapshots[0]).total_seconds()) / 3600.0
    else:
        dt = 1.0
    grid = result.dispatch.get("Grid import", pd.Series(0.0, index=result.dispatch.index))
    gas = result.dispatch.get("Gas turbine", pd.Series(0.0, index=result.dispatch.index))
    solar = result.dispatch.get("Solar", pd.Series(0.0, index=result.dispatch.index))
    return {
        "total_cost_usd": result.energy_cost + result.carbon_cost,
        "energy_cost_usd": result.energy_cost,
        "carbon_cost_usd": result.carbon_cost,
        "emissions_t": result.emissions_t,
        "grid_mwh": float(grid.sum() * dt),
        "gas_mwh": float(gas.sum() * dt),
        "solar_mwh": float(solar.sum() * dt),
        "peak_grid_mw": float(grid.max()),
        "battery_throughput_mwh": result.battery_throughput_mwh,
        "curtailment_mwh": result.curtailment_mwh,
    }
