"""The OPF on the case's *own* network — the same buses the ⚡ sheet solves.

Until now the PyPSA model of the last tab was a four-node contraction (POI,
Site, Solar, Plant).  That was defensible as a teaching model but indefensible
as a claim: the reader was asked to compare a 10-bus load flow with a 4-bus
LP and to believe they were the same plant.  They are not.

This module removes the contraction.  The optimal power flow is built on the
exact :class:`~vessim_pypsa_dashboard.vessim_system.PowerNetwork` the load-flow
tab solves — same buses, same transformers, same cables, same ratings, same
names — because it is *generated from the same function*.  What changes between
the two tabs is not the network but the question:

* the ⚡ sheet **evaluates** one recorded step (Newton–Raphson, fixed setpoints);
* this sheet **optimises** the whole day on that network (linear OPF, free
  setpoints), which is where congestion, nodal prices and curtailment appear.

The linear OPF is lossless by construction: it keeps the topology, the ratings
and the impedance (for the angle differences that price congestion) but drops
the resistive losses that ``grid.run_pf`` charges.  That single difference is
measured, not hidden — see :func:`vessim_system.loss_gap`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import grid
from .pypsa_models import SiteSpec, quiet_pypsa, _add_carriers
from .vessim_runner import CASES
from .vessim_system import site_network, system_for

#: what an unserved MWh costs the optimiser: it should be a last resort, never
#: a cheap alternative to using the network
SHED_USD_PER_MWH = 10_000.0


@dataclass
class CaseResult:
    """The optimum on the case's network: same shape as ``SiteResult``, more detail."""

    status: str
    condition: str
    objective: float
    dt_hours: float
    snapshots: pd.DatetimeIndex
    #: per role — the four columns the rest of the app compares with Vessim
    dispatch: pd.DataFrame
    #: per element of the case's network, with the case's own device names
    device: pd.DataFrame
    soc: pd.DataFrame
    prices: pd.DataFrame
    line_loading: pd.DataFrame
    branch_p0: pd.DataFrame
    runtime_s: float
    energy_cost: float = 0.0
    carbon_cost: float = 0.0
    emissions_t: float = 0.0
    curtailment_mwh: float = 0.0
    battery_throughput_mwh: float = 0.0
    unmet_mwh: float = 0.0
    buses: tuple[str, ...] = ()
    branches: tuple[str, ...] = ()
    network: object = field(default=None, repr=False)


def _snapshots(df: pd.DataFrame, dt_hours: float, start: str) -> pd.DatetimeIndex:
    if dt_hours >= 1 / 60:
        freq = f"{int(round(dt_hours * 60))}min"
    else:
        freq = f"{max(int(round(dt_hours * 3600)), 1)}s"
    return pd.date_range(start, periods=len(df), freq=freq)


def build_case_network(df: pd.DataFrame, spec: SiteSpec, dt_hours: float,
                       start: str = "2026-01-01 00:00"):
    """A PyPSA network that *is* the case's network, ready to be optimised.

    Built from ``vessim_system.site_network`` — the very function the ⚡ sheet
    calls — so the bus set and the branch parameters cannot drift apart.  The
    recorded powers are turned from setpoints into inputs: loads as time series
    at their own buses, the PV availability as ``p_max_pu``, and the grid
    connection as a generator whose marginal cost is that hour's price plus the
    carbon price.
    """
    import pypsa

    quiet_pypsa()
    case = CASES[spec.name]
    system = system_for(case)

    template = site_network({"df": df}, case, 0)     # topology + the case's names
    snapshots = _snapshots(df, dt_hours, start)
    n = grid.to_pypsa(template, snapshots=list(snapshots))
    n.sn_mva = 1.0
    n.snapshot_weightings.loc[:, :] = float(dt_hours)
    _add_carriers(n)

    grid_name = system.grid[0]
    solar_name = system.solar[0]
    gas_name = system.gas[0] if system.gas else None
    battery_name = system.battery[0]

    # ------------------------------------------------------------------ buses
    # an OPF has no slack bus: every bus is a balance equation
    n.buses["v_mag_pu_set"] = 1.0

    # ------------------------------------------------------------- generators
    for name in n.generators.index:
        n.generators.at[name, "control"] = "PQ"
        n.generators.at[name, "p_set"] = 0.0
        n.generators.at[name, "p_min_pu"] = 0.0
        n.generators.at[name, "p_max_pu"] = 1.0
        n.generators.at[name, "marginal_cost"] = 0.0
        n.generators.at[name, "marginal_cost_quadratic"] = 0.0
    n.generators.at[grid_name, "p_nom"] = max(float(spec.grid_limit_mw), 1e-9)
    n.generators.at[solar_name, "p_nom"] = max(float(system.solar[1]), 1e-9)
    if gas_name:
        n.generators.at[gas_name, "p_nom"] = max(float(system.gas[1]), 1e-9)

    carbon_series = (df["carbon_kg_mwh"].to_numpy(dtype=float) / 1000.0
                     if "carbon_kg_mwh" in df.columns
                     else np.full(len(df), float(spec.grid_carbon)))
    n.generators_t.marginal_cost[grid_name] = (
        df["price_usd_mwh"].to_numpy(dtype=float) + float(spec.carbon_price) * carbon_series)
    if gas_name:
        n.generators_t.marginal_cost[gas_name] = float(spec.gas_price) + \
            float(spec.carbon_price) * float(spec.gas_carbon)

    nameplate = max(float(system.solar[1]), 1e-9)
    available = (df["solar_mw"].to_numpy(dtype=float) / nameplate).clip(0.0, 1.0)
    n.generators_t.p_max_pu[solar_name] = available
    if not bool(spec.allow_curtailment):
        n.generators_t.p_min_pu[solar_name] = available     # must-take sun

    # ------------------------------------------------------------------ loads
    for name in n.loads.index:
        n.loads.at[name, "p_set"] = 0.0
        column = f"load_{name}_mw"
        series = (df[column].to_numpy(dtype=float) if column in df.columns
                  else np.zeros(len(df)))
        n.loads_t.p_set[name] = series

    # ---------------------------------------------------------------- storage
    n.storage_units.at[battery_name, "p_set"] = 0.0
    n.storage_units.at[battery_name, "p_nom"] = max(float(spec.battery_mw), 1e-9)
    n.storage_units.at[battery_name, "max_hours"] = max(float(spec.battery_hours), 1e-9)
    n.storage_units.at[battery_name, "efficiency_store"] = float(spec.battery_efficiency)
    n.storage_units.at[battery_name, "efficiency_dispatch"] = float(spec.battery_efficiency)
    n.storage_units.at[battery_name, "marginal_cost"] = float(spec.storage_marginal_cost)
    n.storage_units.at[battery_name, "state_of_charge_initial"] = (
        float(spec.battery_soc0) * float(spec.battery_mw) * float(spec.battery_hours))
    n.storage_units.at[battery_name, "cyclic_state_of_charge"] = bool(spec.battery_cyclic)

    # ---------------------------------------------------- the network itself
    transformer = "POI transformer"
    if transformer in n.transformers.index:
        n.transformers.at[transformer, "s_nom"] = max(float(spec.line_grid_mw), 1e-9)
    for line, value in (("Site–PV collector", spec.line_solar()),
                        ("Turbine feeder", spec.line_gas() if gas_name else None)):
        if value is not None and line in n.lines.index:
            n.lines.at[line, "s_nom"] = max(float(value), 1e-9)

    # ------------------------------------------------------- free variables
    # PyPSA >= 1.2 turns a non-null ``p_set`` into an *equality* constraint
    # (Generator-p_set, StorageUnit-p_set).  In an OPF these two components are
    # variables, not setpoints: NaN is what "free" means here, and leaving 0.0
    # would pin the connection and the battery to zero output.
    n.generators["p_set"] = np.nan
    n.storage_units["p_set"] = np.nan
    for frame in (n.generators_t.p_set, n.storage_units_t.p_set):
        if len(frame.columns):
            frame.drop(columns=list(frame.columns), inplace=True)

    # ------------------------------------------------------- load shedding
    # kept as an explicit last resort so a congested feeder reports "unserved"
    # instead of an infeasible model
    peak = n.loads_t.p_set.max().fillna(0.0)
    for bus in n.buses.index:
        n.add("Generator", f"shed: {bus}", bus=bus, carrier="shed",
              p_nom=max(float(peak.get(bus, 0.0)) * 1.2, 1e-9),
              marginal_cost=SHED_USD_PER_MWH)
    try:
        n.sanitize()
    except Exception:                                   # pragma: no cover - old PyPSA
        pass
    return n


def run_case_opf(df: pd.DataFrame, spec: SiteSpec, dt_hours: float) -> CaseResult:
    """Solve the OPF on the case's own network and collect what the page shows."""
    quiet_pypsa()
    case = CASES[spec.name]
    system = system_for(case)
    n = build_case_network(df, spec, dt_hours)
    t0 = time.perf_counter()
    status, condition = n.optimize(solver_name="highs", log_to_console=False,
                                   include_objective_constant=False)
    runtime = time.perf_counter() - t0

    generators = n.generators_t.p
    grid_name = system.grid[0]
    solar_name = system.solar[0]
    gas_name = system.gas[0] if system.gas else None
    battery_name = system.battery[0]

    def col(frame: pd.DataFrame, name: str) -> pd.Series:
        return (frame[name] if name in frame.columns
                else pd.Series(0.0, index=n.snapshots))

    grid_p = col(generators, grid_name)
    solar_p = col(generators, solar_name)
    gas_p = col(generators, gas_name) if gas_name else pd.Series(0.0, index=n.snapshots)
    shed_p = generators[[c for c in generators.columns if c.startswith("shed:")]].sum(axis=1) \
        if any(c.startswith("shed:") for c in generators.columns) \
        else pd.Series(0.0, index=n.snapshots)

    dispatch = pd.DataFrame({"Grid import": grid_p, "Solar": solar_p,
                             "Gas turbine": gas_p}, index=n.snapshots)
    device = pd.DataFrame({name: generators[name] for name in generators.columns
                           if not name.startswith("shed:")}, index=n.snapshots)
    if len(n.storage_units_t.p.columns):
        device[battery_name] = n.storage_units_t.p[battery_name]

    soc = n.storage_units_t.state_of_charge.copy()
    prices = n.buses_t.marginal_price.copy()
    lines = n.lines_t.p0.copy()
    transformers = n.transformers_t.p0.copy()
    branch_p0 = pd.concat([lines, transformers], axis=1)
    loading = pd.DataFrame(index=n.snapshots)
    for name, row in n.lines.iterrows():
        if name in branch_p0.columns:
            loading[name] = branch_p0[name].abs() / max(float(row["s_nom"]), 1e-9) * 100.0
    for name, row in n.transformers.iterrows():
        if name in branch_p0.columns:
            loading[name] = branch_p0[name].abs() / max(float(row["s_nom"]), 1e-9) * 100.0

    carbon_series = (df["carbon_kg_mwh"].to_numpy(dtype=float) / 1000.0
                     if "carbon_kg_mwh" in df.columns
                     else np.full(len(df), float(spec.grid_carbon)))
    emissions = float((grid_p.to_numpy() * carbon_series).sum() * dt_hours)
    if gas_name:
        emissions += float(gas_p.sum() * float(spec.gas_carbon) * dt_hours)

    energy_cost = float((grid_p.to_numpy() * df["price_usd_mwh"].to_numpy(dtype=float)).sum()
                        * dt_hours)
    if gas_name:
        energy_cost += float(gas_p.sum() * float(spec.gas_price) * dt_hours)
    discharged = float(n.storage_units_t.p.clip(lower=0).to_numpy().sum()) \
        if len(n.storage_units_t.p.columns) else 0.0
    energy_cost += discharged * float(spec.storage_marginal_cost) * dt_hours
    # ``carbon_cost`` is reported as its own line and is *added* to the bill by
    # every caller; it must not be netted out of the energy term, or the bill
    # silently omits the CO2 the OPF actually paid for.
    carbon_cost = emissions * float(spec.carbon_price)

    available_mwh = float((n.generators_t.p_max_pu[solar_name].to_numpy()
                           * float(n.generators.at[solar_name, "p_nom"])).sum() * dt_hours)
    curtailment = max(available_mwh - float(solar_p.sum() * dt_hours), 0.0)
    throughput = float(n.storage_units_t.p.abs().to_numpy().sum() * dt_hours / 2) \
        if len(n.storage_units_t.p.columns) else 0.0

    return CaseResult(
        status=str(status), condition=str(condition), objective=float(n.objective or 0.0),
        dt_hours=float(dt_hours), snapshots=n.snapshots,
        dispatch=dispatch, device=device, soc=soc, prices=prices,
        line_loading=loading, branch_p0=branch_p0, runtime_s=float(runtime),
        energy_cost=energy_cost, carbon_cost=carbon_cost, emissions_t=emissions,
        curtailment_mwh=curtailment, battery_throughput_mwh=throughput,
        unmet_mwh=float(shed_p.sum() * dt_hours),
        buses=tuple(n.buses.index), branches=tuple(branch_p0.columns),
        network=n,
    )
