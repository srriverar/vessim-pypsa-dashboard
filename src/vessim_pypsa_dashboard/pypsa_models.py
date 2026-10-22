"""PyPSA bridges: the reference implementation next to the teaching algorithms.

Two families of models are built here.

* **Dispatch models** (:func:`build_opf_network`) reproduce the ED/UC Simulator's
  problem in PyPSA: same buses, same line ratings, same unit data, a DC optimal
  power flow (``n.optimize``) for dispatch and a MILP when units are
  committable.  They are the "correct answer" the lambda iteration and the
  priority list are compared against.

* **Site models** (:func:`build_site_network`) take the trajectories recorded by
  Vessim (load, solar, gas limit, battery) and re-optimise the same microgrid
  with PyPSA, hour by hour, with a carbon price.  This is the
  *Vessim + PyPSA* view: Vessim produces the physics and the time series, PyPSA
  prices them.

Everything is in MW / MVAr / MWh / USD.  PyPSA's loggers are silenced so the
dashboard and the test-suite stay readable.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .grid import PowerNetwork, norm_bus_type

# --------------------------------------------------------------------------- #
# House-keeping
# --------------------------------------------------------------------------- #

_CARRIERS = {
    "AC": 0.0, "grid": 0.40, "gas": 0.35, "coal": 0.85, "diesel": 0.70,
    "solar": 0.0, "wind": 0.0, "hydro": 0.0, "battery": 0.0, "load": 0.0,
}


def quiet_pypsa() -> None:
    """Silence PyPSA, linopy and HiGHS chatter (the dashboard shows its own)."""
    for name in ("pypsa", "linopy", "highspy"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*carriers which are not defined.*")


def _add_carriers(n) -> None:
    for carrier, co2 in _CARRIERS.items():
        try:
            n.add("Carrier", carrier, co2_emissions=co2)
        except Exception:
            pass


def _no_limit(rating: float) -> bool:
    return rating is None or float(rating) <= 0.0


# --------------------------------------------------------------------------- #
# OPF models for the ED/UC view
# --------------------------------------------------------------------------- #

def build_opf_network(
    net: PowerNetwork,
    load_ts: pd.DataFrame,
    vre_pu: dict[str, pd.Series] | None = None,
    mode: str = "uc",
    hours: int = 24,
    reserve_mw: float = 0.0,
    carbon_price: float = 0.0,
    start: str = "2026-01-01 00:00",
):
    """Build the PyPSA counterpart of an ED/UC network.

    Parameters
    ----------
    net: the network (buses, lines, generators, loads, storage, links).
    load_ts: DataFrame of absolute demand in MW, one column per load, one row
        per hour.  Its index becomes the snapshots.
    vre_pu: per-generator availability series (0-1) for solar/wind farms.
    mode: ``ed`` (no minimum output), ``edm`` (minimum stable output),
        ``uc`` (mixed-integer commitment) or ``lin`` (LP relaxation of it).
    """
    import pypsa

    quiet_pypsa()
    net = net.resolve_types()
    n = pypsa.Network()
    snapshots = pd.date_range(start, periods=hours, freq="h")
    n.set_snapshots(snapshots)
    n.sn_mva = 1.0
    n.snapshot_weightings.loc[:, :] = 1.0   # ED/UC models are hourly
    _add_carriers(n)

    for name, bus in net.buses.iterrows():
        n.add("Bus", name, v_nom=float(bus["v_nom"]), carrier="AC")

    for name, row in net.lines.iterrows():
        rating = 1e4 if _no_limit(row["s_nom"]) else float(row["s_nom"])
        n.add("Line", name, bus0=row["bus0"], bus1=row["bus1"], r=float(row["r"]),
              x=float(row["x"]), s_nom=rating)

    for name, row in net.transformers.iterrows():
        n.add("Transformer", name, bus0=row["bus0"], bus1=row["bus1"],
              s_nom=float(row["s_nom"]), r=float(row["r"]), x=float(row["x"]),
              g=float(row["g"]), b=float(row["b"]), tap_ratio=float(row["tap_ratio"]),
              tap_side=int(row["tap_side"]), phase_shift=float(row["phase_shift"]),
              model=str(row["model"]))

    vre_pu = vre_pu or {}
    for name, row in net.generators.iterrows():
        p_nom = float(row["p_nom"]) or 1e-3
        carrier = str(row["carrier"] or "gas")
        co2 = _CARRIERS.get(carrier, 0.0)
        mc = float(row["marginal_cost"]) + carbon_price * co2
        committable = bool(row["committable"]) and mode in ("uc", "lin")
        # Status history: a unit that is not committable is always on, so the
        # ramp of the first snapshot must start from a running unit.  Writing
        # up_time_before=0 makes PyPSA believe it was off at t=-1 and, with a
        # start-up ramp, caps p_0 at zero (infeasible in PyPSA <= 1.2).
        init = int(row["init"])
        up_before = init * int(row["min_up_time"])
        down_before = (1 - init) * int(row["min_down_time"])
        if not committable:
            up_before, down_before = 1, 0
        elif init and up_before == 0:
            up_before = 1
        kwargs = dict(
            bus=row["bus"], p_nom=p_nom, carrier=carrier,
            marginal_cost=mc, marginal_cost_quadratic=float(row["marginal_cost_quadratic"]),
            committable=committable,
            p_min_pu=float(row["p_min_pu"]) if mode in ("edm", "uc", "lin") else 0.0,
            start_up_cost=float(row["start_up_cost"]),
            shut_down_cost=float(row["shut_down_cost"]),
            min_up_time=int(row["min_up_time"]),
            min_down_time=int(row["min_down_time"]),
            ramp_limit_up=float(row["ramp_limit_up"]),
            ramp_limit_down=float(row["ramp_limit_down"]),
            # PyPSA fills a missing ramp_limit_shut_down with 1.0, which makes
            # a start-up jump by (1 - ramp_limit_down) x p_nom; set it to the
            # normal down ramp so that units can start at their minimum.
            ramp_limit_start_up=float(row["ramp_limit_up"]),
            ramp_limit_shut_down=float(row["ramp_limit_down"]),
            up_time_before=int(up_before),
            down_time_before=int(down_before),
        )
        n.add("Generator", name, **kwargs)
        if name in vre_pu:
            n.generators_t.p_max_pu[name] = vre_pu[name].to_numpy()

    for name, row in net.loads.iterrows():
        n.add("Load", name, bus=row["bus"], p_set=0.0)
        series = load_ts[name] if name in load_ts else pd.Series(0.0, index=range(hours))
        values = series.to_numpy(dtype=float)
        if len(values) < hours:
            values = np.pad(values, (0, hours - len(values)), mode="edge")
        n.loads_t.p_set[name] = values[:hours]

    for name, row in net.storages.iterrows():
        eff = float(np.sqrt(max(float(row["efficiency_store"]) * float(row["efficiency_dispatch"]), 0.0)))
        n.add("StorageUnit", name, bus=row["bus"], p_nom=float(row["p_nom"]),
              max_hours=float(row["max_hours"]),
              efficiency_store=eff, efficiency_dispatch=eff,
              state_of_charge_initial=float(row["soc0"]) * float(row["p_nom"]) * float(row["max_hours"]),
              cyclic_state_of_charge=True,
              marginal_cost=float(row["marginal_cost"]))
        try:
            n.storage_units.at[name, "efficiency_store"] = float(row["efficiency_store"])
            n.storage_units.at[name, "efficiency_dispatch"] = float(row["efficiency_dispatch"])
        except Exception:
            pass

    for name, row in net.links.iterrows():
        n.add("Link", name, bus0=row["bus0"], bus1=row["bus1"], p_nom=float(row["p_nom"]),
              efficiency=float(row["efficiency"]), p_min_pu=-1.0)

    try:
        n.sanitize()
    except Exception:
        pass
    return n


@dataclass
class OpfResult:
    status: str
    condition: str
    objective: float
    dispatch: pd.DataFrame
    commitment: pd.DataFrame
    storage_p: pd.DataFrame
    storage_soc: pd.DataFrame
    prices: pd.DataFrame
    line_flow: pd.DataFrame
    line_loading: pd.DataFrame
    network: object
    runtime_s: float = 0.0
    solver: str = "highs"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def solve_opf(n, mode: str = "uc", solver: str = "highs", log: bool = False) -> OpfResult:
    quiet_pypsa()
    t0 = time.perf_counter()
    kwargs = dict(solver_name=solver, log_to_console=log, include_objective_constant=False)
    if mode == "lin":
        kwargs["linearized_unit_commitment"] = True
    status, condition = n.optimize(**kwargs)
    runtime = time.perf_counter() - t0

    dispatch = n.generators_t.p.copy()
    commitment = n.generators_t.status if len(getattr(n.generators_t, "status", pd.DataFrame())) else pd.DataFrame(index=n.snapshots)
    try:
        storage_p = n.storage_units_t.p.copy()
        storage_soc = n.storage_units_t.state_of_charge.copy()
    except Exception:
        storage_p = pd.DataFrame(index=n.snapshots)
        storage_soc = pd.DataFrame(index=n.snapshots)
    prices = n.buses_t.marginal_price.copy() if len(n.buses_t.marginal_price.columns) else pd.DataFrame(index=n.snapshots)
    line_flow = n.lines_t.p0.copy() if len(n.lines) else pd.DataFrame(index=n.snapshots)
    if len(n.lines):
        loading = line_flow.abs().div(n.lines.s_nom.replace(0, np.nan), axis=1) * 100
    else:
        loading = pd.DataFrame(index=n.snapshots)
    return OpfResult(
        status=status, condition=condition, objective=float(n.objective or 0.0),
        dispatch=dispatch, commitment=commitment, storage_p=storage_p,
        storage_soc=storage_soc, prices=prices, line_flow=line_flow,
        line_loading=loading, network=n, runtime_s=runtime,
    )


def price_with_fixed_commitment(n, commitment: pd.DataFrame, solver: str = "highs"):
    """Re-solve as an LP with the commitment frozen to read nodal prices.

    A MILP has no duals, so the ED/UC Simulator prices a commitment by solving
    a second, linear problem with the on/off pattern held fixed.  PyPSA can do
    the same: turn the binary variables into per-snapshot bounds.
    """
    quiet_pypsa()
    m = pypsa_network_copy(n)
    gens = list(m.generators.index)
    for g in gens:
        commit = commitment[g] if g in commitment else pd.Series(1.0, index=m.snapshots)
        p_min = float(m.generators.at[g, "p_min_pu"])
        m.generators_t.p_min_pu[g] = (commit * p_min).to_numpy()
        p_max = np.where(commit.to_numpy() > 0.5, 1.0, 0.0)
        m.generators_t.p_max_pu[g] = p_max
        m.generators.at[g, "committable"] = False
        m.generators.at[g, "p_min_pu"] = 0.0
    m.optimize(solver_name=solver, log_to_console=False, include_objective_constant=False)
    return m.buses_t.marginal_price.copy()


def pypsa_network_copy(n):
    import copy

    return copy.deepcopy(n)


# --------------------------------------------------------------------------- #
# Site models: Vessim trajectories re-optimised with PyPSA
# --------------------------------------------------------------------------- #

@dataclass
class SiteSpec:
    """Electrical parameters of the simulated site, shared by Vessim and PyPSA."""

    name: str = "site"
    grid_limit_mw: float = 1.0
    solar_mw: float = 1.0
    gas_mw: float = 0.0
    battery_mw: float = 1.0
    battery_hours: float = 2.0
    line_grid_mw: float = 1.0
    line_solar_mw: float | None = None
    line_gas_mw: float | None = None
    grid_price: float = 90.0            # USD/MWh, wholesale
    grid_carbon: float = 0.40           # tCO2/MWh
    gas_price: float = 55.0             # USD/MWh thermal? (kept as USD/MWh electric)
    gas_carbon: float = 0.35
    carbon_price: float = 0.0           # USD/tCO2
    battery_efficiency: float = 1.0   # Vessim's SimpleBattery is lossless
    # MeritOrderPolicy prices every MWh the battery discharges at
    # ``degradation_cost_usd_per_kwh`` (0.02 USD/kWh).  PyPSA must charge the
    # same, otherwise the comparison would measure a preference, not the value
    # of information.  Set to 0 for the "free cycling" sensitivity.
    storage_marginal_cost: float = 20.0
    battery_soc0: float = 0.5
    battery_cyclic: bool = False      # Vessim does not constrain the final SoC
    allow_curtailment: bool = True

    def line_solar(self) -> float:
        return self.line_solar_mw or max(self.solar_mw * 1.2, 0.1)

    def line_gas(self) -> float:
        return self.line_gas_mw or max(self.gas_mw * 1.2, 0.1)

    def marginal_costs(self) -> dict[str, float]:
        return {
            "grid": self.grid_price + self.carbon_price * self.grid_carbon,
            "gas": self.gas_price + self.carbon_price * self.gas_carbon,
            "solar": 0.0,
        }


@dataclass
class SiteResult:
    status: str
    condition: str
    objective: float
    dispatch: pd.DataFrame
    soc: pd.DataFrame
    prices: pd.DataFrame
    line_loading: pd.DataFrame
    runtime_s: float
    network: object
    spec: SiteSpec
    # derived
    energy_cost: float = 0.0
    carbon_cost: float = 0.0
    emissions_t: float = 0.0
    curtailment_mwh: float = 0.0
    battery_throughput_mwh: float = 0.0
    unmet_mwh: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def cost_breakdown(self) -> pd.DataFrame:
        return pd.DataFrame({
            "component": ["Energy", "CO₂ price", "Total"],
            "usd": [self.energy_cost, self.carbon_cost, self.energy_cost + self.carbon_cost],
        })


def build_site_network(
    df: pd.DataFrame,
    spec: SiteSpec,
    dt_hours: float,
    start: str = "2026-01-01 00:00",
):
    """Build a 4-bus site network on a Vessim trajectory.

    The trajectory needs ``load_mw`` and ``solar_mw`` columns (absolute MW).
    Buses: ``POI`` (grid import), ``Site`` (load + battery), ``PV`` and ``TG``
    (captive gas turbine) — the same names the AC network of the load-flow tab
    uses, so the two networks can be read side by side.  The three lines can be tight, which is
    what makes the PyPSA view interesting: congestion, curtailment and nodal
    prices appear, none of which a priority dispatch policy can see.
    """
    import pypsa

    quiet_pypsa()
    n = pypsa.Network()
    if dt_hours >= 1 / 60:
        freq = f"{int(round(dt_hours * 60))}min"
    else:
        freq = f"{max(int(round(dt_hours * 3600)), 1)}s"
    snapshots = pd.date_range(start, periods=len(df), freq=freq)
    n.set_snapshots(snapshots)
    n.sn_mva = 1.0
    # Every snapshot lasts ``dt_hours``: PyPSA defaults to 1 h, which would
    # make the battery four times too fast on a 15-minute trace.
    n.snapshot_weightings.loc[:, :] = float(dt_hours)
    _add_carriers(n)

    for bus in ("POI", "Site", "PV", "TG"):
        n.add("Bus", bus, carrier="AC")
    # lossless branches: Vessim treats the site as a copper plate, so the OPF
    # must not charge for losses the heuristic never pays.  x is kept so the
    # DC load flow (and the KVL constraint) has a well-defined impedance.
    n.add("Line", "POI–Site", bus0="POI", bus1="Site", x=0.02, r=0.0,
          s_nom=max(spec.line_grid_mw, 1e-3))
    n.add("Line", "Site–PV collector", bus0="Site", bus1="PV", x=0.05, r=0.0,
          s_nom=max(spec.line_solar(), 1e-3))
    if spec.gas_mw > 0:
        n.add("Line", "Turbine feeder", bus0="Site", bus1="TG", x=0.05, r=0.0,
              s_nom=max(spec.line_gas(), 1e-3))
    n.add("Link", "POI–TG", bus0="POI", bus1="TG", p_nom=0.0, efficiency=1.0)

    mc = spec.marginal_costs()
    n.add("Generator", "Grid import", bus="POI", p_nom=max(spec.grid_limit_mw, 1e-3),
          marginal_cost=mc["grid"], carrier="grid", up_time_before=1)
    if "price_usd_mwh" in df.columns:
        # the grid's marginal cost must be the price Vessim actually saw,
        # otherwise the two methods optimise different problems
        carbon_series = (df["carbon_kg_mwh"] / 1000.0 if "carbon_kg_mwh" in df.columns
                         else pd.Series(spec.grid_carbon, index=df.index))
        n.generators_t.marginal_cost["Grid import"] = (
            df["price_usd_mwh"].to_numpy(dtype=float)
            + spec.carbon_price * carbon_series.to_numpy(dtype=float))
    solar_pu = (df["solar_mw"].to_numpy() / spec.solar_mw).clip(0, 1) if spec.solar_mw > 0 else np.zeros(len(df))
    n.add("Generator", "Solar", bus="PV", p_nom=max(spec.solar_mw, 1e-3),
          marginal_cost=mc["solar"], carrier="solar", p_max_pu=1.0, up_time_before=1)
    n.generators_t.p_max_pu["Solar"] = solar_pu
    if spec.gas_mw > 0:
        n.add("Generator", "Gas turbine", bus="TG", p_nom=spec.gas_mw,
              marginal_cost=mc["gas"], carrier="gas", up_time_before=1)

    n.add("Load", "Site load", bus="Site", p_set=0.0)
    n.loads_t.p_set["Site load"] = df["load_mw"].to_numpy()
    n.add("StorageUnit", "Battery", bus="Site", p_nom=max(spec.battery_mw, 1e-3),
          max_hours=max(spec.battery_hours, 1e-3),
          efficiency_store=spec.battery_efficiency,
          efficiency_dispatch=spec.battery_efficiency,
          marginal_cost=float(spec.storage_marginal_cost),
          state_of_charge_initial=spec.battery_soc0 * spec.battery_mw * spec.battery_hours,
          cyclic_state_of_charge=bool(spec.battery_cyclic))
    try:
        n.sanitize()
    except Exception:
        pass
    return n


def run_site_optimisation(df: pd.DataFrame, spec: SiteSpec, dt_hours: float) -> SiteResult:
    quiet_pypsa()
    n = build_site_network(df, spec, dt_hours)
    t0 = time.perf_counter()
    status, condition = n.optimize(solver_name="highs", log_to_console=False,
                                   include_objective_constant=False)
    runtime = time.perf_counter() - t0

    dispatch = n.generators_t.p.copy()
    soc = n.storage_units_t.state_of_charge.copy()
    prices = n.buses_t.marginal_price.copy()
    loading = n.lines_t.p0.abs().div(n.lines.s_nom, axis=1) * 100

    grid_p = dispatch.get("Grid import", pd.Series(0.0, index=n.snapshots))
    gas_p = dispatch.get("Gas turbine", pd.Series(0.0, index=n.snapshots))
    solar_p = dispatch.get("Solar", pd.Series(0.0, index=n.snapshots))
    # ``prices`` stays the *nodal* price (the dual of the balance): that is what
    # the diagrams label and what the maths sheet differentiates.  The cost
    # accounting below needs the generator cost series, kept separately.
    marginal = n.generators_t.marginal_cost
    prices = n.buses_t.marginal_price.copy()
    energy_cost = 0.0
    for name in dispatch.columns:
        if name in getattr(marginal, "columns", []):
            cost = dispatch[name] * marginal[name].reindex(n.snapshots).fillna(0.0)
        else:
            cost = dispatch[name] * float(n.generators.at[name, "marginal_cost"])
        energy_cost += float(cost.sum() * dt_hours)
    storage_p = n.storage_units_t.p
    if len(storage_p.columns):
        discharged = storage_p.clip(lower=0).to_numpy(dtype=float)
        energy_cost += float(discharged.sum() * float(spec.storage_marginal_cost) * dt_hours)
    if "carbon_kg_mwh" in df.columns:
        carbon_t = df["carbon_kg_mwh"].to_numpy(dtype=float) / 1000.0
        grid_emissions = float((grid_p.to_numpy() * carbon_t).sum() * dt_hours)
    else:
        grid_emissions = float(grid_p.sum() * spec.grid_carbon * dt_hours)
    emissions = grid_emissions + float(gas_p.sum() * spec.gas_carbon * dt_hours)
    energy_cost -= emissions * spec.carbon_price   # report energy and CO2 separately
    carbon_cost = emissions * spec.carbon_price
    available = float((solar_pu_series(df, spec) * spec.solar_mw).sum() * dt_hours)
    used = float(solar_p.sum() * dt_hours)
    curtailment = max(available - used, 0.0)
    battery_throughput = float(n.storage_units_t.p.abs().sum().sum() * dt_hours / 2)
    served = dispatch.sum(axis=1) + n.storage_units_t.p.sum(axis=1)
    unmet = float((n.loads_t.p_set["Site load"] - served).clip(lower=0).sum() * dt_hours)

    return SiteResult(
        status=status, condition=condition, objective=float(n.objective or 0.0),
        dispatch=dispatch, soc=soc, prices=prices, line_loading=loading, runtime_s=runtime,
        network=n, spec=spec, energy_cost=energy_cost, carbon_cost=carbon_cost,
        emissions_t=emissions, curtailment_mwh=curtailment,
        battery_throughput_mwh=battery_throughput, unmet_mwh=unmet,
    )


def solar_pu_series(df: pd.DataFrame, spec: SiteSpec) -> pd.Series:
    if spec.solar_mw <= 0:
        return pd.Series(0.0, index=df.index)
    return (df["solar_mw"] / spec.solar_mw).clip(0, 1)


def carbon_tax_sweep(df: pd.DataFrame, spec: SiteSpec, dt_hours: float,
                     taxes: list[float] | None = None) -> pd.DataFrame:
    """Re-optimise the site for a range of carbon prices (the teaching lever)."""
    taxes = taxes or [0, 25, 50, 100, 150, 200, 300]
    rows = []
    for tax in taxes:
        variant = SiteSpec(**{**spec.__dict__, "carbon_price": float(tax)})
        result = run_site_optimisation(df, variant, dt_hours)
        grid_p = result.dispatch.get("Grid import", pd.Series(0.0))
        gas_p = result.dispatch.get("Gas turbine", pd.Series(0.0))
        rows.append({
            "carbon_price": float(tax),
            "cost_usd": result.energy_cost + result.carbon_cost,
            "energy_cost_usd": result.energy_cost,
            "carbon_cost_usd": result.carbon_cost,
            "emissions_t": result.emissions_t,
            "grid_mwh": float(grid_p.sum() * dt_hours),
            "gas_mwh": float(gas_p.sum() * dt_hours),
            "status": result.status,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Manual-method helpers: build the "same problem, no network" view
# --------------------------------------------------------------------------- #

def vre_availability(net: PowerNetwork, gen_profiles: dict[str, str], hours: int) -> dict[str, pd.Series]:
    from .profiles import shape

    out = {}
    for name, prof in gen_profiles.items():
        if name not in net.generators.index:
            continue
        values = shape(prof, hours)
        out[name] = pd.Series(values, index=range(hours), dtype=float).clip(0, 1)
    return out


def thermal_units(net: PowerNetwork, vre: dict[str, pd.Series]) -> pd.DataFrame:
    """Units the manual heuristics dispatch (everything that is not must-take VRE)."""
    mask = ~net.generators.index.isin(list(vre))
    return net.generators.loc[mask].copy()
