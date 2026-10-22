"""The Vessim test system, seen as a power system and as a dispatch problem.

The dashboard's other tabs start from the microgrid Vessim simulates.  These two
tabs — load flow and ED/UC — used to start from a library of textbook networks
(WSCC 9, PJM 5, …), which is exactly the disconnect a reader feels: the buses
had nothing to do with the case.  This module builds, **from the case itself**:

* :func:`system_for` — the case's own system, as equipment: voltage levels, the
  transformer at the point of interconnection, the photovoltaic collector, the
  battery converter, the turbine, and the switchboards the actors are metered
  on.  Case 05 is not case 01 scaled up: it has a 110/13.8 kV envelope, a
  dedicated 13.8/0.4 kV data-centre transformer and a turbine feeder, because
  that is the system the case describes.
* :func:`site_network` — that system at one snapshot, loaded with the powers the
  simulation recorded.  Voltages, loading and losses are then whatever the
  case's own powers produce.
* :func:`ed_at` — the economic dispatch (λ-iteration) of the case's *own*
  thermal fleet and grid connection at one snapshot, with that snapshot's price
  and carbon intensity.
* :func:`uc_day` — the commitment over the whole day: a priority list with
  minimum up and down times, dispatch economically inside each commitment.

Nothing here invents a system: the powers come from the recorded Vessim trace,
the costs from the case's own parameters, the prices from the case's own signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ed_uc import DispatchResult, economic_dispatch
from .grid import PowerNetwork


# --------------------------------------------------------------------------- #
# The case's own system
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Group:
    """A switchboard: the actors metered together, and how it is fed."""

    name: str
    kv: float
    actors: tuple[str, ...]
    transformer_mva: float = 0.0      # 0 → same voltage as the site bar, cable only
    cable_km: float = 0.05


@dataclass(frozen=True)
class SiteSystem:
    """Everything the case installs, with nameplates instead of peaks."""

    key: str
    number: str
    title: str
    mv_kv: float                      # voltage at the point of interconnection
    site_kv: float                    # voltage of the site busbar
    tx_mva: float                     # step-down transformer at the POI
    cable_km: float                   # length of a collector run
    grid: tuple[str, float]           # (name, MW of import capacity)
    solar: tuple[str, float]          # (name, MWp)
    battery: tuple[str, float, float]  # (name, MWh, MW)
    gas: tuple[str, float] | None     # (name, MW) or None
    groups: tuple[Group, ...]
    power_factor: float = 0.95

    @property
    def tan_phi(self) -> float:
        phi = np.arccos(self.power_factor)
        return float(np.tan(phi))

    def equipment(self) -> list[tuple[str, str, str]]:
        """(kind, name, rating) — what the case has, for the page header."""
        def scale(value: float, unit_mw: str, unit_kw: str) -> str:
            return (f"{value * 1000:,.3g} {unit_kw}" if value < 1.0
                    else f"{value:,.3g} {unit_mw}")

        rows = [("Connection", self.grid[0],
                 f"{scale(self.grid[1], 'MW', 'kW')} import")]
        rows.append(("Generator", self.solar[0], scale(self.solar[1], "MWp", "kWp")))
        if self.gas:
            rows.append(("Generator", self.gas[0], scale(self.gas[1], "MW", "kW")))
        rows.append(("Storage", self.battery[0],
                     f"{scale(self.battery[1], 'MWh', 'kWh')} / "
                     f"{scale(self.battery[2], 'MW', 'kW')}"))
        for group in self.groups:
            where = (f"{group.kv:g} kV unit transformers" if group.kv != self.site_kv
                     else f"{group.kv:g} kV board")
            rows.append(("Switchboard", group.name,
                         f"{where} · {group.cable_km:g} km feeder · "
                         f"{', '.join(group.actors)}"))
        return rows


SYSTEMS: dict[str, SiteSystem] = {
    "basic": SiteSystem(
        key="basic", number="01", title="Basic simulation",
        mv_kv=20.0, site_kv=0.4, tx_mva=0.003, cable_km=0.15,
        grid=("Grid connection", 0.002), solar=("Solar producer", 0.0024),
        battery=("Battery", 0.001, 0.0005), gas=None,
        groups=(Group("Consumer panel", 0.4, ("facility", "edge-node", "home-nas")),),
    ),
    "solar_battery": SiteSystem(
        key="solar_battery", number="02", title="Solar + battery",
        mv_kv=20.0, site_kv=0.4, tx_mva=0.003, cable_km=0.15,
        grid=("Grid connection", 0.0025), solar=("Campus solar", 0.0032),
        battery=("Campus battery", 0.003, 0.0015), gas=None,
        groups=(Group("Campus panel", 0.4, ("campus", "home-server", "nas-backup")),),
    ),
    "carbon_aware": SiteSystem(
        key="carbon_aware", number="03", title="Carbon-aware workload",
        mv_kv=20.0, site_kv=0.4, tx_mva=0.004, cable_km=0.15,
        grid=("Grid connection", 0.003), solar=("Rooftop solar", 0.002),
        battery=("Facility battery", 0.004, 0.002), gas=None,
        groups=(Group("Facility panel", 0.4, ("facility",)),
                Group("Computing panel", 0.4, ("training", "inference", "storage"),
                      cable_km=0.03)),
    ),
    "sil": SiteSystem(
        key="sil", number="04", title="Software-in-the-loop",
        mv_kv=20.0, site_kv=0.4, tx_mva=0.0013, cable_km=0.15,
        grid=("Grid connection", 0.001), solar=("Node solar", 0.00025),
        battery=("Node battery", 0.00002, 0.00002), gas=None,
        groups=(Group("Node panel", 0.4, ("cpu", "inference-api", "storage")),),
    ),
    "montecarlo": SiteSystem(
        key="montecarlo", number="06", title="Probabilistic campus",
        mv_kv=20.0, site_kv=0.4, tx_mva=0.32, cable_km=0.20,
        grid=("Grid connection", 0.260), solar=("Campus PV", 0.180),
        battery=("Campus battery", 0.300, 0.150), gas=None,
        groups=(Group("Campus panel", 0.4, ("campus",), cable_km=0.10),
                Group("Computing panel", 0.4, ("training", "inference", "storage"),
                      cable_km=0.04)),
    ),
    "ecopetrol": SiteSystem(
        key="ecopetrol", number="05", title="Ecopetrol industrial",
        mv_kv=110.0, site_kv=13.8, tx_mva=48.0, cable_km=1.2,
        grid=("Grid connection", 12.0), solar=("Solar farm", 25.0),
        battery=("BESS", 40.0, 10.0), gas=("Gas turbine", 20.0),
        groups=(Group("Plant panel", 13.8, ("plant",), cable_km=0.4),
                Group("Data centre", 0.4, ("gpu-training", "inference", "storage"),
                      cable_km=0.08)),
    ),
}


def system_for(spec) -> SiteSystem:
    """The system of the case, keyed by the case it belongs to."""
    try:
        return SYSTEMS[spec.key]
    except KeyError:  # pragma: no cover - a new case must be declared, not guessed
        raise KeyError(f"case {spec.key!r} has no declared test system in SYSTEMS") from None


@dataclass(frozen=True)
class Layout:
    """The electrical envelope of a site, for the sentence that states it."""

    mv_kv: float
    site_kv: float
    tx_mva: float
    cable_km: float
    power_factor: float = 0.95

    @property
    def tan_phi(self) -> float:
        phi = np.arccos(self.power_factor)
        return float(np.tan(phi))


def layout_for(spec) -> Layout:
    system = system_for(spec)
    return Layout(mv_kv=system.mv_kv, site_kv=system.site_kv, tx_mva=system.tx_mva,
                  cable_km=system.cable_km, power_factor=system.power_factor)


# --------------------------------------------------------------------------- #
# The network at one snapshot
# --------------------------------------------------------------------------- #

def actor_columns(run: dict, spec) -> dict[str, str]:
    """Map a human name to the recorded column of each actor in this case."""
    columns = {}
    for column in run["df"].columns:
        if column.startswith("load_") and column.endswith("_mw") and column != "load_mw":
            columns[column[len("load_"):-len("_mw")]] = column
    return columns


def snapshot(run: dict, spec, step: int) -> dict:
    """Every power of the case at one step, in MW, with the sign conventions named."""
    df = run["df"]
    index = int(np.clip(step, 0, len(df) - 1))
    row = df.iloc[index]
    loads = {name: float(row[column]) for name, column in actor_columns(run, spec).items()
             if column in row}
    return {
        "step": index,
        "time": df.index[index],
        "load_mw": float(row["load_mw"]),
        "solar_mw": float(row["solar_mw"]),
        "battery_mw": float(row["battery_mw"]),      # positive = discharging
        "grid_mw": float(row["grid_mw"]),            # positive = importing
        "gas_mw": float(row.get("gas_mw", 0.0)),
        "soc_mwh": float(row.get("soc_mwh", 0.0)),
        "price_usd_mwh": float(row["price_usd_mwh"]),
        "carbon_kg_mwh": float(row["carbon_kg_mwh"]),
        "actors": loads,
    }


def _feeder_rating(run: dict, actors: tuple[str, ...], floor: float) -> float:
    """A cable is rated for the peak the actors ever drew **together**.

    A feeder serves every actor behind it, so the rating is the sum of their
    individual peaks times a 1.3 margin — not the largest one.  Sizing it for
    the largest actor (the first version of this function) made a switchboard
    carrying three halls look congested whenever two of them were busy, and the
    OPF, which enforces ratings where the power flow only reports them, had to
    shed load that the network can physically deliver.
    """
    peak = 0.0
    for name in actors:
        column = f"load_{name}_mw"
        if column in run["df"].columns:
            peak += float(np.max(run["df"][column]))
    # six decimals, not four: rounding a 30 W actor to four decimals of a
    # megawatt gives a zero-rated cable, and a zero-rated cable sheds load
    return round(max(peak * 1.3, floor), 6)


def site_network(run: dict, spec, step: int) -> PowerNetwork:
    """The case's own system, loaded with the powers the simulation recorded."""
    snap = snapshot(run, spec, step)
    system = system_for(spec)
    tan_phi = system.tan_phi
    missing = set(snap["actors"]) - {a for g in system.groups for a in g.actors}
    if missing:                       # pragma: no cover - declared systems are exhaustive
        raise KeyError(f"case {spec.key!r} records actors with no switchboard: {sorted(missing)}")

    floor = max(system.solar[1], system.battery[2], system.grid[1]) * 1e-3

    def cable(name: str, bus0: str, bus1: str, kv: float, length_km: float,
              s_nom: float) -> dict:
        """A cable as ohms, not as a label: per-km impedance times its length.

        LV cables are resistive and lossy, MV cables are dominated by
        reactance — the solver must see the difference, which is why the
        length is multiplied here and not left to the reader.
        """
        r_per_km, x_per_km = (0.32, 0.08) if kv <= 1.0 else (0.16, 0.24)
        return dict(name=name, bus0=bus0, bus1=bus1, length=length_km,
                    s_nom=s_nom, r=r_per_km * length_km, x=x_per_km * length_km, b=0.0)

    buses = [
        dict(name="POI", v_nom=system.mv_kv, type="Slack", carrier="grid"),
        dict(name="Site", v_nom=system.site_kv, type="PQ", carrier="ac"),
        dict(name="PV", v_nom=system.site_kv, type="PV", v_set=1.01, carrier="solar"),
        dict(name="BESS", v_nom=system.site_kv, type="PQ", carrier="battery"),
    ]
    transformers = [
        dict(name="POI transformer", bus0="POI", bus1="Site",
             s_nom=system.tx_mva, v_nom=system.mv_kv, v_nom1=system.mv_kv,
             v_nom2=system.site_kv, x=0.10, r=0.005),
    ]
    lines = [
        cable("Site–PV collector", "Site", "PV", system.site_kv, system.cable_km,
              round(max(system.solar[1] * 1.1, floor), 6)),
        cable("Site–BESS converter", "Site", "BESS", system.site_kv, system.cable_km * 0.4,
              round(max(system.battery[2] * 1.25, floor), 6)),
    ]
    loads = []
    if system.gas:
        buses.append(dict(name="TG", v_nom=system.site_kv, type="PV", v_set=1.01,
                          carrier="gas"))
        lines.append(cable("Turbine feeder", "Site", "TG", system.site_kv,
                           system.cable_km * 0.25,
                           round(max(system.gas[1] * 1.1, floor), 6)))

    for group in system.groups:
        bus = group.name
        feeder = _feeder_rating(run, group.actors, floor)
        buses.append(dict(name=bus, v_nom=system.site_kv, type="PQ", carrier="ac"))
        if group.kv != system.site_kv:
            # a switchboard at another voltage: one cable at site voltage, then
            # a unit transformer per actor — how a data hall is actually fed
            lines.append(cable(f"Site–{bus} feeder", "Site", bus, system.site_kv,
                               group.cable_km, feeder))
            for actor in group.actors:
                mva = round(max(_feeder_rating(run, (actor,), floor) * 1.15, 0.001), 6)
                buses.append(dict(name=actor, v_nom=group.kv, type="PQ", carrier="ac"))
                transformers.append(dict(
                    name=f"{actor} transformer", bus0=bus, bus1=actor, s_nom=mva,
                    v_nom=system.site_kv, v_nom1=system.site_kv, v_nom2=group.kv,
                    x=0.060, r=0.006))
                loads.append(dict(name=actor, bus=actor,
                                  p_set=snap["actors"].get(actor, 0.0),
                                  q_set=snap["actors"].get(actor, 0.0) * tan_phi))
            continue
        if group.transformer_mva:
            transformers.append(dict(name=f"{bus} transformer", bus0="Site", bus1=bus,
                                     s_nom=group.transformer_mva, v_nom=system.site_kv,
                                     v_nom1=system.site_kv, v_nom2=group.kv,
                                     x=0.065, r=0.008))
        else:
            lines.append(cable(f"Site–{bus} feeder", "Site", bus, group.kv,
                               group.cable_km, feeder))
        if len(group.actors) == 1:
            actor = group.actors[0]
            loads.append(dict(name=actor, bus=bus, p_set=snap["actors"].get(actor, 0.0),
                              q_set=snap["actors"].get(actor, 0.0) * tan_phi))
            continue
        for actor in group.actors:                    # one meter per actor
            buses.append(dict(name=actor, v_nom=group.kv, type="PQ", carrier="ac"))
            lines.append(cable(f"{bus}–{actor}", bus, actor, group.kv, 0.005,
                               _feeder_rating(run, (actor,), floor)))
            loads.append(dict(name=actor, bus=actor, p_set=snap["actors"].get(actor, 0.0),
                              q_set=snap["actors"].get(actor, 0.0) * tan_phi))

    rows = {
        "buses": buses,
        "transformers": transformers,
        "lines": lines,
        "generators": [
            # the connection itself: slack, so the load flow reproduces Vessim's import
            dict(name=system.grid[0], bus="POI", control="Slack",
                 p_nom=max(snap["grid_mw"], system.grid[1]), carrier="grid",
                 marginal_cost=snap["price_usd_mwh"]),
            dict(name=system.solar[0], bus="PV", control="PV", p_set=snap["solar_mw"],
                 p_nom=max(system.solar[1], floor), marginal_cost=0.0, carrier="solar"),
        ],
        "loads": loads,
        "storages": [
            dict(name=system.battery[0], bus="BESS", p_set=snap["battery_mw"], q_set=0.0,
                 p_nom=max(system.battery[2], floor),
                 max_hours=system.battery[1] / max(system.battery[2], 1e-9),
                 marginal_cost=0.0),
        ],
    }
    if system.gas:
        rows["generators"].append(
            dict(name=system.gas[0], bus="TG", control="PV", p_set=snap["gas_mw"],
                 p_nom=max(system.gas[1], floor), marginal_cost=1000.0 * spec.gas_cost_usd_kwh,
                 carrier="gas"))
    stamp = (f"{snap['time']:%H:%M}" if hasattr(snap["time"], "strftime")
             else f"step {snap['step']}")
    return PowerNetwork.from_rows(
        f"{spec.title} · {stamp}", rows,
        source=(f"Case {spec.number} · {spec.title} — built from the Vessim trace at "
                f"step {snap['step']}"))


def correspondence(system: SiteSystem) -> pd.DataFrame:
    """Every bus of the AC network and the node of the OPF it survives in.

    The load-flow sheet solves the case's real network — switchboards, meters,
    the battery converter, the turbine feeder.  The PyPSA sheet solves a
    four-node contraction of it.  The two are the *same* system at two
    resolutions, and this table is the map: one row per AC bus, the node it is
    represented by, and whether it survives or is collapsed into the site bar.

    It is generated from the :class:`SiteSystem` — not hand-written — so a new
    case cannot silently disagree with its own drawing.
    """
    rows = [
        ("POI", system.mv_kv, "point of interconnection (slack)", "POI", "kept"),
        ("Site", system.site_kv, "site busbar", "Site", "kept"),
        ("PV", system.site_kv, "photovoltaic collector", "PV", "kept"),
    ]
    if system.gas:
        rows.append(("TG", system.site_kv, "captive gas turbine", "TG", "kept"))
    rows.append(("BESS", system.site_kv, "battery and its converter", "Site",
                 "collapsed — the converter is inside the site bar"))
    for group in system.groups:
        rows.append((group.name, group.kv, f"switchboard · {len(group.actors)} meter(s)",
                     "Site", "collapsed — the OPF has one load node"))
        if len(group.actors) > 1:
            for actor in group.actors:
                rows.append((actor, group.kv, "metered actor", "Site",
                             "collapsed — summed into the site load"))
    return pd.DataFrame(rows, columns=["AC bus", "kV", "it is", "PyPSA node", "relation"])


def loss_gap(run: dict, spec, samples: int = 12) -> dict:
    """What the lossless OPF does not charge for: the AC losses of the same powers.

    Both models are fed the same recorded trace, so the *energy* they dispatch
    is identical by construction.  What the four-node model drops is the
    impedance between them: the load flow of the case's own network loses a few
    per cent on the way to the meters, and the OPF prices it at zero.  This
    function measures that difference instead of asserting it away.
    """
    import numpy as _np

    from . import grid

    df = run["df"]
    steps = _np.unique(_np.linspace(0, len(df) - 1, min(int(samples), len(df))).astype(int))
    losses = []
    for step in steps:
        result = grid.run_pf(site_network(run, spec, int(step)), start="dc")
        losses.append(float(result.summary["losses_mw"]) if result.converged else float("nan"))
    dt = spec.step_s / 3600.0
    mean_mw = float(_np.nanmean(losses)) if losses else 0.0
    energy = mean_mw * len(df) * dt
    price = float((df["price_usd_mwh"] * dt).sum() / max(len(df) * dt, 1e-12))
    load = float(df["load_mw"].sum() * dt)
    return {"steps": [int(s) for s in steps], "losses_mw": losses,
            "mean_loss_mw": mean_mw, "energy_mwh": energy,
            "usd": energy * price, "price_usd_mwh": price,
            "share_pct": 100.0 * energy / max(load, 1e-12), "load_mwh": load}


# --------------------------------------------------------------------------- #
# Cost model shared by ED and UC
# --------------------------------------------------------------------------- #

def carbon_cost_usd_per_mwh(intensity_kg_per_kwh: float, carbon_price: float) -> float:
    """USD/MWh of a fuel or connection emitting ``intensity_kg_per_kwh``."""
    return float(intensity_kg_per_kwh) * float(carbon_price)


def units_at(run: dict, spec, step: int, carbon_price: float) -> pd.DataFrame:
    """The dispatchable fleet of the case at one step, priced for λ-iteration.

    The units carry the names the case uses in its diagram — ``Grid connection``,
    ``Gas turbine`` — so the two sheets and the single-line diagram speak of the
    same equipment.
    """
    snap = snapshot(run, spec, step)
    system = system_for(spec)
    rows = []
    if spec.gas_kw > 0:
        rows.append(dict(
            name=system.gas[0], bus="TG", carrier="gas",
            p_nom=float(spec.gas_kw) / 1000.0,
            p_min_pu=0.0,
            marginal_cost=1000.0 * float(spec.gas_cost_usd_kwh)
            + carbon_cost_usd_per_mwh(spec.gas_co2, carbon_price),
            # a convex heat-rate term: the dispatch problem needs a strictly
            # increasing supply curve, or the marginal unit cannot be split
            marginal_cost_quadratic=1e-3,
            min_up=2, min_down=2, start_cost=0.0,
            co2=float(spec.gas_co2),
        ))
    grid_carbon = float(snap["carbon_kg_mwh"]) / 1000.0     # this step's factor
    rows.append(dict(
        name=system.grid[0], bus="POI", carrier="grid",
        p_nom=float(spec.grid_import_kw) / 1000.0,
        p_min_pu=0.0,
        marginal_cost=float(snap["price_usd_mwh"])
        + carbon_cost_usd_per_mwh(grid_carbon, carbon_price),
        marginal_cost_quadratic=5e-4,
        min_up=1, min_down=1, start_cost=0.0,
        co2=grid_carbon,
    ))
    return pd.DataFrame(rows).set_index("name")


def demand_at(run: dict, spec, step: int) -> float:
    """What the thermal fleet and the grid have to cover, in MW.

    The battery is *not* a decision of the manual method: it follows the recorded
    Vessim schedule, so that the comparison isolates the thermal/grid split.
    """
    snap = snapshot(run, spec, step)
    return float(snap["load_mw"] - snap["solar_mw"] - snap["battery_mw"])


# --------------------------------------------------------------------------- #
# Economic dispatch at one step
# --------------------------------------------------------------------------- #

def _empty_dispatch(units: pd.DataFrame, demand: float) -> DispatchResult:
    """A no-op dispatch result with the columns the rest of the code expects."""
    frame = pd.DataFrame({
        "p_mw": np.zeros(len(units)),
        "pmin": np.zeros(len(units)),
        "pmax": units["p_nom"].to_numpy(dtype=float),
        "marginal_usd": units["marginal_cost"].to_numpy(dtype=float),
        "cost_usd": np.zeros(len(units)),
        "on": np.ones(len(units), dtype=bool),
    }, index=units.index)
    return DispatchResult(units=frame, demand=float(demand), lam=0.0, cost=0.0,
                          served=0.0, shed=0.0, spill=0.0, converged=True,
                          history=pd.DataFrame(columns=["iteration", "lambda", "total_mw",
                                                        "error_mw"]))

def ed_at(run: dict, spec, step: int, carbon_price: float) -> dict:
    """λ-iteration on the case's fleet at one snapshot, and the Vessim answer."""
    system = system_for(spec)
    snap = snapshot(run, spec, step)
    demand = demand_at(run, spec, step)
    units = units_at(run, spec, step, carbon_price)

    result = economic_dispatch(units, demand) if demand > 1e-9 else _empty_dispatch(units, demand)

    dispatch = pd.DataFrame({
        "unit": units.index,
        "mw": result.units["p_mw"].to_numpy(dtype=float),
        "marginal USD/MWh": units["marginal_cost"].to_numpy(dtype=float),
        "co2 kg/MWh": 1000.0 * units["co2"].to_numpy(dtype=float),
    })
    dispatch["cost USD/h"] = dispatch["mw"] * dispatch["marginal USD/MWh"]
    dispatch["emissions kg/h"] = dispatch["mw"] * dispatch["co2 kg/MWh"]

    names = [system.grid[0]] if spec.gas_kw <= 0 else [system.gas[0], system.grid[0]]
    vessim = pd.DataFrame({
        "unit": names,
        "mw": ([snap["gas_mw"]] if spec.gas_kw > 0 else []) + [snap["grid_mw"]],
    })
    return {
        "snapshot": snap, "steps": len(run["df"]),
        "demand_mw": demand, "result": result, "dispatch": dispatch, "vessim": vessim,
        "deltas": units, "cost_usd_h": float(result.cost),
        "emissions_kg_h": float(dispatch["emissions kg/h"].sum()),
        "vessim_cost_usd_h": float(
            (snap["gas_mw"] * (1000.0 * spec.gas_cost_usd_kwh
                               + carbon_cost_usd_per_mwh(spec.gas_co2, carbon_price))
             if spec.gas_kw > 0 else 0.0)
            + snap["grid_mw"] * (snap["price_usd_mwh"]
                                 + carbon_cost_usd_per_mwh(float(snap["carbon_kg_mwh"]) / 1000.0,
                                                           carbon_price))),
        "lam": float(result.lam),
        "converged": bool(result.converged),
        "iterations": int(len(result.history)),
        "history": result.history,
    }


# --------------------------------------------------------------------------- #
# Commitment over the day
# --------------------------------------------------------------------------- #

def uc_day(run: dict, spec, carbon_price: float, unit: str | None = None) -> dict:
    """Priority-list commitment + λ dispatch over every step of the day.

    The rule is the one the AdjieBrotot tool implements: rank the committed
    units by average cost, commit the cheapest that can cover the demand, honour
    minimum up and down times, then dispatch economically inside the commitment.
    The difference here is the system: these are the case's own units, and the
    demand is the case's own net load.
    """
    system = system_for(spec)
    unit = system.gas[0] if unit is None and system.gas else unit
    grid_name = system.grid[0]
    df = run["df"]
    steps = len(df)
    dt = float(getattr(spec, "step_s", 900)) / 3600.0
    min_up = int(max(2, round(0.5 / dt))) if spec.gas_kw > 0 else 1
    min_down = min_up
    rows = []
    run_length = 0                      # + while on, − while off
    committed = spec.gas_kw > 0         # industrial cases start with the turbine on
    for step in range(steps):
        units = units_at(run, spec, step, carbon_price)
        demand = demand_at(run, spec, step)
        gas = units.loc[unit] if unit in units.index else None
        grid = units.loc[grid_name]
        cap_grid = float(grid["p_nom"])

        if gas is not None:
            wanted = committed
            if demand > cap_grid + 1e-9:                 # the grid cannot do it alone
                wanted = True
            elif float(gas["marginal_cost"]) < float(grid["marginal_cost"]):
                wanted = True
            else:
                wanted = False
            if committed and not wanted and run_length < min_up:
                wanted = True                            # still inside the minimum up time
            if not committed and wanted and -run_length and run_length > -min_down:
                wanted = False                           # still inside the minimum down time
            committed = wanted
            run_length = run_length + 1 if committed else min(run_length - 1, -1)
            active = units.loc[[unit, grid_name]]
        else:
            active = units.loc[[grid_name]]

        result = (economic_dispatch(active, demand) if demand > 1e-9
                  else _empty_dispatch(active, demand))
        dispatch = dict(zip(active.index, result.units["p_mw"].to_numpy(dtype=float)))
        gas_mw = float(dispatch.get(unit, 0.0))
        grid_mw = float(dispatch.get(grid_name, 0.0))
        unserved = float(max(demand - gas_mw - grid_mw, 0.0))
        rows.append({
            "step": step,
            "time": df.index[step],
            "demand_mw": demand,
            "gas_mw": gas_mw,
            "grid_mw": grid_mw,
            "vessim_gas_mw": float(df["gas_mw"].iloc[step]),
            "vessim_grid_mw": float(df["grid_mw"].iloc[step]),
            "committed": bool(committed),
            "lam": float(result.lam),
            "unserved_mw": unserved,
            "cost_usd": float(result.cost) * dt,
            "emissions_kg": float(gas_mw * spec.gas_co2
                                  + grid_mw * float(df["carbon_kg_mwh"].iloc[step]) / 1000.0) * 1000.0 * dt,
            "price_usd_mwh": float(df["price_usd_mwh"].iloc[step]),
            "vessim_carbon_kg_mwh": float(df["carbon_kg_mwh"].iloc[step]),
        })
    frame = pd.DataFrame(rows).set_index("step")
    starts = int((frame["committed"] & ~frame["committed"].shift(1, fill_value=False)).sum())
    return {
        "frame": frame, "starts": starts, "dt": dt,
        "cost_usd": float(frame["cost_usd"].sum()),
        "emissions_t": float(frame["emissions_kg"].sum() / 1000.0),
        "gas_mwh": float(frame["gas_mw"].sum() * dt),
        "grid_mwh": float(frame["grid_mw"].sum() * dt),
        "unserved_mwh": float(frame["unserved_mw"].sum() * dt),
        "vessim_cost_usd": float((
            frame["vessim_gas_mw"] * (1000.0 * spec.gas_cost_usd_kwh
                                      + carbon_cost_usd_per_mwh(spec.gas_co2, carbon_price))
            + frame["vessim_grid_mw"] * (frame["price_usd_mwh"]
                                         + float(carbon_price)
                                         * frame["vessim_carbon_kg_mwh"] / 1000.0)
        ).sum() * dt),
        "vessim_emissions_t": float((frame["vessim_gas_mw"] * spec.gas_co2
                                     + frame["vessim_grid_mw"]
                                     * frame["vessim_carbon_kg_mwh"] / 1000.0).sum() * dt),
    }


# --------------------------------------------------------------------------- #
# The fleet, with what is a decision and what is not
# --------------------------------------------------------------------------- #

def fleet_table(run: dict, spec, step: int, carbon_price: float) -> pd.DataFrame:
    """Every unit of the case, its rating, and whether λ may move it.

    This is the table that keeps the two methods honest.  The λ-iteration
    decides the thermal fleet and the grid connection; the sun is a profile and
    the battery follows the schedule the case's policy produced.  Saying so,
    unit by unit, is the difference between a comparison and a coincidence.
    """
    system = system_for(spec)
    snap = snapshot(run, spec, step)
    rows = []

    def add(equipment: str, role: str, rating: str, power: float, cost: str, decision: str):
        rows.append({"Equipment": equipment, "Role": role, "Rating": rating,
                     "At this step (MW)": round(float(power), 4),
                     "Marginal (USD/MWh)": cost, "In the λ decision": decision})

    add(system.grid[0], "connection · slack", f"{system.grid[1]:,.3g} MW",
        snap["grid_mw"], f"{snap['price_usd_mwh']:,.1f} + τ·e", "yes")
    if system.gas:
        add(system.gas[0], "on-site thermal", f"{system.gas[1]:,.3g} MW", snap["gas_mw"],
            f"{1000 * spec.gas_cost_usd_kwh:,.1f} + τ·e", "yes")
    add(system.solar[0], "must-take generation", f"{system.solar[1]:,.3g} MWp",
        snap["solar_mw"], "0 (no fuel)", "no — profile")
    add(system.battery[0], "storage", f"{system.battery[1]:,.3g} MWh / "
        f"{system.battery[2]:,.3g} MW", snap["battery_mw"], "wear + opportunity",
        "no — follows the policy")
    for group in system.groups:
        for actor in group.actors:
            add(actor, f"load · {group.name}", f"{group.kv:g} kV",
                snap["actors"].get(actor, 0.0), "—", "no — demand")
    return pd.DataFrame(rows)
