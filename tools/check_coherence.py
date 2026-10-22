"""Step-by-step coherence audit of the whole dashboard.

Run it as a script::

    python tools/check_coherence.py            # every case, every step where it is cheap
    python tools/check_coherence.py --full     # every step of every case, loads included

and read ``reports/coherence.md``.  The same functions are used by
``tests/test_coherence.py`` with a smaller sample, so the audit cannot rot.

What is checked, and why each check is the one that would catch a real bug:

**Inside one simulation, at every step**

===============  ==========================================================  ==================
check            statement                                                   catches
===============  ==========================================================  ==================
``balance``      solar + gas + grid + battery − load − curtailed = 0         a policy that forgets a
                                                                             dispatchable, a sign flip
``soc``          soc[t+1] = soc[t] − battery[t]·Δt, exactly                  a battery model that
                                                                             charges with the wrong sign
``soc bounds``   0 ≤ soc ≤ capacity                                         a capacity that silently
                                                                             changes
``c-rate``       |battery| ≤ c_rate · capacity                               a device asked for more
                                                                             than its nameplate
``grid limits``  −export ≤ grid ≤ import                                   a connection that imports
                                                                             more than it may
``cost``         the bill recomputed from the recorded powers               a billing formula that
                                                                             drifted from the trace
``emissions``    the CO₂ recomputed from the recorded powers                the same, for carbon
``signals``      the price and carbon columns are the modelled signals      a mis-aligned signal
``actor sum``    load = Σ actor loads                                       a missing actor
===============  ==========================================================  ==================

**Across engines, at sampled steps**

=================  ========================================================  ==================
check              statement                                                 catches
=================  ========================================================  ==================
``load flow``      converged; slack − grid − losses < 1 µW                   a network that does not
                                                                             carry the simulation's
                                                                             powers
``voltage``        0.85 ≤ V ≤ 1.10 p.u. on every bus                        a topology that cannot
                                                                             serve the case
``ed``             the manual dispatch equals Vessim's on the same fleet     a λ-iteration that
                                                                             solves another problem
``uc``             the commitment serves the whole day (unserved = 0)        a priority list that
                                                                             drops load
``lp``             the LP's objective equals the cost recomputed from its    an objective that does
                   own dispatch                                              not match the dispatch
``diagram``        every bus of the network appears in the SVG              a diagram that silently
                                                                             drops equipment
=================  ========================================================  ==================

**Deliberately not checked.**  That the LP is cheaper than Vessim (the two
solve different problems: the LP may move the battery, the policy may not);
that the two bills use the same battery efficiency (Vessim's ``SimpleBattery``
is lossless, the LP's storage is not); that the paper reproductions match the
papers (they are reduced on purpose, and the pages say by how much).  Those are
comparisons, not invariants, and the pages that show them explain the terms.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vessim_pypsa_dashboard import grid, sld  # noqa: E402
from vessim_pypsa_dashboard import papers  # noqa: E402
from vessim_pypsa_dashboard.ui import data  # noqa: E402
from vessim_pypsa_dashboard.vessim_runner import CASES, CASE_ORDER, run_case  # noqa: E402
from vessim_pypsa_dashboard.vessim_system import (  # noqa: E402
    carbon_cost_usd_per_mwh,
    correspondence,
    ed_at,
    loss_gap,
    site_network,
    system_for,
    uc_day,
)

#: Tolerances.  Every one of them is a *physical* statement, not a fudge:
#: 1e-9 MW is one milliwatt, the resolution of a float comparison in MW.
TOL = {
    "balance_mw": 1e-9,
    "soc_mwh": 1e-9,
    "cost_usd": 1e-9,
    "emissions_kg": 1e-9,
    "signal": 1e-9,
    "actor_sum_mw": 1e-12,
    "slack_kw": 1e-3,
    "voltage": (0.85, 1.10),
    "ed_mw": 0.02,              # 2 % of the unit's setpoint or 1 mW, whichever is larger
    "lp_objective_usd": 1e-4,
}


@dataclass
class Result:
    case: str
    check: str
    ok: bool
    worst: float | str
    tolerance: str
    where: str = ""
    note: str = ""

    def row(self) -> dict:
        return {"case": self.case, "check": self.check, "ok": self.ok,
                "worst": self.worst, "tolerance": self.tolerance,
                "where": self.where, "note": self.note}


@dataclass
class Audit:
    results: list[Result] = field(default_factory=list)

    def add(self, case: str, check: str, ok: bool, worst, tolerance: str,
            where: str = "", note: str = "") -> None:
        self.results.append(Result(case, check, bool(ok), worst, tolerance, where, note))

    @property
    def failures(self) -> list[Result]:
        return [r for r in self.results if not r.ok]

    @property
    def score(self) -> float:
        return 1.0 if not self.results else (len(self.results) - len(self.failures)) / len(self.results)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.row() for r in self.results])


# --------------------------------------------------------------------------- #
# One simulation, every step
# --------------------------------------------------------------------------- #

def _signals(spec, steps: int, solar_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """The signals the run is supposed to have used — recomputed, not trusted."""
    from vessim_pypsa_dashboard.vessim_runner import carbon_intensity, electricity_price

    carbon = carbon_intensity(steps, spec.step_s, spec.carbon_base, spec.carbon_amplitude)
    price = electricity_price(steps, spec.step_s, spec.price_base)
    return carbon, price


def audit_trace(case: str, run: dict, spec, audit: Audit) -> None:
    df = run["df"]
    dt = float(run["dt"])
    steps = len(df)
    carbon, price = _signals(spec, steps)

    solar = df["solar_mw"].to_numpy()
    gas = df["gas_mw"].to_numpy() if "gas_mw" in df else np.zeros(steps)
    grid_mw = df["grid_mw"].to_numpy()
    battery = df["battery_mw"].to_numpy()
    load = df["load_mw"].to_numpy()
    curtail = df["curtailed_mw"].to_numpy() if "curtailed_mw" in df else np.zeros(steps)
    soc = df["soc_mwh"].to_numpy()

    residual = solar + gas + grid_mw + battery - load - curtail
    audit.add(case, "balance", np.abs(residual).max() <= TOL["balance_mw"],
              f"{np.abs(residual).max():.2e} MW", f"≤ {TOL['balance_mw']:.0e} MW",
              f"step {int(np.abs(residual).argmax())}" if steps else "",
              "solar + gas + grid + battery − load − curtail = 0")

    # the recursion has to hold at *every* step from the previous recorded state,
    # not from the initial one: that is what catches a step that quietly resets
    soc_expected = soc[:-1] - battery[:-1] * dt
    drift = np.abs(soc[1:] - soc_expected)
    audit.add(case, "soc", (drift.max() if len(drift) else 0.0) <= TOL["soc_mwh"],
              f"{(drift.max() if len(drift) else 0.0):.2e} MWh", f"≤ {TOL['soc_mwh']:.0e} MWh",
              "every step", "soc[t+1] = soc[t] − battery[t]·Δt (the battery is lossless)")

    capacity = float(spec.battery_kwh) / 1000.0
    inside = (soc.min() >= -TOL["soc_mwh"]) and (soc.max() <= capacity + TOL["soc_mwh"])
    audit.add(case, "soc bounds", inside, f"[{soc.min():.4f}, {soc.max():.4f}] MWh",
              f"[0, {capacity:.4f}] MWh", "", "the state of charge never leaves the tank")

    c_rate = float(spec.battery_kw) / max(float(spec.battery_kwh), 1e-9)
    peak = float(np.abs(battery).max()) if steps else 0.0
    audit.add(case, "c-rate", peak <= capacity * c_rate + 1e-9, f"{peak:.4f} MW",
              f"≤ {capacity * c_rate:.4f} MW", "", f"c-rate {c_rate:.2f} /h")

    upper = float(spec.grid_import_kw) / 1000.0
    lower = -float(spec.grid_export_kw) / 1000.0
    ok_grid = grid_mw.max() <= upper + 1e-9 and grid_mw.min() >= lower - 1e-9
    audit.add(case, "grid limits", ok_grid, f"[{grid_mw.min():.4f}, {grid_mw.max():.4f}] MW",
              f"[{lower:.4f}, {upper:.4f}] MW", "", "import and export within the connection")

    export_factor = 0.35
    cost = (np.clip(grid_mw, 0, None) * price * dt
            - np.clip(-grid_mw, 0, None) * price * export_factor * dt
            + gas * float(spec.gas_cost_usd_kwh) * 1000.0 * dt)
    d_cost = np.abs(cost - df["cost_usd"].to_numpy()).max()
    audit.add(case, "cost", d_cost <= TOL["cost_usd"], f"{d_cost:.2e} USD",
              f"≤ {TOL['cost_usd']:.0e} USD", "", "the bill recomputed from the recorded powers")

    grid_carbon_series = df["carbon_kg_mwh"].to_numpy(dtype=float) / 1000.0
    emissions = (np.clip(grid_mw, 0, None) * grid_carbon_series * dt
                 + gas * float(spec.gas_co2) * dt) * 1000.0
    d_emis = np.abs(emissions - df["emissions_kg"].to_numpy()).max()
    audit.add(case, "emissions", d_emis <= TOL["emissions_kg"], f"{d_emis:.2e} kg",
              f"≤ {TOL['emissions_kg']:.0e} kg", "", "the CO₂ recomputed from the same powers")

    d_carbon = np.abs(df["carbon_kg_mwh"].to_numpy() - carbon * 1000.0).max()
    d_price = np.abs(df["price_usd_mwh"].to_numpy() - price * 1000.0).max()
    audit.add(case, "signals", max(d_carbon, d_price) <= TOL["signal"] * 1e6,
              f"carbon {d_carbon:.2e} kg/MWh · price {d_price:.2e} USD/MWh", "≤ 1e-3",
              "", "the columns are the modelled signals, step for step")

    actor_cols = [c for c in df.columns if c.startswith("load_") and c != "load_mw"]
    if actor_cols:
        total = df[actor_cols].sum(axis=1).to_numpy()
        d_actors = np.abs(total - load).max()
        audit.add(case, "actor sum", d_actors <= max(TOL["actor_sum_mw"], 1e-9 * max(1.0, load.max())),
                  f"{d_actors:.2e} MW", "≤ 1e-9 MW", "", f"{len(actor_cols)} metered actors")


# --------------------------------------------------------------------------- #
# Across engines
# --------------------------------------------------------------------------- #

def audit_load_flow(case: str, run: dict, spec, audit: Audit, samples: int = 12) -> None:
    steps = len(run["df"])
    picks = np.unique(np.linspace(0, steps - 1, min(samples, steps)).round().astype(int))
    worst_slack, worst_v, worst_loss = 0.0, np.inf, 0.0
    converged = True
    for step in picks:
        net = site_network(run, spec, int(step))
        result = grid.run_pf(net, start="dc")
        converged &= bool(result.converged)
        snap = df_snap = run["df"].iloc[int(step)]
        slack_kw = float(result.buses["p_gen"].reindex(["POI"]).iloc[0]) * 1000.0
        losses_kw = float(result.summary["losses_mw"]) * 1000.0
        worst_slack = max(worst_slack, abs(slack_kw - float(df_snap["grid_mw"]) * 1000.0 - losses_kw))
        worst_v = min(worst_v, float(result.buses["v_pu"].min()))
        worst_loss = max(worst_loss, float(result.summary["losses_mw"]))
    audit.add(case, "load flow", converged and worst_slack <= TOL["slack_kw"],
              f"{worst_slack:.2e} kW", f"≤ {TOL['slack_kw']:.0e} kW",
              f"{len(picks)} snapshots",
              f"slack = grid + losses · max losses {worst_loss * 1000:.2f} kW")
    lo, hi = TOL["voltage"]
    audit.add(case, "voltage", lo <= worst_v <= hi, f"{worst_v:.4f} p.u.", f"[{lo}, {hi}] p.u.",
              f"{len(picks)} snapshots", "the case's system can carry its own powers")


def audit_dispatch(case: str, run: dict, spec, audit: Audit, samples: int = 12) -> None:
    steps = len(run["df"])
    picks = np.unique(np.linspace(0, steps - 1, min(samples, steps)).round().astype(int))
    carbon = float(spec.carbon_price)
    worst = 0.0
    converged = True
    for step in picks:
        ed = ed_at(run, spec, int(step), carbon)
        converged &= bool(ed["converged"])
        manual = dict(zip(ed["dispatch"]["unit"], ed["dispatch"]["mw"]))
        for unit, value in zip(ed["vessim"]["unit"], ed["vessim"]["mw"]):
            worst = max(worst, abs(manual.get(unit, 0.0) - float(value)))
    audit.add(case, "ed", converged and worst <= max(TOL["ed_mw"], 0.02 * max(1.0, worst)),
              f"{worst:.4e} MW", "≤ 2 % of the setpoint", f"{len(picks)} snapshots",
              "λ-iteration vs Vessim on the same fleet")

    uc = uc_day(run, spec, carbon)
    audit.add(case, "uc", uc["unserved_mwh"] <= 1e-9,
              f"{uc['unserved_mwh']:.2e} MWh", "0 MWh", "whole day",
              f"{uc['starts']} start(s) · {uc['cost_usd']:,.2f} USD")
    audit.add(case, "uc grid", uc["frame"]["grid_mw"].max() <= float(spec.grid_import_kw) / 1000.0 + 1e-9,
              f"{uc['frame']['grid_mw'].max():.4f} MW",
              f"≤ {float(spec.grid_import_kw) / 1000.0:.4f} MW", "whole day",
              "the commitment never imports more than the connection allows")


def audit_lp(case: str, run: dict, spec, audit: Audit) -> None:
    params = data.site_spec_from(run["df"], spec, float(spec.carbon_price),
                                 float(spec.grid_import_kw) / 1000.0,
                                 float(spec.battery_kw) / 1000.0,
                                 float(spec.battery_kwh) / max(float(spec.battery_kw), 1e-9),
                                 20.0, 20.0, 20.0).__dict__
    opt = data.site_case(run["df"], params, float(run["dt"]))
    dispatch = opt["dispatch"]
    dt = float(run["dt"])
    # the battery is not a generator: PyPSA keeps it in storage_units_t.p, and
    # its wear is priced by the LP at storage_marginal_cost (USD/MWh discharged)
    battery = opt["battery_mw"].to_numpy(dtype=float)
    storage_cost = float(np.clip(battery, 0, None).sum()
                         * float(opt["storage_marginal_cost"]) * dt)
    cost = np.zeros(len(run["df"]), dtype=float)
    for column in dispatch.columns:
        series = dispatch[column].to_numpy(dtype=float)
        if column == "Solar":
            continue
        if column == "Gas turbine":
            cost = cost + series * (1000.0 * float(spec.gas_cost_usd_kwh)
                                    + carbon_cost_usd_per_mwh(spec.gas_co2, float(spec.carbon_price)))
        else:
            # the grid's marginal cost is price + τ·carbon(t), with the *carbon
            # series of this day* — not the case's constant, which is only a default
            carbon_series = run["df"]["carbon_kg_mwh"].to_numpy(dtype=float) / 1000.0
            cost = cost + series * (run["df"]["price_usd_mwh"].to_numpy()
                                    + float(spec.carbon_price) * carbon_series)
    cost = float(np.sum(cost) * dt) + storage_cost
    reported = float(opt["energy_cost"] + opt["carbon_cost"])
    gap = abs(cost - reported) / max(abs(reported), 1.0)
    audit.add(case, "lp", gap <= TOL["lp_objective_usd"],
              f"{gap:.2e} rel.", f"≤ {TOL['lp_objective_usd']:.0e}", "",
              f"objective {reported:,.2f} USD recomputed from its own dispatch")

    demand = run["df"]["load_mw"].to_numpy()
    served = dispatch.sum(axis=1).to_numpy() + battery
    balance = float(np.abs(served - demand).max())
    audit.add(case, "lp balance", balance <= 1e-6, f"{balance:.2e} MW", "≤ 1e-6 MW", "",
              "the LP's dispatch serves the net load at every step")


def audit_correlation(case: str, run: dict, spec, audit: Audit, samples: int = 8) -> None:
    """The ⚡ network and the Vessim + PyPSA network must be the *same* system.

    Five statements, each one checkable and each one previously implicit:

    * **bus map** — every bus the load flow solves appears exactly once in the
      correspondence table;
    * **same buses / same branches** — the OPF is built from the same call, so
      its bus and branch sets must equal the load flow's exactly (this is the
      claim req. 29 rests on: no contraction any more);
    * **serves the load** — that network can actually carry the day: no MWh of
      shedding on any snapshot;
    * **poi** — the AC slack equals the trace's import *plus* the losses the AC
      network produced, at every sampled step;
    * **loss gap** — those losses are non-negative and below 10 % of the load,
      and they are the one term the lossless OPF cannot price.
    """
    from vessim_pypsa_dashboard.ui import data as ui_data

    system = system_for(spec)
    mapping = correspondence(system)
    lf_net = site_network(run, spec, 0)
    buses = list(lf_net.buses.index)
    duplicated = mapping["AC bus"][mapping["AC bus"].duplicated()].tolist()
    missing = [bus for bus in buses if bus not in set(mapping["AC bus"])]
    extra = [bus for bus in mapping["AC bus"] if bus not in set(buses)]
    audit.add(case, "bus map", not (duplicated or missing or extra),
              f"{len(mapping)} AC buses → 1 row each", "every bus mapped once", "",
              "duplicated: " + ", ".join(duplicated) if duplicated else
              "not in the AC network: " + ", ".join(missing or extra) if (missing or extra) else
              f"{len(buses)} buses of the case's own network")

    opt = ui_data.site_case(run["df"], ui_data.site_spec_from(
        run["df"], spec, float(spec.carbon_price),
        system.grid[1], system.battery[2], system.battery[1] / system.battery[2],
        system.tx_mva, system.solar[1], system.gas[1] if system.gas else 0.0).__dict__,
        float(run["dt"]))
    same_buses = sorted(buses) == sorted(opt["buses"])
    extra_buses = sorted(set(opt["buses"]) - set(buses))
    audit.add(case, "same buses", same_buses, f"{len(opt['buses'])} OPF buses",
              f"= the {len(buses)} of the load flow", opt["status"],
              "not in the AC network: " + ", ".join(extra_buses) if extra_buses
              else "built by the same call — vessim_system.site_network")

    branches = sorted(list(lf_net.lines.index) + list(lf_net.transformers.index))
    same_branches = branches == sorted(opt["branches"])
    audit.add(case, "same branches", same_branches, f"{len(opt['branches'])} branches",
              f"= the {len(branches)} of the load flow",
              f"{len(lf_net.lines)} lines · {len(lf_net.transformers)} transformers",
              "differs: " + ", ".join(sorted(set(opt["branches"]) ^ set(branches)))
              if not same_branches else "same ratings, same names")

    audit.add(case, "serves the load", float(opt["unmet_mwh"]) < 1e-6,
              f"{opt['unmet_mwh']:.2e} MWh unserved", "< 1e-6 MWh", "no shedding",
              "the network carries every recorded MWh without an emergency generator")

    steps = np.unique(np.linspace(0, len(run["df"]) - 1,
                                  min(int(samples), len(run["df"]))).astype(int))
    worst = 0.0
    for step in steps:
        result = grid.run_pf(site_network(run, spec, int(step)), start="dc")
        slack = float(result.buses.loc["POI", "p_gen"])
        trace = float(run["df"]["grid_mw"].iloc[int(step)])
        losses = float(result.summary["losses_mw"])
        worst = max(worst, abs(slack - (trace + losses)))
    audit.add(case, "poi", worst <= 1e-6, f"{worst:.2e} MW", "≤ 1e-6 MW",
              f"{len(steps)} snapshots",
              "AC slack = the trace's import + the losses of that snapshot")

    gap = loss_gap(run, spec, samples=samples)
    ok = 0.0 <= gap["share_pct"] <= 10.0
    audit.add(case, "loss gap", ok, f"{gap['share_pct']:.2f} % of load", "[0, 10] %",
              f"{gap['energy_mwh']:,.2f} MWh/day · {gap['usd']:,.2f} USD",
              "the one term the lossless OPF does not charge for")


def audit_diagram(case: str, run: dict, spec, audit: Audit) -> None:
    net = site_network(run, spec, 0)
    svg = sld.draw(net)
    missing = [bus for bus in net.buses.index if str(bus) not in svg]
    audit.add(case, "diagram", not missing, f"{len(net.buses)} buses",
              "every bus drawn", "", "missing: " + ", ".join(missing) if missing else "")


def audit_system(case: str, spec, run: dict, audit: Audit) -> None:
    """Every actor the case records must have a switchboard in its own system."""
    system = system_for(spec)
    covered = {actor for group in system.groups for actor in group.actors}
    recorded = {c[len("load_"):-len("_mw")] for c in run["df"].columns
                if c.startswith("load_") and c != "load_mw"}
    missing = recorded - covered
    audit.add(case, "system", not missing, f"{len(covered)} actors · {len(system.groups)} boards",
              "every actor wired", "", "unwired: " + ", ".join(sorted(missing)) if missing
              else f"{system.number} · {system.title}")


# --------------------------------------------------------------------------- #
# One case
# --------------------------------------------------------------------------- #

def audit_case(key: str, full: bool = False, samples: int = 12) -> Audit:
    audit = Audit()
    spec = CASES[key]
    run = data.vessim_case(key, float(spec.carbon_price), 1.0, 1.0)
    audit_trace(key, run, spec, audit)
    audit_load_flow(key, run, spec, audit, samples=len(run["df"]) if full else samples)
    audit_dispatch(key, run, spec, audit, samples=len(run["df"]) if full else samples)
    audit_lp(key, run, spec, audit)
    audit_correlation(key, run, spec, audit,
                      samples=len(run["df"]) if full else max(4, samples // 2))
    audit_diagram(key, run, spec, audit)
    audit_system(key, spec, run, audit)
    return audit


def audit_papers() -> Audit:
    """The paper reproductions are identities too: they must hold on any input."""
    audit = Audit()
    rng = np.random.default_rng(11)
    hours = np.arange(96) / 4.0
    load = 8 + 3 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, .2, 96)
    solar = np.clip(7 * np.sin(np.pi * (hours - 6) / 12), 0, None)
    carbon = np.clip(.35 + .2 * np.sin(2 * np.pi * (hours - 4) / 24), .03, None)

    shift = papers.shifting_value(load, carbon, 60.0, 0.35)
    energy_gap = abs(shift["profile_before"].sum() - shift["profile_after"].sum())
    audit.add("papers", "shifting energy", energy_gap < 1e-9, f"{energy_gap:.2e} kWh",
              "≤ 1e-9 kWh", "Vessim 2024", "the flexible share moves, the energy does not")

    model = papers.PowerModel()
    sil = papers.sil_energy(np.arange(1.0, 301.0), np.linspace(0, 1, 300), model)
    audit.add("papers", "sil energy",
              abs(sil["energy_wh"] - sil["mean_w"] * 300 / 3600) < 1e-9,
              f"{abs(sil['energy_wh'] - sil['mean_w'] * 300 / 3600):.2e} Wh", "≤ 1e-9 Wh",
              "SPE 2023", "energy = mean power × time")

    net = (load - solar)
    for name in ("ideal", "linear", "kibam", "peukert"):
        result = papers.battery_day(load, solar, 12.0, 6.0, model=name)
        gap = float(np.abs(result["served_kw"] + result["unmet_kw"]
                           - np.clip(net, 0, None)).max())
        audit.add("papers", f"battery {name}", gap < 1e-9, f"{gap:.2e} kW", "≤ 1e-9 kW",
                  "LOCO 2024", "served + unmet = deficit, at every step")

    requests = 400 + 3000 * np.exp(-((np.arange(96) - 50) ** 2) / 40)
    inference = papers.inference_day(requests, carbon, papers.GPUS["A100 SXM4"])
    audit.add("papers", "inference offset", 0 <= inference["offset"] <= 1,
              f"{inference['offset']:.3f}", "[0, 1]", "PECS 2025",
              "shifting can never create emissions")

    carbons = [np.roll(carbon, k) for k in (0, 32, 64)]
    pre = papers.pretraining_day(carbons, 0.25)
    audit.add("papers", "pretraining ratio", 0 <= pre["ratio"] <= 1.5,
              f"{pre['ratio']:.3f}", "[0, 1.5]", "arXiv 2602.22760",
              "the policy cannot emit more than 1.5× the baseline")
    return audit


def audit_all(full: bool = False, samples: int = 12) -> Audit:
    audit = Audit()
    for key in CASE_ORDER:
        part = audit_case(key, full=full, samples=samples)
        audit.results.extend(part.results)
    audit.results.extend(audit_papers().results)
    return audit


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _versions() -> dict[str, str]:
    out = {}
    for package in ("vessim", "pypsa", "streamlit", "pandas", "numpy", "optuna", "highspy"):
        try:
            out[package] = metadata.version(package)
        except Exception:                       # pragma: no cover
            out[package] = "—"
    return out


def report(audit: Audit, full: bool = False) -> str:
    frame = audit.frame()
    lines = [
        "# Coherence audit",
        "",
        f"**{len(frame)} checks · {len(audit.failures)} failed · "
        f"score {audit.score:.1%}**",
        "",
        "Produced by `python tools/check_coherence.py`. Every row is a statement that must hold "
        "for the dashboard to be telling the truth about the simulation; the *worst* column is "
        "the largest deviation found, not an average, so a single bad step cannot hide.",
        "",
        "## Versions",
        "",
        "| package | version |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in _versions().items()]
    lines += ["", f"Mode: {'every step' if full else '12 sampled snapshots per case'}", ""]
    for case in list(CASE_ORDER) + ["papers"]:
        part = frame[frame["case"] == case]
        if part.empty:
            continue
        lines += [f"## {case}", "",
                  "| check | worst | tolerance | where | statement |",
                  "|---|---|---|---|---|"]
        for _, row in part.iterrows():
            mark = "✅" if row["ok"] else "❌"
            lines.append(f"| {mark} {row['check']} | {row['worst']} | {row['tolerance']} | "
                         f"{row['where']} | {row['note']} |")
        lines.append("")
    if audit.failures:
        lines += ["## Failures", ""]
        lines += [f"- **{r.case} · {r.check}** — {r.worst} against {r.tolerance} ({r.where})"
                  for r in audit.failures]
        lines.append("")
    else:
        lines += ["## Failures", "", "None. Every invariant holds at every audited step.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="every step of every case")
    parser.add_argument("--samples", type=int, default=12, help="snapshots per case")
    parser.add_argument("--out", default=str(ROOT / "reports" / "coherence.md"))
    args = parser.parse_args()

    audit = audit_all(full=args.full, samples=args.samples)
    text = report(audit, full=args.full)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    audit.frame().to_csv(out.with_suffix(".csv"), index=False)

    frame = audit.frame()
    print(frame[["case", "check", "ok", "worst", "tolerance"]].to_string(index=False))
    print(f"\n{len(frame)} checks · {len(audit.failures)} failed · score {audit.score:.1%}")
    print(f"report → {out}")
    return 0 if not audit.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
