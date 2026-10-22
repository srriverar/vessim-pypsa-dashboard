"""Cached glue between the Streamlit UI and the simulation library.

Every expensive call lives here behind ``st.cache_data`` so that moving a
slider only recomputes what actually changed.  Functions return plain
DataFrames and dicts: no PyPSA or Vessim object ever reaches the render layer,
which keeps the cache cheap to serialise and the session state small.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .. import grid as gridlib
from ..analysis import (
    carrier_co2,
    comparison_table,
    manual_summary,
    opf_summary,
    prepare_manual_case,
    run_manual_ed,
    run_manual_uc,
    vessim_baseline_metrics,
)
from ..case_opf import run_case_opf
from ..ed_presets import ED_PRESET_INFO, build_ed_preset
from ..grid import PowerNetwork
from ..presets import build_preset
from ..pypsa_models import (
    SiteSpec,
    build_opf_network,
    carbon_tax_sweep,
    quiet_pypsa,
    run_site_optimisation,
    solve_opf,
)
from ..vessim_runner import CASES, compare_strategies, run_case
from ..vessim_system import system_for

PARTS = ("buses", "lines", "transformers", "generators", "loads", "shunts", "storages", "links")


# --------------------------------------------------------------------------- #
# Network <-> payload
# --------------------------------------------------------------------------- #

def net_payload(net: PowerNetwork) -> dict:
    return {
        "name": net.name,
        "source": net.source,
        "parts": {part: getattr(net, part).copy() for part in PARTS},
    }


def net_from(payload: dict) -> PowerNetwork:
    return PowerNetwork(name=payload.get("name", "network"),
                        source=payload.get("source", ""),
                        **{part: payload["parts"][part] for part in PARTS})


# --------------------------------------------------------------------------- #
# Vessim
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

#: The four traces shipped with Vessim (MIT) and what this dashboard does with them.
DATASETS: dict[str, tuple[str, str]] = {
    "solar_example.csv": ("Solar trace for ten cities, 5 min, watts per kWp installed",
                          "Cases 01 and 02: the `vs.Trace` profile behind the `Producer`."),
    "load_example.csv": ("Hourly household load for eight homes (trace_a … trace_h)",
                         "Reference load shape for the residential cases."),
    "watttime_example.csv": ("WattTime carbon intensity (MOER, kg CO₂/MWh) for CAISO North, 5 min",
                             "Case 03: the carbon signal the merit-order policy chases."),
    "replay_example.csv": ("A 6 h replay: constant load, triangular solar",
                           "Case 04 and reproducible runs: the minimal `replay_csv.py` trace."),
}


@st.cache_data(show_spinner=False, max_entries=8)
def dataset(name: str) -> dict:
    """Load one of the bundled Vessim CSVs and describe it.

    ``watttime_example.csv`` ships with a corrupted header (``oootime``), so the
    first line is repaired before parsing — the same trace the Vessim example
    reads once you fix it by hand.
    """
    path = DATA_DIR / name
    if not path.exists():
        return {"frame": pd.DataFrame()}
    if name.startswith("watttime"):
        lines = path.read_text().splitlines()
        lines[0] = "time,moer_kg_mwh"
        frame = pd.read_csv(io.StringIO("\n".join(lines)))
        frame["time"] = pd.to_datetime(frame["time"])
    else:
        frame = pd.read_csv(path)
    return {"frame": frame}


def _run_payload(run) -> dict:
    """Normalise one :class:`VessimRun` into the plain dicts the UI caches."""
    dt = run.spec.step_s / 3600.0
    return {
        "key": run.key,
        "policy_kind": run.policy_kind,
        "policy_state": dict(run.policy_state or {}),
        "df": run.df,
        "metrics": run.metrics,
        "policy_log": list(run.policy_log or []),
        "runtime_s": run.runtime_s,
        "spec": run.spec,
        "baseline": vessim_baseline_metrics(run, run.spec, dt),
        "dt": dt,
    }


@st.cache_data(show_spinner=False, max_entries=48)
def vessim_case(key: str, carbon_price: float, solar_scale: float, load_scale: float) -> dict:
    return _run_payload(run_case(key, carbon_price=carbon_price, solar_scale=solar_scale,
                                 load_scale=load_scale))


@st.cache_data(show_spinner=False, max_entries=16)
def noise_sweep(key: str, carbon_price: float, solar_scale: float, load_scale: float,
                sigmas: tuple, seeds: tuple) -> pd.DataFrame:
    """Cost of a controller that does not know tomorrow's price and carbon.

    The policy sees ``(1 + σ·N(0,1))`` times the true signal; the bill is settled
    at the true prices, so every extra dollar is the price of the missing
    information — exactly the information PyPSA's LP is given for free.
    """
    rows = []
    for sigma in sigmas:
        for seed in seeds:
            run = run_case(key, carbon_price=carbon_price, solar_scale=solar_scale,
                           load_scale=load_scale, signal_noise=float(sigma), noise_seed=int(seed))
            dt = run.spec.step_s / 3600.0
            metrics = vessim_baseline_metrics(run, run.spec, dt)
            rows.append({
                "sigma": float(sigma), "seed": int(seed),
                "bill": float(metrics["total_cost_usd"]),
                "co2_t": float(metrics["emissions_t"]),
                "gas_mwh": float(metrics["gas_mwh"]),
                "grid_mwh": float(metrics["grid_mwh"]),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=32)
def policy_variant(key: str, carbon_price: float, solar_scale: float, load_scale: float,
                   kind: str, params: tuple) -> dict:
    """Run one hand-built policy on this case.

    ``kind``/``params`` describe a policy the reader composed in the page — a
    :class:`policies.MeritOrderPolicy` with their own numbers, or a
    :class:`policies.SocFloorPolicy`.  Passing the instance to
    ``run_case(policy_override=…)`` is the same door a user's own class goes
    through; nothing in the runner knows these classes exist.
    """
    from ..policies import MeritOrderPolicy, SocFloorPolicy

    if kind == "custom_merit":
        strategy, degradation, export_factor, soc_target, charge_bonus = params
        policy = MeritOrderPolicy(
            carbon_price_usd_per_t=carbon_price, strategy=strategy,
            degradation_cost_usd_per_kwh=degradation, export_price_factor=export_factor,
            soc_target=soc_target, charge_bonus_usd_per_kwh=charge_bonus,
            default_price_usd_per_kwh=CASES[key].price_base,
            default_carbon_kg_per_kwh=CASES[key].carbon_base)
    elif kind == "soc_floor":
        floor, ceiling = params
        policy = SocFloorPolicy(floor=floor, ceiling=ceiling,
                                carbon_price_usd_per_t=carbon_price,
                                default_price_usd_per_kwh=CASES[key].price_base,
                                default_carbon_kg_per_kwh=CASES[key].carbon_base)
    else:
        raise ValueError(f"unknown policy kind {kind!r}")
    return _run_payload(run_case(key, carbon_price=carbon_price, solar_scale=solar_scale,
                                 load_scale=load_scale, policy_override=policy))


@st.cache_data(show_spinner=False, max_entries=24)
def policy_matrix(key: str, carbon_price: float, solar_scale: float, load_scale: float,
                  policies: tuple) -> dict:
    """Run the same day under several dispatch policies.

    ``policies`` is a tuple of :class:`policies.PolicySpec` (frozen, so it hashes
    and the result is cached).  Everything else — load, sun, prices, carbon —
    is identical across rows: only the decision rule changes.
    """
    out = {}
    for spec in policies:
        run = run_case(key, carbon_price=carbon_price, solar_scale=solar_scale,
                       load_scale=load_scale, policy_spec=spec)
        dt = run.spec.step_s / 3600.0
        out[spec.key] = {
            "df": run.df,
            "baseline": vessim_baseline_metrics(run, run.spec, dt),
            "policy_kind": run.policy_kind,
            "policy_state": dict(run.policy_state or {}),
            "policy_log": list(run.policy_log or []),
            "dt": dt,
        }
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def optimal_case(key: str, carbon_price: float, solar_scale: float, load_scale: float) -> dict:
    """Ask PyPSA for the schedule, then make Vessim *execute* it.

    The LP plans in MW; Vessim works in W with the opposite sign for generation.
    The plan is converted field by field, handed to
    :class:`policies.OptimalPlanPolicy`, and the run that comes back is the
    optimum as the plant would live it: same schedule, real battery, real
    efficiency, real balance at every step.
    """
    from ..policies import OptimalPlanPolicy

    run = run_case(key, carbon_price=carbon_price, solar_scale=solar_scale,
                   load_scale=load_scale)
    payload = _run_payload(run)
    optimum = site_case(payload["df"], maths_default_params(run.spec, carbon_price), payload["dt"])

    steps = len(payload["df"])
    # A nano-slack on every device rating: Vessim requires the residual to be
    # exactly zero, and an LP plan that sits precisely on a limit leaves the
    # float dust of the solver with nowhere to go.
    slack = 1.0 - 1e-9
    battery_limit = max(run.spec.battery_kw, 1e-9) * 1000.0 * slack
    gas_limit = max(run.spec.gas_kw, 1e-9) * 1000.0 * slack
    import_limit = max(run.spec.grid_import_kw, 1e-9) * 1000.0 * slack
    export_limit = max(run.spec.grid_export_kw, 1e-9) * 1000.0 * slack

    def _clip(values, low, high):
        return np.clip(np.asarray(values, dtype=float), low, high)

    plan = {
        "battery": _clip(-np.asarray(optimum["battery_mw"], dtype=float) * 1e6,
                         -battery_limit, battery_limit),
        "gas": _clip(-optimum["dispatch"].get("Gas turbine",
                                              pd.Series(0.0, index=payload["df"].index)).to_numpy(float) * 1e6,
                     -gas_limit, 0.0),
        "grid": _clip(-optimum["dispatch"].get("Grid import",
                                               pd.Series(0.0, index=payload["df"].index)).to_numpy(float) * 1e6,
                      -import_limit, export_limit),
    }
    policy = OptimalPlanPolicy(plan, grid_name="grid", label=f"PyPSA plan · {run.spec.title}")
    tracked = run_case(key, carbon_price=carbon_price, solar_scale=solar_scale,
                       load_scale=load_scale, policy_override=policy)
    tracked_payload = _run_payload(tracked)
    from .maths import gap_table

    _, closed = gap_table(tracked_payload, run.spec, optimum)
    return {
        "run": tracked_payload,
        "optimum": optimum,
        "plan": plan,
        "steps": steps,
        "lp_cost": float(optimum["energy_cost"] + optimum["carbon_cost"]),
        "tracking_bill": float(tracked_payload["baseline"]["total_cost_usd"]),
        "closed_gap": float(closed["gap"]),
        "ledger": closed,
        "tracking_wh": float(policy.tracking_wh),
        "max_error_w": float(policy.max_error_w),
        "policy_state": dict(policy.state()),
    }


def maths_default_params(spec, carbon_price: float) -> dict:
    """The PyPSA defaults of the dashboard, without importing the UI layer."""
    from .maths import default_params

    return default_params(spec, carbon_price)


@st.cache_data(show_spinner="Running the study …", max_entries=4)
def study_case(study_id: str, case_key: str) -> dict:
    """Build (and cache) one study: an Optuna search or a policy comparison."""
    from ..study import dispatch_study, policy_study

    if study_id == "optuna-search":
        return policy_study(case_key)
    return dispatch_study(case_key)


@st.cache_data(show_spinner="Compiling and running your policy …", max_entries=8)
def custom_policy_run(key: str, carbon_price: float, solar_scale: float, load_scale: float,
                      source: str) -> dict:
    """Run a policy the reader wrote in the page.

    The text is compiled in a namespace that already contains ``vs`` (vessim),
    ``np`` and ``math``; it must define ``MyPolicy(vs.DispatchPolicy)``.  There
    is no import statement to write, because the point of the exercise is the
    class, not the boilerplate — and because a teaching app on the open internet
    should not run arbitrary imports.  Everything else is the real thing: the
    instance goes through the same ``run_case(policy_override=…)`` door a
    library policy uses.
    """
    import math

    import vessim

    from ..vessim_runner import run_case

    import re

    forbidden = ("open(", "eval(", "exec(", "__import__", "__builtins__", "__class__",
                 "__globals__", "__subclasses__", "__getattribute__")
    hit = next((token for token in forbidden if token in source), None)
    if hit is None and re.search(r"^\s*(?:import|from)\s+\w", source, re.MULTILINE):
        hit = "import"
    if hit:
        raise ValueError(f"`{hit.strip('(_')}` is not allowed here: `vs`, `np` and `math` are "
                         "already in scope, and this box only runs the policy class")
    namespace: dict = {"vs": vessim, "np": np, "math": math}
    exec(compile(source, "<your policy>", "exec"), namespace)  # noqa: S102 - teaching sandbox
    policy_class = namespace.get("MyPolicy")
    if not (isinstance(policy_class, type) and issubclass(policy_class, vessim.DispatchPolicy)):
        raise TypeError("the code must define `class MyPolicy(vs.DispatchPolicy)`")
    policy = policy_class()
    run = run_case(key, carbon_price=carbon_price, solar_scale=solar_scale,
                   load_scale=load_scale, policy_override=policy)
    payload = _run_payload(run)
    payload["source_lines"] = source.count("\n") + 1
    return payload


@st.cache_data(show_spinner=False, max_entries=8)
def sil_probe(steps: int = 6, step_s: float = 0.4) -> dict:
    """Run the real software-in-the-loop monitor for a few steps.

    This is the honest demonstration for the hardware-in-the-loop page: a real
    process is launched, ``os.wait4`` reports the CPU time it consumed, and the
    simulation is paced by that process.  No numbers are simulated.
    """
    import vessim as vs

    from ..sil import SoftwareInTheLoopMonitor

    signal = vs.StaticSignal(value=0.0)
    monitor = SoftwareInTheLoopMonitor(signal=signal, step_s=step_s, base_w=300.0,
                                       core_w=12.0, pue=1.2)
    monitor.start(None)
    try:
        for _ in range(int(steps)):
            monitor.step(None, {})
    finally:
        monitor.finalize()
    samples = pd.DataFrame(monitor.samples)
    return {"samples": samples, "step_s": step_s}


@st.cache_data(show_spinner=False, max_entries=12)
def strategy_comparison(key: str, carbon_price: float) -> dict:
    runs = compare_strategies(key, carbon_price=carbon_price)
    out = {}
    for name, run in runs.items():
        dt = run.spec.step_s / 3600.0
        out[name] = {"df": run.df, "baseline": vessim_baseline_metrics(run, run.spec, dt)}
    return out


# --------------------------------------------------------------------------- #
# Load flow
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=64)
def power_flow(payload: dict, start: str, tol: float, max_iter: int) -> dict:
    net = net_from(payload)
    result = gridlib.run_pf(net, tol=tol, max_iter=max_iter, start=start)
    return {
        "buses": result.buses,
        "branches": result.branches,
        "history": result.history,
        "summary": result.summary,
        "ybus": result.ybus,
        "bus_order": list(result.bus_order),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "max_mismatch": float(result.max_mismatch),
    }


@st.cache_data(show_spinner=False, max_entries=64)
def dc_power_flow(payload: dict) -> dict:
    net = net_from(payload)
    result = gridlib.run_dc_pf(net)
    return {
        "buses": result.buses,
        "branches": result.branches,
        "history": result.history,
        "summary": result.summary,
        "ybus": result.ybus,
        "bus_order": list(result.bus_order),
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "max_mismatch": float(result.max_mismatch),
    }


@st.cache_data(show_spinner=False, max_entries=32)
def power_flow_vs_pypsa(payload: dict) -> dict:
    """Same model twice: our Newton-Raphson and ``pypsa.Network.pf()``."""
    quiet_pypsa()
    net = net_from(payload)
    ours = gridlib.run_pf(net, start="dc")
    pn = gridlib.to_pypsa(net)
    try:
        pn.pf()
        v = pn.buses_t.v_mag_pu.iloc[0]
        theta = np.degrees(pn.buses_t.v_ang.iloc[0])
        frame = pd.DataFrame({
            "v_ours": ours.buses["v_pu"],
            "v_pypsa": v.reindex(ours.buses.index),
            "angle_ours": ours.buses["angle_deg"],
            "angle_pypsa": theta.reindex(ours.buses.index),
        })
        frame["dv"] = frame["v_ours"] - frame["v_pypsa"]
        frame["dangle"] = frame["angle_ours"] - frame["angle_pypsa"]
        ok = True
        error = ""
    except Exception as exc:  # pragma: no cover - defensive
        frame = pd.DataFrame()
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    return {"frame": frame, "ok": ok, "error": error,
            "max_dv": float(frame["dv"].abs().max()) if ok and len(frame) else float("nan"),
            "max_dangle": float(frame["dangle"].abs().max()) if ok and len(frame) else float("nan")}


# --------------------------------------------------------------------------- #
# Economic dispatch / unit commitment
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=32)
def educ_case(preset: str, generators: pd.DataFrame, loads: pd.DataFrame,
              storages: pd.DataFrame, lines: pd.DataFrame, mode: str, hours: int,
              reserve_frac: float, hour: int, voll: float, carbon_price: float = 0.0) -> dict:
    quiet_pypsa()
    net = build_ed_preset(preset)
    net.generators = generators.copy()
    net.loads = loads.copy()
    net.storages = storages.copy()
    net.lines = lines.copy()
    if carbon_price:
        # bake the carbon price into the merit order so both methods see it
        net.generators["marginal_cost"] = (net.generators["marginal_cost"].astype(float)
                                           + float(carbon_price) * carrier_co2(net))

    info = dict(ED_PRESET_INFO[preset])
    info["hours"] = int(hours)
    info["mode"] = mode
    case = prepare_manual_case(net, info, hours=int(hours))

    ed_series_rows = []
    for h in range(int(hours)):
        one = run_manual_ed(case, h, voll=voll)
        ed_series_rows.append({"hour": h, "demand_mw": float(case.net_load.iloc[h]),
                               "lambda": float(one.lam), "cost_usd": float(one.cost),
                               "served_mw": float(one.served), "shed_mw": float(one.shed)})
    ed_series = pd.DataFrame(ed_series_rows)
    ed = run_manual_ed(case, hour, voll=voll)
    uc = run_manual_uc(case, reserve_frac=reserve_frac, voll=voll)

    n = build_opf_network(net, case.load_ts, case.vre_pu, mode=mode, hours=int(hours),
                          reserve_mw=reserve_frac * float(case.net_load.max()), carbon_price=0.0)
    opf = solve_opf(n, mode=mode)
    summary = opf_summary(opf, net)
    manual = manual_summary(uc, net)

    return {
        "ed": {
            "lam": float(ed.lam), "cost": float(ed.cost), "served": float(ed.served),
            "shed": float(ed.shed), "history": ed.history, "units": ed.units,
            "demand": float(ed.demand), "converged": bool(ed.converged),
        },
        "uc": {
            "hourly": uc.hourly, "dispatch": uc.dispatch, "status": uc.status,
            "total_cost": float(uc.total_cost), "energy_cost": float(uc.energy_cost),
            "startup_cost": float(uc.startup_cost), "shed_mwh": float(uc.shed_mwh),
            "spill_mwh": float(uc.spill_mwh), "committed_hours": int(uc.committed_hours),
        },
        "opf": {
            "status": opf.status, "condition": opf.condition, "objective": float(opf.objective),
            "dispatch": opf.dispatch, "commitment": opf.commitment, "prices": opf.prices,
            "line_loading": opf.line_loading, "runtime_s": float(opf.runtime_s),
        },
        "ed_series": ed_series,
        "manual": manual,
        "optimal": summary,
        "comparison": comparison_table(manual, summary),
        "net_load": case.net_load,
        "load": case.load,
        "vre": case.vre,
        "battery": case.battery_mw,
        "curtailment_mwh": float(case.curtailment_mwh),
    }


# --------------------------------------------------------------------------- #
# Site: Vessim trajectory re-optimised with PyPSA
# --------------------------------------------------------------------------- #

def site_spec_from(df: pd.DataFrame, spec, carbon_price: float, grid_limit: float,
                   battery_mw: float, battery_hours: float, line_grid: float,
                   line_solar: float, line_gas: float, allow_curtailment: bool = True,
                   battery_efficiency: float = 1.0) -> SiteSpec:
    return SiteSpec(
        name=spec.key,
        grid_limit_mw=float(grid_limit),
        solar_mw=max(float(spec.solar_kwp) / 1000.0, 1e-3),
        gas_mw=max(float(spec.gas_kw) / 1000.0, 0.0),
        # the floors are megawatts: 1e-9 MW is a milliwatt, not a kilowatt.  A
        # 1e-3 floor silently resized the watt-scale cases' batteries by 50x.
        battery_mw=max(float(battery_mw), 1e-9),
        battery_hours=max(float(battery_hours), 1e-9),
        line_grid_mw=max(float(line_grid), 1e-9),
        line_solar_mw=max(float(line_solar), 1e-9),
        line_gas_mw=max(float(line_gas), 1e-9) if spec.gas_kw > 0 else None,
        grid_price=float(spec.price_base) * 1000.0,
        grid_carbon=float(spec.grid_carbon),
        gas_price=float(spec.gas_cost_usd_kwh) * 1000.0,
        gas_carbon=float(spec.gas_co2),
        carbon_price=float(carbon_price),
        battery_efficiency=float(battery_efficiency),
        battery_cyclic=False,
        allow_curtailment=bool(allow_curtailment),
    )


@st.cache_data(show_spinner=False, max_entries=32)
def site_case(df: pd.DataFrame, spec_params: dict, dt_hours: float) -> dict:
    quiet_pypsa()
    spec = SiteSpec(**spec_params)
    # the OPF runs on the case's own network — the buses the load-flow tab solves
    result = run_case_opf(df, spec, dt_hours)
    snapshots = result.snapshots
    battery = result.network.storage_units_t.p
    return {
        "status": result.status,
        "device": result.device,
        "branch_p0": result.branch_p0,
        "buses": result.buses,
        "branches": result.branches,
        "battery_mw": (battery.iloc[:, 0] if len(battery.columns)
                       else pd.Series(0.0, index=snapshots)),
        "storage_marginal_cost": float(spec.storage_marginal_cost),
        "storage_discharge_mwh": float(battery.clip(lower=0).to_numpy().sum() * dt_hours),
        "condition": result.condition,
        "objective": float(result.objective),
        "dispatch": result.dispatch,
        "soc": result.soc,
        "prices": result.prices,
        "line_loading": result.line_loading,
        "runtime_s": float(result.runtime_s),
        "energy_cost": float(result.energy_cost),
        "carbon_cost": float(result.carbon_cost),
        "emissions_t": float(result.emissions_t),
        "curtailment_mwh": float(result.curtailment_mwh),
        "battery_throughput_mwh": float(result.battery_throughput_mwh),
        "unmet_mwh": float(result.unmet_mwh),
        "snapshots": pd.DataFrame({"time": snapshots}),
        "dt": float(dt_hours),
    }


@st.cache_data(show_spinner=False, max_entries=8)
def tax_sweep(df: pd.DataFrame, spec_params: dict, dt_hours: float,
              taxes: tuple[float, ...]) -> pd.DataFrame:
    quiet_pypsa()
    rows = []
    for tax in taxes:
        params = dict(spec_params)
        params["carbon_price"] = float(tax)
        res = run_case_opf(df, SiteSpec(**params), dt_hours)
        rows.append({"carbon_price": float(tax),
                     "cost_usd": res.energy_cost + res.carbon_cost,
                     "energy_cost_usd": res.energy_cost,
                     "carbon_cost_usd": res.carbon_cost,
                     "emissions_t": res.emissions_t,
                     "grid_mwh": float(res.dispatch["Grid import"].sum() * dt_hours),
                     "gas_mwh": float(res.dispatch["Gas turbine"].sum() * dt_hours),
                     "status": res.status})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=8)
def line_sweep(df: pd.DataFrame, spec_params: dict, dt_hours: float,
               limits: tuple[float, ...]) -> pd.DataFrame:
    quiet_pypsa()
    rows = []
    for limit in limits:
        params = dict(spec_params)
        params["line_grid_mw"] = float(limit)
        spec = SiteSpec(**params)
        res = run_case_opf(df, spec, dt_hours)
        grid_p = res.dispatch.get("Grid import", pd.Series(0.0))
        rows.append({
            "line_limit_mw": float(limit),
            "cost_usd": res.energy_cost + res.carbon_cost,
            "emissions_t": res.emissions_t,
            "curtailment_mwh": res.curtailment_mwh,
            "grid_mwh": float(grid_p.sum() * dt_hours),
            "congested": bool((res.line_loading.get("POI–Site", pd.Series(0.0)) > 99.5).any()),
            "status": res.status,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Overview (all five cases at once)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=4)
def overview(carbon_price: dict | None = None) -> pd.DataFrame:
    quiet_pypsa()
    rows = []
    for key, spec in CASES.items():
        run = run_case(key)
        dt = spec.step_s / 3600.0
        base = vessim_baseline_metrics(run, spec, dt)
        system = system_for(spec)
        site = SiteSpec(
            name=key,
            grid_limit_mw=system.grid[1],
            solar_mw=max(system.solar[1], 1e-9),
            gas_mw=system.gas[1] if system.gas else 0.0,
            battery_mw=max(system.battery[2], 1e-9),
            battery_hours=max(system.battery[1] / max(system.battery[2], 1e-9), 1e-9),
            line_grid_mw=system.tx_mva,
            grid_price=spec.price_base * 1000.0,
            grid_carbon=spec.grid_carbon,
            gas_price=spec.gas_cost_usd_kwh * 1000.0,
            gas_carbon=spec.gas_co2,
            carbon_price=spec.carbon_price,
        )
        opt = run_case_opf(run.df, site, dt)
        rows.append({
            "key": key,
            "number": spec.number,
            "title": spec.title,
            "steps": spec.steps,
            "dt_min": spec.step_s / 60.0,
            "vessim_cost": base["total_cost_usd"],
            "pypsa_cost": opt.energy_cost + opt.carbon_cost,
            "vessim_emissions_kg": base["emissions_t"] * 1000.0,
            "pypsa_emissions_kg": opt.emissions_t * 1000.0,
            "grid_mwh": base["grid_mwh"],
            "solar_mwh": base["solar_mwh"],
            "vessim_runtime_s": run.runtime_s,
            "pypsa_runtime_s": opt.runtime_s,
            "status": opt.status,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def fmt(value: float, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if abs(value) >= 1e6:
        return f"{value/1e6:,.2f}M{suffix}"
    if abs(value) >= 1e4:
        return f"{value/1e3:,.1f}k{suffix}"
    if abs(value) >= 100:
        return f"{value:,.0f}{suffix}"
    return f"{value:,.{digits}f}{suffix}"


def fmt_usd(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"${value:,.0f}" if abs(value) >= 100 else f"${value:,.2f}"


def fmt_mwh(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}%"


def tone_for_delta(value: float, tolerance: float = 0.0) -> str:
    if not np.isfinite(value):
        return ""
    if value > tolerance:
        return "bad"
    if value < -tolerance:
        return "good"
    return ""


def download(df: pd.DataFrame, filename: str, label: str = "⬇️ Descargar CSV", key: str = "") -> None:
    st.download_button(label, data=df.to_csv(index=True).encode("utf-8"),
                       file_name=filename, mime="text/csv", key=key or filename)
