"""Real Vessim simulations behind every case of the dashboard.

Each case is a genuine :class:`vessim.Environment` run — actors, dispatchables,
dispatch policy, grid signals and a ``Recorder`` — not a re-implementation.
The recorded trajectory is then normalised to MW/MWh so that PyPSA can be asked
to re-optimise exactly the same system.

Cases
-----
``basic``           official ``basic_example.py``: constant load, solar trace,
                    battery and grid connection.
``solar_battery``   official ``strategy_comparison.py``: self-consumption with a
                    battery, plus a three-strategy comparison.
``carbon_aware``    official ``demand_response.py`` idea: a deferrable workload
                    and a battery chasing low-carbon hours.
``sil``             software-in-the-loop: a real CPU workload is measured with
                    ``os.wait4`` and its power drives the microgrid.
``ecopetrol``       new industrial case: plant + data centre, captive gas
                    turbine, solar farm, BESS and a capped grid connection.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

import vessim as vs

from .computing import ComputingSpec, build as build_computing
from .policies import CurtailmentSink, DeferrableLoad, FuelGenerator, MeritOrderPolicy

DATA_DIR = Path(__file__).parent / "data"
SOLAR_CSV = DATA_DIR / "solar_example.csv"


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #

def _series(values: np.ndarray, step_s: int) -> vs.Trace:
    index = np.arange(0, len(values) * step_s, step_s)
    return vs.Trace(pd.Series(values, index=index))


def solar_profile(kwp: float, steps: int, step_s: int, cloudy_seed: int = 3,
                  source: str = "berlin") -> np.ndarray:
    """Solar power in kW for ``steps`` samples.

    ``berlin`` uses the 5-minute trace bundled with Vessim (January days, so a
    low winter sun); ``bogota`` uses a synthetic equatorial day.
    """
    hours = np.arange(steps) * step_s / 3600.0
    if source == "berlin" and SOLAR_CSV.exists():
        frame = pd.read_csv(SOLAR_CSV, nrows=max(steps, 1))
        values = frame["Berlin"].to_numpy(dtype=float)
        if len(values) < steps:
            values = np.resize(values, steps)
        return values * kwp
    day = np.maximum(np.sin(np.pi * (hours % 24 - 6.0) / 12.0), 0.0)
    rng = np.random.default_rng(cloudy_seed)
    clouds = 0.75 + 0.35 * rng.random(len(hours))
    clouds = pd.Series(clouds).rolling(4, min_periods=1).mean().to_numpy()
    return kwp * day * clouds


def carbon_intensity(steps: int, step_s: int, base: float = 0.35, amplitude: float = 0.15,
                     peak_hour: float = 14.0) -> np.ndarray:
    """kg CO2 per kWh, WattTime-like daily shape."""
    hours = np.arange(steps) * step_s / 3600.0
    return base + amplitude * np.sin(2 * np.pi * (hours - peak_hour) / 24.0)


def electricity_price(steps: int, step_s: int, base: float = 0.18,
                      peak: float = 0.42, off_peak: float = 0.11) -> np.ndarray:
    """USD per kWh with a 17:00-21:00 peak (and a small morning shoulder)."""
    hours = np.arange(steps) * step_s / 3600.0
    price = np.full(len(hours), base)
    price[(hours % 24 >= 17) & (hours % 24 < 21)] = peak
    price[(hours % 24 < 6)] = off_peak
    return price


def industrial_load(steps: int, step_s: int, peak_mw: float) -> np.ndarray:
    hours = np.arange(steps) * step_s / 3600.0
    shape = 0.78 + 0.22 * np.sin(np.pi * np.clip((hours % 24 - 5) / 14, 0, 1))
    return peak_mw * shape


def datacenter_load(steps: int, step_s: int, peak_mw: float) -> np.ndarray:
    hours = np.arange(steps) * step_s / 3600.0
    shape = 0.86 + 0.14 * np.sin(2 * np.pi * (hours % 24 - 9) / 24)
    return peak_mw * shape


# --------------------------------------------------------------------------- #
# Case specification
# --------------------------------------------------------------------------- #

@dataclass
class CaseSpec:
    key: str
    number: str
    title: str
    subtitle: str
    description: str
    source: str
    steps: int
    step_s: int
    load_kw: float = 0.7
    solar_kwp: float = 2.4
    solar_source: str = "berlin"
    battery_kwh: float = 1.0
    battery_kw: float = 0.5
    grid_import_kw: float = 2.0
    grid_export_kw: float = 1.5
    gas_kw: float = 0.0
    carbon_base: float = 0.35
    carbon_amplitude: float = 0.15
    price_base: float = 0.18
    carbon_price: float = 60.0
    policy: str = "default"          # default | merit
    load_kind: str = "flat"          # flat | residential | industrial | datacenter | deferrable
    deferrable_fraction: float = 0.0
    extra_loads: list[tuple[str, str, float]] = field(default_factory=list)
    #: the computing layer of the case: what is being dispatched *for*
    computing: tuple[ComputingSpec, ...] = ()
    gas_cost_usd_kwh: float = 0.055
    gas_co2: float = 0.35
    grid_carbon: float = 0.40
    code: str = ""

    @property
    def horizon_hours(self) -> float:
        return self.steps * self.step_s / 3600.0

    @property
    def total_s(self) -> int:
        return self.steps * self.step_s


CASES: dict[str, CaseSpec] = {
    "basic": CaseSpec(
        key="basic", number="01", title="Basic simulation",
        subtitle="Consumer + Producer + battery + grid",
        description=(
            "The official Vasim getting-started example: a constant 700 W load, Berlin solar from "
            "the bundled trace, a 1 kWh battery and the public grid. It is the baseline every other "
            "case builds on."
        ),
        source="vessim/examples/basic_example.py",
        steps=96, step_s=900, load_kw=0.7, solar_kwp=2.4, battery_kwh=1.0, battery_kw=0.5,
        grid_import_kw=2.0, grid_export_kw=1.0, policy="default", load_kind="flat",
        computing=(
            ComputingSpec("edge-node", "edge", 0.045, pue=1.0, flexible=0.0,
                          detail="Raspberry Pi + sensor HAT: 45 W, always on"),
            ComputingSpec("home-nas", "storage", 0.020, pue=1.05, flexible=0.5,
                          detail="NAS with a nightly backup window"),
        ),
        code=(
            "env = vs.Environment(sim_start='2026-01-01', step_size=900)\n"
            "env.add_microgrid(\n"
            "    name='site',\n"
            "    actors=[vs.Consumer('load', signal=700),\n"
            "            vs.Producer('solar', signal=vs.Trace.from_csv('solar_example.csv',\n"
            "                        column='Berlin', scale=2400))],\n"
            "    dispatchables=[vs.SimpleBattery('battery', capacity=1000),\n"
            "                   vs.GridConnection(max_import=2000)],\n"
            ")\n"
            "env.add_controller(vs.Recorder())\n"
            "env.run(until=24 * 3600)"
        ),
    ),
    "solar_battery": CaseSpec(
        key="solar_battery", number="02", title="Solar + battery",
        subtitle="Self-consumption versus peak shaving versus carbon-aware",
        description=(
            "The official strategy_comparison.py: a campus with 3.2 kW of solar, an evening peak and "
            "a battery. The same traces are dispatched by three policies, and the dashboard shows "
            "what each one costs and emits."
        ),
        source="vessim/examples/strategy_comparison.py",
        steps=96, step_s=900, load_kw=1.4, solar_kwp=3.2, battery_kwh=3.0, battery_kw=1.5,
        grid_import_kw=2.5, grid_export_kw=2.0, policy="default", load_kind="residential",
        carbon_base=0.30, carbon_amplitude=0.16, carbon_price=80.0,
        computing=(
            ComputingSpec("home-server", "inference", 0.090, pue=1.15, flexible=0.0,
                          detail="self-hosted assistant on the campus LAN: 90 W IT"),
            ComputingSpec("nas-backup", "storage", 0.050, pue=1.05, flexible=0.8,
                          detail="nightly backup of the lab datasets, movable to cheap hours"),
        ),
        code=(
            "env = vs.Environment(sim_start='2026-01-01', step_size=900)\n"
            "env.add_microgrid(\n"
            "    name='campus',\n"
            "    actors=[vs.Producer('solar', solar_trace),\n"
            "            vs.Consumer('servers', load_trace)],\n"
            "    dispatchables=[vs.SimpleBattery('battery', 3000, initial_soc=0.5),\n"
            "                   vs.GridConnection(max_import=2500)],\n"
            "    policy=vs.DefaultDispatchPolicy(),\n"
            "    grid_signals={'carbon': carbon_trace, 'price': price_trace},\n"
            ")"
        ),
    ),
    "carbon_aware": CaseSpec(
        key="carbon_aware", number="03", title="Carbon-aware workload",
        subtitle="Deferrable jobs + a battery that follows the carbon signal",
        description=(
            "A data-centre workload with a flexible share defers work out of the dirty afternoon "
            "hours, while the merit-order policy charges the battery when electricity is clean and "
            "cheap. Vessim records what happened; PyPSA then asks whether it was optimal."
        ),
        source="vessim/examples/demand_response.py + carbon signals",
        steps=96, step_s=900, load_kw=0.975, solar_kwp=2.0, battery_kwh=4.0, battery_kw=2.0,
        grid_import_kw=3.0, grid_export_kw=2.0, policy="merit", load_kind="flat",
        deferrable_fraction=0.35, carbon_base=0.38, carbon_amplitude=0.18, carbon_price=120.0,
        computing=(
            ComputingSpec("training", "training", 0.525, pue=1.10, flexible=1.0,
                          detail="fine-tuning jobs, interruptible, no deadline: the flexible 35 %"),
            ComputingSpec("inference", "inference", 0.120, pue=1.10, flexible=0.0,
                          detail="online model serving, latency-bound"),
            ComputingSpec("storage", "storage", 0.030, pue=1.05, flexible=0.5,
                          detail="checkpoints and logs, written in the cheap window"),
        ),
        code=(
            "env = vs.Environment(sim_start='2026-01-01', step_size=900)\n"
            "env.add_microgrid(\n"
            "    name='dc',\n"
            "    actors=[vs.Consumer('workload', DeferrableLoad(base_w=900, flexible_w=600,\n"
            "                                                   carbon=carbon_at)),\n"
            "            vs.Producer('solar', solar_trace)],\n"
            "    dispatchables=[vs.SimpleBattery('battery', 4000, initial_soc=0.4),\n"
            "                   vs.GridConnection(max_import=2000)],\n"
            "    policy=MeritOrderPolicy(carbon_price_usd_per_t=120),\n"
            "    grid_signals={'carbon': carbon_trace, 'price': price_trace},\n"
            ")"
        ),
    ),
    "sil": CaseSpec(
        key="sil", number="04", title="Software-in-the-loop",
        subtitle="A real CPU workload drives the microgrid",
        description=(
            "Ten one-second steps while a real Python process burns CPU. os.wait4() reports the "
            "CPU time of that child; the measured watts (plus a base IT load) become the actor's "
            "power, and Vessim dispatches against it in real time."
        ),
        source="vessim/examples/live_server_monitoring.py (software-in-the-loop)",
        steps=8, step_s=1, load_kw=0.30, solar_kwp=0.25, battery_kwh=0.02, battery_kw=0.02,
        grid_import_kw=1.0, grid_export_kw=0.5, policy="default", load_kind="sil",
        computing=(
            ComputingSpec("inference-api", "inference", 0.120, pue=1.10, flexible=0.0,
                          detail="requests served during the test"),
            ComputingSpec("storage", "storage", 0.030, pue=1.05, flexible=0.0,
                          detail="the log store the job writes to"),
        ),
        carbon_base=0.32, carbon_amplitude=0.10, carbon_price=60.0,
        code=(
            "monitor = SoftwareInTheLoopMonitor(base_w=300, core_w=12.0)\n"
            "env = vs.Environment(sim_start='2026-01-01', step_size=1)\n"
            "env.add_microgrid(\n"
            "    name='node',\n"
            "    actors=[vs.Consumer('cpu', monitor.signal)],\n"
            "    dispatchables=[vs.SimpleBattery('battery', 20000),\n"
            "                   vs.GridConnection(max_import=1000)],\n"
            ")\n"
            "env.add_controller(monitor)   # runs a real process every step\n"
            "env.run(until=8)             # eight seconds of wall clock"
        ),
    ),
    "montecarlo": CaseSpec(
        key="montecarlo", number="06", title="Probabilistic campus",
        subtitle="Monte Carlo · a real dataset · SAM-style PV · degrading storage",
        description=(
            "The roadmap case: a 120 kW campus with 180 kWp of photovoltaic, a 300 kWh battery "
            "and a 260 kW connection, run Monte Carlo over weather, load, prices and carbon. The "
            "carbon trace is the WattTime MOER dataset bundled with Vessim, the PV comes from "
            "irradiance through a PVWatts/SAM-style model, and the battery's fading capacity is "
            "priced every day — the four items on Vessim's roadmap, reduced to what fits in a page "
            "and labelled as such."
        ),
        source="vessim roadmap (probabilistic modelling · datasets · SAM · BLAST-Lite), reduced",
        steps=48, step_s=1800, load_kw=120, solar_kwp=180.0, solar_source="bogota",
        battery_kwh=300, battery_kw=150, grid_import_kw=260, grid_export_kw=80,
        policy="merit", load_kind="campus",
        carbon_base=0.42, carbon_amplitude=0.06, price_base=0.18, carbon_price=60.0,
        gas_co2=0.35, grid_carbon=0.42,
        computing=(
            ComputingSpec("training", "training", 30.0, pue=1.15, flexible=0.8,
                          detail="interruptible fine-tuning, 30 kW IT: the flexible share"),
            ComputingSpec("inference", "inference", 18.0, pue=1.15, flexible=0.0,
                          detail="online serving for the campus, latency-bound"),
            ComputingSpec("storage", "storage", 6.0, pue=1.05, flexible=0.5,
                          detail="backups and checkpoints, movable to the cheap window"),
        ),
        code=(
            "env = vs.Environment(sim_start='2026-01-01', step_size=1800)\n"
            "env.add_microgrid(\n"
            "    name='campus',\n"
            "    actors=[vs.Consumer('campus', campus_trace),\n"
            "            vs.Consumer('datacenter', computing_trace),\n"
            "            vs.Producer('solar', pv_trace)],\n"
            "    dispatchables=[vs.SimpleBattery('battery', 300_000, initial_soc=0.5),\n"
            "                   vs.GridConnection(max_import=260_000, max_export=80_000)],\n"
            "    policy=MeritOrderPolicy(carbon_price_usd_per_t=60),\n"
            "    grid_signals={'carbon': watttime_trace, 'price': price_trace},\n"
            ")\n"
            "# this case is run once per Monte Carlo draw, with the traces above\n"
            "# replaced by the draw: that loop lives in roadmap.monte_carlo()."
        ),
    ),
    "ecopetrol": CaseSpec(
        key="ecopetrol", number="05", title="Ecopetrol industrial",
        subtitle="Plant + data centre, gas turbine, solar, BESS and a capped grid",
        description=(
            "The applied case: an 18 MW process plant and a 6 MW data centre behind a 12 MW grid "
            "connection, a captive 20 MW gas turbine, a 25 MW solar farm and a 40 MWh battery. "
            "Vessim dispatches by effective marginal cost; PyPSA re-optimises the same day and the "
            "difference is the price of the heuristic."
        ),
        source="new case study (Colombian industrial microgrid)",
        steps=96, step_s=900, load_kw=18_000, solar_kwp=25_000, solar_source="bogota",
        battery_kwh=40_000, battery_kw=10_000, grid_import_kw=12_000, grid_export_kw=4_000,
        gas_kw=20_000, policy="merit", load_kind="industrial",
        computing=(
            ComputingSpec("gpu-training", "training", 2_600.0, pue=1.25, flexible=0.6,
                          detail="2 600 kW IT of GPUs; 60 % of the jobs are interruptible"),
            ComputingSpec("inference", "inference", 1_400.0, pue=1.25, flexible=0.0,
                          detail="the plant's control and vision models, latency-bound"),
            ComputingSpec("storage", "storage", 800.0, pue=1.15, flexible=0.3,
                          detail="historian and object storage, backups move to the night"),
        ),
        carbon_base=0.18, carbon_amplitude=0.08, price_base=0.16, carbon_price=60.0,
        gas_cost_usd_kwh=0.105, gas_co2=0.35, grid_carbon=0.19,
        code=(
            "env = vs.Environment(sim_start='2026-01-01', step_size=900)\n"
            "env.add_microgrid(\n"
            "    name='ecopetrol',\n"
            "    actors=[vs.Consumer('plant', plant_trace),\n"
            "            vs.Consumer('datacenter', dc_trace),\n"
            "            vs.Producer('solar', solar_trace)],\n"
            "    dispatchables=[vs.SimpleBattery('bess', 40000, initial_soc=0.5),\n"
            "                   FuelGenerator('gas', p_max=20000, fuel_cost=0.105),\n"
            "                   vs.GridConnection(max_import=12000, max_export=4000)],\n"
            "    policy=MeritOrderPolicy(carbon_price_usd_per_t=60),\n"
            "    grid_signals={'carbon': carbon_trace, 'price': price_trace},\n"
            ")"
        ),
    ),
}

CASE_ORDER = ["basic", "solar_battery", "carbon_aware", "sil", "ecopetrol",
              "montecarlo"]


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

@dataclass
class VessimRun:
    key: str
    spec: CaseSpec
    df: pd.DataFrame            # timestamp index, MW/MWh/USD columns
    metrics: dict
    runtime_s: float
    policy_log: list[dict] = field(default_factory=list)
    policy_kind: str = ""
    policy_state: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def hours(self) -> float:
        return len(self.df) * self.spec.step_s / 3600.0

    def summary(self) -> dict:
        df = self.df
        energy = df["load_mw"].sum() * self.spec.step_s / 3600
        return {
            "load_mwh": float(energy),
            "solar_mwh": float(df["solar_mw"].sum() * self.spec.step_s / 3600),
            "grid_import_mwh": float(df["grid_mw"].clip(lower=0).sum() * self.spec.step_s / 3600),
            "grid_export_mwh": float((-df["grid_mw"]).clip(lower=0).sum() * self.spec.step_s / 3600),
            "gas_mwh": float(df.get("gas_mw", pd.Series(0, index=df.index)).sum() * self.spec.step_s / 3600),
            "cost_usd": float(df["cost_usd"].sum()),
            "emissions_kg": float(df["emissions_kg"].sum()),
            "peak_grid_mw": float(df["grid_mw"].clip(lower=0).max()),
            "self_sufficiency": float(
                1 - df["grid_mw"].clip(lower=0).sum() / max(df["load_mw"].sum(), 1e-9)
            ),
        }


def run_case(key: str, carbon_price: float | None = None, solar_scale: float = 1.0,
             load_scale: float = 1.0, progress=None,
             policy_spec=None, policy_override=None,
             signal_noise: float = 0.0, noise_seed: int = 7,
             signals: dict | None = None) -> VessimRun:
    """Execute one case with Vessim and normalise the recording.

    ``policy_override`` takes a ready-made :class:`vessim.DispatchPolicy`
    instance (the page that teaches how to write one uses it); ``policy_spec``
    is the declarative alternative.  ``signal_noise`` multiplies the price and
    carbon traces *the policy sees* by ``1 + σ·N(0,1)`` while the settlement in
    :func:`normalise` keeps using the true traces — that is the value of perfect
    foresight, measured instead of assumed.

    ``signals`` replaces the generated traces with arrays of the caller's own —
    ``solar_kw``, ``carbon``, ``price`` and ``load_factor`` — which is how the
    Monte Carlo case draws a sample and how a dataset is fed in.  The arrays
    must have ``spec.steps`` entries, and the accounting follows them.
    """
    if key not in CASES:
        raise KeyError(f"Unknown case '{key}'. Available: {', '.join(CASES)}")
    spec = CASES[key]
    step_s = spec.step_s
    steps = spec.steps
    load_scale = float(load_scale)
    solar_scale = float(solar_scale)
    carbon_price = CASES[key].carbon_price if carbon_price is None else float(carbon_price)
    # The baseline accounting reads ``spec.carbon_price``; the policy reads the
    # argument.  Keep them the same object so the bill can never disagree with
    # the signal the dispatch actually saw.
    spec = replace(CASES[key], carbon_price=carbon_price)

    solar_kw = solar_profile(spec.solar_kwp * solar_scale, steps, step_s, source=spec.solar_source)
    carbon = carbon_intensity(steps, step_s, spec.carbon_base, spec.carbon_amplitude)
    price = electricity_price(steps, step_s, spec.price_base)
    hours = np.arange(steps) * step_s / 3600.0

    # --- caller-supplied traces (a Monte Carlo draw, a dataset, a scenario) -- #
    signals = dict(signals or {})
    if signals:
        if "solar_kw" in signals:
            solar_kw = np.asarray(signals["solar_kw"], dtype=float)
        if "carbon" in signals:
            carbon = np.asarray(signals["carbon"], dtype=float)
        if "price" in signals:
            price = np.asarray(signals["price"], dtype=float)
        for name, values in (("solar_kw", solar_kw), ("carbon", carbon), ("price", price)):
            if len(values) != steps:
                raise ValueError(f"signal '{name}' has {len(values)} steps, the case has {steps}")
    load_factor = (np.asarray(signals["load_factor"], dtype=float)
                   if "load_factor" in signals else np.ones(steps))

    # --- actors ----------------------------------------------------------- #
    actors = []
    if spec.load_kind == "flat":
        load_values = np.full(steps, spec.load_kw * load_scale)
        actor = vs.Consumer("facility", signal=_series(load_values * 1000, step_s))
    elif spec.load_kind == "campus":
        # a campus with an evening peak and a day-time shoulder; the Monte Carlo
        # draw multiplies it by a correlated factor around 1
        shape = (0.80 + 0.25 * np.exp(-(((hours % 24 - 19) / 2.6) ** 2))
                 + 0.10 * np.sin(2 * np.pi * (hours % 24 - 13) / 24))
        load_values = spec.load_kw * load_scale * shape * load_factor
        actor = vs.Consumer("campus", signal=_series(load_values * 1000, step_s))
    elif spec.load_kind == "residential":
        shape = 0.85 + 0.45 * np.exp(-(((hours % 24 - 19) / 2.5) ** 2))
        load_values = spec.load_kw * load_scale * shape
        actor = vs.Consumer("campus", signal=_series(load_values * 1000, step_s))
    elif spec.load_kind == "deferrable":
        base = spec.load_kw * load_scale * (1 - spec.deferrable_fraction)
        flexible = spec.load_kw * load_scale * spec.deferrable_fraction

        def carbon_at(hour: float) -> float:
            return float(carbon_intensity(max(int(hour * 3600 / step_s) + 1, 1), step_s,
                                          spec.carbon_base, spec.carbon_amplitude)[-1])

        actor = vs.Consumer("workload", signal=DeferrableLoad(
            base * 1000, flexible * 1000, carbon_at, threshold=spec.carbon_base,
            max_w=spec.grid_import_kw * 1000.0 * 0.95))
        load_values = np.array([actor.signal.at(h * 3600) for h in hours]) / 1000.0
    elif spec.load_kind == "industrial":
        load_values = industrial_load(steps, step_s, spec.load_kw / 1000) * load_scale
        actor = vs.Consumer("plant", signal=_series(load_values * 1e6, step_s))
    elif spec.load_kind == "sil":
        actor = vs.Consumer("cpu", signal=vs.StaticSignal(spec.load_kw * 1000))
        load_values = np.full(steps, spec.load_kw * load_scale)
    else:
        raise ValueError(spec.load_kind)
    actors.append(actor)

    for name, kind, kw in spec.extra_loads:
        values = datacenter_load(steps, step_s, kw / 1000) * load_scale
        actors.append(vs.Consumer(name, signal=_series(values * 1e6, step_s)))
        load_values = load_values + values

    # --- the computing layer ---------------------------------------------- #
    # Every case carries explicit computing workloads (see ``computing.py``):
    # they are the actors Vessim exists for, and the reason the dispatch has a
    # carbon signal at all.
    dt_hours = step_s / 3600.0
    hours_grid = (np.arange(steps) * dt_hours) % 24.0
    carbon_of_hour = (lambda hours: np.interp(np.asarray(hours, dtype=float) % 24.0,
                                              hours_grid, carbon)) if steps > 1 else (lambda h: 0.35)
    computing = build_computing(spec.computing, steps, step_s, carbon_of_hour, spec.carbon_base)
    for name, trace in computing.items():
        values = trace * load_scale
        actors.append(vs.Consumer(name, signal=_series(values * 1000.0, step_s)))
        load_values = load_values + values

    actors.append(vs.Producer("solar", signal=_series(solar_kw * 1000, step_s)))

    # --- dispatchables ---------------------------------------------------- #
    battery = vs.SimpleBattery(
        "battery",
        capacity=spec.battery_kwh * 1000,
        initial_soc=0.5,
        c_rate=(spec.battery_kw / spec.battery_kwh) if spec.battery_kwh else None,
    )
    dispatchables = [battery]
    gas = None
    if spec.gas_kw > 0:
        gas = FuelGenerator("gas", p_max=spec.gas_kw * 1000, fuel_cost=spec.gas_cost_usd_kwh,
                            co2_kg_per_kwh=spec.gas_co2)
        dispatchables.append(gas)
    grid = vs.GridConnection("grid", max_import=spec.grid_import_kw * 1000,
                             max_export=spec.grid_export_kw * 1000)
    curtailment = CurtailmentSink("curtailment")
    dispatchables.append(grid)
    dispatchables.append(curtailment)  # spillway: balances sun the site cannot use

    if policy_override is not None:
        policy = policy_override
    elif policy_spec is not None:
        # an explicit scenario from the policy lab
        from .policies import build_policy
        policy = build_policy(policy_spec, spec, carbon_price=carbon_price,
                              default_price=spec.price_base, default_carbon=spec.carbon_base)
        if policy_spec.kind == "peak_shaving":
            dispatchables = [vs.GridConnection("base_import", max_import=policy_spec.peak_cap_kw * 1000,
                                               max_export=0.0),
                             battery, vs.GridConnection("overflow",
                                                        max_import=spec.grid_import_kw * 1000)]
            if spec.gas_kw > 0:
                dispatchables.insert(1, gas)
            dispatchables.append(curtailment)
    elif spec.policy == "merit":
        policy = MeritOrderPolicy(carbon_price_usd_per_t=carbon_price,
                                  default_price_usd_per_kwh=spec.price_base,
                                  default_carbon_kg_per_kwh=spec.carbon_base)
    else:
        policy = vs.DefaultDispatchPolicy()

    # --- controller: SiL monitor or recorder ------------------------------ #
    from .sil import SoftwareInTheLoopMonitor

    # what the *controller* sees can differ from what the meter records
    seen_price, seen_carbon = price, carbon
    if signal_noise:
        rng = np.random.default_rng(noise_seed)
        seen_price = np.clip(price * (1.0 + signal_noise * rng.standard_normal(steps)), 0.0, None)
        seen_carbon = np.clip(carbon * (1.0 + signal_noise * rng.standard_normal(steps)), 0.0, None)

    monitor = None
    env = vs.Environment(sim_start="2026-01-01", step_size=step_s, name=spec.title)
    env.add_microgrid(
        actors=actors,
        dispatchables=dispatchables,
        policy=policy,
        name="site",
        grid_signals={
            "carbon": _series(seen_carbon, step_s),
            "price": _series(seen_price, step_s),
        },
    )
    if spec.load_kind == "sil":
        monitor = SoftwareInTheLoopMonitor(actor.signal, step_s=step_s,
                                           base_w=spec.load_kw * 1000,
                                           core_w=12.0, load_scale=load_scale)
        env.add_controller(monitor)
    recorder = env.add_controller(vs.Recorder(run_id=f"case-{key}"))

    t0 = time.perf_counter()
    env.run(until=steps * step_s)
    runtime = time.perf_counter() - t0

    df = normalise(recorder, spec, load_values, solar_kw, carbon, price, steps)
    run = VessimRun(
        key=key, spec=spec, df=df, metrics=recorder.metrics, runtime_s=runtime,
        policy_log=list(getattr(policy, "log", [])),
        policy_kind=type(policy).__name__,
        policy_state=policy.state() if hasattr(policy, "state") else {},
        extra={"measured_w": monitor.samples if monitor else None,
               "signal_noise": float(signal_noise)},
    )
    return run


def normalise(recorder, spec: CaseSpec, load_values, solar_kw, carbon, price, steps) -> pd.DataFrame:
    raw = recorder.to_df()
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=1, drop=True)
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()

    step_h = spec.step_s / 3600.0
    n = len(raw)
    def col(name, default=0.0):
        return raw[name].to_numpy(dtype=float) if name in raw.columns else np.full(n, default)

    # Vessim's sign convention: consumption is negative, production positive.
    # The dashboard normalises to positive load, positive generation and
    # positive grid import, which is what PyPSA expects.
    load_cols = [c for c in raw.columns
                 if c.startswith("actor_states.") and c.endswith(".power") and "solar" not in c]
    load_mw = np.zeros(n)
    for c in load_cols:
        load_mw = load_mw - col(c) / 1e6
    solar_mw = col("actor_states.solar.power") / 1e6
    grid_mw = -col("p_grid") / 1e6
    battery_mw = -col("dispatch_states.battery.current_power") / 1e6
    soc_mwh = col("dispatch_states.battery.charge_level") / 1e6
    # custom dispatchables record "power"; built-ins record "current_power"
    gas_key = ("dispatch_states.gas.current_power" if "dispatch_states.gas.current_power" in raw.columns
               else "dispatch_states.gas.power")
    gas_mw = -col(gas_key) / 1e6 if spec.gas_kw > 0 else np.zeros(n)
    if "dispatch_states.curtailment.curtailed_kwh" in raw.columns:
        cumulative = raw["dispatch_states.curtailment.curtailed_kwh"].to_numpy(dtype=float)
        step_kwh = np.diff(np.concatenate([[0.0], cumulative]))
        curtailed_mw = step_kwh / max(step_h, 1e-9) / 1000.0
    else:
        curtailed_mw = np.zeros(n)

    # signals aligned to the recorded samples (they are deterministic functions)
    idx = np.minimum((np.arange(n) * spec.step_s // spec.step_s), steps - 1)
    carbon_step = carbon[idx]
    price_step = price[idx]

    cost = grid_mw.clip(min=0) * price_step * step_h - (-grid_mw).clip(min=0) * price_step * 0.35 * step_h
    cost += gas_mw * spec.gas_cost_usd_kwh * 1000.0 * step_h
    # the CO2 uses the *carbon series of this day*, the same one the policy read:
    # billing a constant while dispatching against a signal would make the two
    # halves of the app describe different grids.
    emissions = (grid_mw.clip(min=0) * carbon_step + gas_mw * spec.gas_co2) * step_h

    df = pd.DataFrame({
        "load_mw": load_mw,
        "solar_mw": solar_mw,
        "grid_mw": grid_mw,
        "battery_mw": battery_mw,
        "soc_mwh": soc_mwh,
        "gas_mw": gas_mw,
        "curtailed_mw": curtailed_mw,
        "carbon_kg_mwh": carbon_step * 1000.0,
        "price_usd_mwh": price_step * 1000.0,
        "cost_usd": cost,
        "emissions_kg": emissions * 1000.0,
    }, index=raw.index)
    for c in load_cols:
        actor = c.split("actor_states.", 1)[1].rsplit(".power", 1)[0]
        df[f"load_{actor}_mw"] = -col(c) / 1e6
    df.index.name = "time"
    return df


# --------------------------------------------------------------------------- #
# Strategy comparison (case 02)
# --------------------------------------------------------------------------- #

def compare_strategies(key: str = "solar_battery", carbon_price: float = 80.0) -> dict[str, VessimRun]:
    """Run the same traces under three policies and return the three runs.

    ``self_consumption``  DefaultDispatchPolicy: battery before grid.
    ``peak_shaving``      a 1.2 kW import cap plus an overflow connection.
    ``carbon_aware``      MeritOrderPolicy with a 120 USD/t carbon price.
    """
    spec = CASES[key]
    results = {}
    for strategy in ("self_consumption", "peak_shaving", "carbon_aware"):
        step_s, steps = spec.step_s, spec.steps
        solar_kw = solar_profile(spec.solar_kwp, steps, step_s, source=spec.solar_source)
        carbon = carbon_intensity(steps, step_s, spec.carbon_base, spec.carbon_amplitude)
        price = electricity_price(steps, step_s, spec.price_base)
        hours = np.arange(steps) * step_s / 3600.0
        shape = 0.85 + 0.45 * np.exp(-(((hours % 24 - 19) / 2.5) ** 2))
        load = spec.load_kw * shape

        env = vs.Environment(sim_start="2026-01-01", step_size=step_s, name=strategy)
        battery = vs.SimpleBattery("battery", capacity=spec.battery_kwh * 1000, initial_soc=0.5,
                                   c_rate=spec.battery_kw / spec.battery_kwh)
        if strategy == "peak_shaving":
            dispatchables = [vs.GridConnection("base_import", max_import=1200, max_export=0),
                             battery, vs.GridConnection("overflow")]
            policy = vs.DefaultDispatchPolicy()
        elif strategy == "carbon_aware":
            dispatchables = [battery, vs.GridConnection("grid", max_import=spec.grid_import_kw * 1000)]
            policy = MeritOrderPolicy(carbon_price_usd_per_t=120,
                                      default_price_usd_per_kwh=spec.price_base,
                                      default_carbon_kg_per_kwh=spec.carbon_base)
        else:
            dispatchables = [battery, vs.GridConnection("grid", max_import=spec.grid_import_kw * 1000)]
            policy = vs.DefaultDispatchPolicy()
        env.add_microgrid(
            actors=[vs.Consumer("servers", signal=_series(load * 1000, step_s)),
                    vs.Producer("solar", signal=_series(solar_kw * 1000, step_s))],
            dispatchables=dispatchables, policy=policy, name="campus",
            grid_signals={"carbon": _series(carbon, step_s), "price": _series(price, step_s)},
        )
        recorder = env.add_controller(vs.Recorder(run_id=f"strategy-{strategy}"))
        env.run(until=steps * step_s)
        df = normalise(recorder, spec, load, solar_kw, carbon, price, steps)
        # peak shaving may use two grid connections: sum them
        if strategy == "peak_shaving":
            extra = recorder.to_df()
            if isinstance(extra.index, pd.MultiIndex):
                extra = extra.reset_index(level=1, drop=True)
            if "dispatch_states.overflow.current_power" in extra.columns:
                overflow = extra["dispatch_states.overflow.current_power"].to_numpy() / 1e6
                df["grid_mw"] = df["grid_mw"] + overflow
        results[strategy] = VessimRun(key=f"{key}:{strategy}", spec=spec, df=df,
                                      metrics=recorder.metrics, runtime_s=0.0)
    return results
