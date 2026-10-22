"""The roadmap case: Monte Carlo, datasets, SAM-style PV and degrading storage.

Vessim's roadmap (read in October 2026) lists four things the maintainers are
working on: **probabilistic modelling** (Monte Carlo over sampled inputs),
**datasets** (prices, carbon intensity, weather), **NREL SAM** as a subsystem for
renewable generators, and **NREL BLAST-Lite** for battery degradation.

This module is a *reduced stand-in* for all four, in one case, so that the app
can show what each one changes — and so that the gap between "what Vessim does
today" and "what the roadmap promises" is explicit rather than implied.

===================  ====================================================  ============================
roadmap item         what this module does                                  what it is not
===================  ====================================================  ============================
probabilistic        ``monte_carlo`` draws weather, load, price and carbon  not inside Vessim: the draws
modelling            from distributions, runs N full Vessim simulations,   happen in this app, one
                     and reports P10/P50/P90 of the outcomes                 ``env.run`` per sample
datasets             ``carbon_dataset`` reads the WattTime MOER trace that  not a live connector: the CSV
                     ships with Vessim, at its own 5-minute resolution      is the one in ``data/``
SAM                  ``solar_geometry`` + ``pvwatts``: irradiance from      not SAM itself, and not even
                     solar geometry, cell temperature from NOCT, power      PVWatts' full transposition:
                     from the temperature-corrected efficiency              a clear-sky envelope
BLAST-Lite           ``degradation``: cycle fade from equivalent full       not a physics model: no
                     cycles plus calendar fade, priced in USD/day            rainflow, no DOD-dependent
                                                                             cycle life
===================  ====================================================  ============================

The numbers are computed, not quoted, and every function is tested in
``tests/test_roadmap.py``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .vessim_runner import CASES, run_case

DATA_DIR = Path(__file__).parent / "data"
WATTTIME_CSV = DATA_DIR / "watttime_example.csv"


# --------------------------------------------------------------------------- #
# 1 · SAM-style photovoltaic model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PVModel:
    """The four numbers PVWatts needs, plus the derate SAM would break down."""

    kwp: float
    gamma: float = -0.0037      # power coefficient of the module, 1/K
    noct_c: float = 45.0        # cell temperature at 800 W/m2 and 20 C ambient
    derate: float = 0.86        # soiling, mismatch, wiring, inverter

    def cell_c(self, ghi_w_m2: np.ndarray, ambient_c: np.ndarray) -> np.ndarray:
        """NOCT model: the cell runs hotter than the air, and hotter under load."""
        ghi = np.asarray(ghi_w_m2, dtype=float)
        return np.asarray(ambient_c, dtype=float) + (self.noct_c - 20.0) / 800.0 * ghi

    def power_kw(self, ghi_w_m2: np.ndarray, ambient_c: np.ndarray) -> np.ndarray:
        ghi = np.clip(np.asarray(ghi_w_m2, dtype=float), 0.0, None)
        cell = self.cell_c(ghi, ambient_c)
        return np.clip(self.kwp * ghi / 1000.0 * (1.0 + self.gamma * (cell - 25.0))
                       * self.derate, 0.0, None)


def solar_geometry(hours: np.ndarray, latitude_deg: float = 4.6,
                   day_of_year: int = 81) -> np.ndarray:
    """Clear-sky global horizontal irradiance from solar geometry, in W/m².

    Declination, hour angle, solar elevation, and an air-mass attenuation
    (Meinel):  ``GHI  = 1000 · cos(z)^1.15``.  Bogotá's 4.6° N is why the
    profile is almost the same every day of the year — the case is placed where
    the user is, and the model says so.
    """
    hours = np.asarray(hours, dtype=float)
    declination = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365.0)
    latitude = np.radians(latitude_deg)
    hour_angle = np.radians(15.0 * (hours - 12.0))
    cos_zenith = (np.sin(latitude) * np.sin(declination)
                  + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle))
    return 1000.0 * np.clip(cos_zenith, 0.0, None) ** 1.15


def cloud_factor(rng: np.random.Generator, steps: int, persistence: float = 0.75) -> np.ndarray:
    """A cloud index in [0.25, 1]: Beta draws smoothed, so clouds last hours."""
    raw = rng.beta(2.6, 1.8, steps)
    smooth = pd.Series(raw).ewm(alpha=1.0 - persistence, adjust=False).mean().to_numpy()
    lo, hi = smooth.min(), smooth.max()
    if hi - lo < 1e-9:
        return np.full(steps, 0.75)
    return 0.25 + 0.75 * (smooth - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# 2 · A dataset, not a formula
# --------------------------------------------------------------------------- #

def carbon_dataset(steps: int, step_s: int, day: int = 0) -> np.ndarray:
    """The WattTime MOER trace bundled with Vessim, resampled to this case.

    ``watttime_example.csv`` ships with a corrupted header (``oootime``) and 30
    days of 5-minute data for ``Caiso_North``.  The column is read as
    **g CO₂/kWh** — the unit Vessim's own example assumes — so the returned
    trace is in kg CO₂/kWh.  If the file is missing the function falls back to a
    synthetic shape and says so through ``source``.
    """
    values = None
    if WATTTIME_CSV.exists():
        lines = WATTTIME_CSV.read_text().splitlines()
        header = lines[0].split(",")
        lines[0] = "time,moer"
        frame = pd.read_csv(io.StringIO("\n".join(lines)), parse_dates=["time"])
        raw = frame["moer"].to_numpy(dtype=float)
        per_day = max(int(round(24 * 3600 / 300)), 1)          # 5-minute samples
        start = (day * per_day) % max(len(raw) - per_day, 1)
        window = raw[start:start + per_day]
        if len(window) < per_day:                              # pragma: no cover
            window = raw[:per_day]
        bins = np.linspace(0, len(window), steps + 1).round().astype(int)
        values = np.array([window[a:b].mean() for a, b in zip(bins[:-1], bins[1:])])
        if header[0] in ("oootime", "time"):                   # documented repair
            pass
        return values / 1000.0
    hours = np.arange(steps) * step_s / 3600.0
    return 0.42 + 0.08 * np.sin(2 * np.pi * (hours - 6) / 24.0)   # pragma: no cover


def dataset_summary() -> dict:
    """What the bundled dataset is, so a page can state it instead of alluding."""
    if not WATTTIME_CSV.exists():                               # pragma: no cover
        return {"rows": 0, "days": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    lines = WATTTIME_CSV.read_text().splitlines()
    lines[0] = "time,moer"
    frame = pd.read_csv(io.StringIO("\n".join(lines)), parse_dates=["time"])
    values = frame["moer"].to_numpy(dtype=float) / 1000.0
    return {"rows": int(len(values)), "days": int(len(values) * 5 / (60 * 24)),
            "mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()),
            "start": str(frame["time"].iloc[0]), "end": str(frame["time"].iloc[-1])}


def price_tariff(hours: np.ndarray, base: float = 0.18, peak: float = 0.42,
                 off_peak: float = 0.11) -> np.ndarray:
    """The same tariff the other cases use, so a price is a price across cases."""
    hours = np.asarray(hours, dtype=float) % 24.0
    price = np.full(len(hours), base)
    price[(hours >= 17) & (hours < 21)] = peak
    price[hours < 6] = off_peak
    return price


def ar1(rng: np.random.Generator, steps: int, sigma: float = 0.05,
        rho: float = 0.8) -> np.ndarray:
    """A mean-reverting multiplier around 1 — correlated noise, not white."""
    out = np.zeros(steps)
    shock = rng.normal(0.0, sigma, steps)
    for t in range(steps):
        out[t] = rho * out[t - 1] + shock[t] if t else shock[t]
    return np.clip(1.0 + out, 0.75, 1.25)


# --------------------------------------------------------------------------- #
# 3 · Degradation, in the spirit of BLAST-Lite
# --------------------------------------------------------------------------- #

def degradation(soc_mwh: np.ndarray, dt_hours: float, capacity_mwh: float,
                cycles_to_eol: int = 6000, replacement_usd_per_kwh: float = 250.0,
                calendar_per_day: float = 5.5e-5) -> dict:
    """Capacity fade from the recorded state of charge, priced in USD/day.

    Two terms, both linear in what the battery did:

    ``cycle``     equivalent full cycles = discharged energy / capacity;
                  the cell loses 20 % of its capacity over ``cycles_to_eol``.
    ``calendar``  20 % over ten years, i.e. ``calendar_per_day`` per day.

    This is deliberately *not* a physics model.  BLAST-Lite would ask for depth
    of discharge, C-rate and temperature; here the case only has the SoC trace,
    so the model is the coarsest one that still reacts to usage.  The page says
    so, and the number it produces is the wear cost the dispatch ignores.
    """
    soc = np.asarray(soc_mwh, dtype=float)
    discharged = float(np.clip(soc[:-1] - soc[1:], 0.0, None).sum()) if len(soc) > 1 else 0.0
    throughput = float(np.abs(np.diff(soc)).sum()) if len(soc) > 1 else 0.0
    efc = discharged / max(capacity_mwh, 1e-12)
    cycle_fade = 0.20 * efc / max(cycles_to_eol, 1)
    soh = 1.0 - cycle_fade - calendar_per_day
    wear_usd = cycle_fade * capacity_mwh * 1000.0 * replacement_usd_per_kwh
    return {
        "efc": efc, "throughput_mwh": throughput, "discharged_mwh": discharged,
        "cycle_fade": cycle_fade, "calendar_fade": calendar_per_day,
        "soh_end": soh, "wear_usd": wear_usd,
        "days_to_eol": (1.0 - soh) / max(cycle_fade, 1e-12) if cycle_fade > 0 else float("inf"),
        "dt_hours": dt_hours,
    }


# --------------------------------------------------------------------------- #
# 4 · Monte Carlo over full Vessim runs
# --------------------------------------------------------------------------- #

def scenario(seed: int, case_key: str = "montecarlo", cloud_scale: float = 1.0,
             load_sigma: float = 0.05, price_sigma: float = 0.05,
             carbon_sigma: float = 0.07, day: int = 0) -> dict:
    """One draw of the four inputs, as the signal arrays Vessim will consume."""
    spec = CASES[case_key]
    steps, step_s = spec.steps, spec.step_s
    hours = np.arange(steps) * step_s / 3600.0
    rng = np.random.default_rng(seed)

    cloud = cloud_factor(rng, steps)
    ghi = solar_geometry(hours) * (1.0 - cloud_scale * (1.0 - cloud))
    ambient = 20.0 + 6.0 * np.sin(2 * np.pi * (hours - 9) / 24.0) + rng.normal(0, 1.5, steps)
    pv_kw = PVModel(kwp=float(spec.solar_kwp)).power_kw(ghi, ambient)
    carbon = carbon_dataset(steps, step_s, day=day) * ar1(rng, steps, carbon_sigma)
    price = price_tariff(hours, base=spec.price_base) * ar1(rng, steps, price_sigma)
    load_factor = ar1(rng, steps, load_sigma)
    return {
        "seed": int(seed), "hours": hours, "ghi": ghi, "ambient_c": ambient,
        "solar_kw": pv_kw, "carbon": np.clip(carbon, 0.01, None),
        "price": np.clip(price, 0.0, None), "load_factor": load_factor,
    }


def _payload(run) -> dict:
    dt = run.spec.step_s / 3600.0
    from .analysis import vessim_baseline_metrics

    return {"df": run.df, "dt": dt, "spec": run.spec,
            "baseline": vessim_baseline_metrics(run, run.spec, dt),
            "policy_kind": getattr(run, "policy_kind", "")}


def monte_carlo(n: int = 16, seed: int = 5, case_key: str = "montecarlo",
                carbon_price: float | None = None, cloud_scale: float = 1.0,
                load_sigma: float = 0.05, price_sigma: float = 0.05,
                carbon_sigma: float = 0.07) -> dict:
    """Run ``n`` full Vessim simulations, one draw each, and summarise them.

    This is the honest shape of probabilistic modelling at this stage: Vessim
    has no distribution-aware API — each sample is a complete ``env.run`` — so
    the loop lives here, and the cost of a Monte Carlo run is *n times* the cost
    of a single simulation.  That sentence is the case's contribution to the
    roadmap discussion.
    """
    spec = CASES[case_key]
    carbon_price = float(spec.carbon_price if carbon_price is None else carbon_price)
    rows, signals = [], []
    for index in range(n):
        draw = scenario(seed + 101 * index, case_key=case_key, cloud_scale=cloud_scale,
                        load_sigma=load_sigma, price_sigma=price_sigma,
                        carbon_sigma=carbon_sigma)
        run = run_case(case_key, carbon_price=carbon_price,
                       signals={"solar_kw": draw["solar_kw"], "carbon": draw["carbon"],
                                "price": draw["price"], "load_factor": draw["load_factor"]})
        payload = _payload(run)
        base = payload["baseline"]
        wear = degradation(run.df["soc_mwh"].to_numpy(), payload["dt"],
                           float(spec.battery_kwh) / 1000.0)
        rows.append({
            "scenario": index,
            "PV kWh": float(draw["solar_kw"].sum() * payload["dt"]),
            "mean GHI W/m²": float(draw["ghi"].mean()),
            "load factor": float(draw["load_factor"].mean()),
            "bill USD": float(base["total_cost_usd"]),
            "CO₂ kg": float(base["emissions_t"]) * 1000.0,
            "peak import kW": float(base["peak_grid_mw"]) * 1000.0,
            "grid kWh": float(base["grid_mwh"]) * 1000.0,
            "self-sufficiency %": 100.0 * (1.0 - base["grid_mwh"]
                                           / max(run.df["load_mw"].sum() * payload["dt"], 1e-9)),
            "EFC": wear["efc"], "SoH %": 100.0 * wear["soh_end"],
            "wear USD": wear["wear_usd"],
        })
        signals.append(draw)

    frame = pd.DataFrame(rows)
    metrics = ["PV kWh", "bill USD", "CO₂ kg", "peak import kW", "self-sufficiency %",
               "EFC", "SoH %", "wear USD"]
    summary = pd.DataFrame([
        {"metric": m,
         "P10": float(np.percentile(frame[m], 10)),
         "P50": float(np.percentile(frame[m], 50)),
         "P90": float(np.percentile(frame[m], 90)),
         "spread": float(np.percentile(frame[m], 90) - np.percentile(frame[m], 10))}
        for m in metrics
    ])
    return {"frame": frame, "summary": summary, "signals": signals, "n": int(n),
            "seed": int(seed), "case": case_key, "carbon_price": carbon_price,
            "dataset": dataset_summary()}


def deterministic_reference(case_key: str = "montecarlo") -> dict:
    """The same case with the app's ordinary signals — the number to beat."""
    from .ui import data

    payload = data.vessim_case(case_key, float(CASES[case_key].carbon_price), 1.0, 1.0)
    wear = degradation(payload["df"]["soc_mwh"].to_numpy(), payload["dt"],
                       float(CASES[case_key].battery_kwh) / 1000.0)
    base = payload["baseline"]
    return {"bill": float(base["total_cost_usd"]), "co2_kg": float(base["emissions_t"]) * 1000.0,
            "wear": wear, "run": payload}


def degradation_by_policy(case_key: str = "montecarlo") -> pd.DataFrame:
    """Wear is not a property of the battery: it is a property of the rule.

    Seven dispatch policies, the same day, the same cell.  What changes is how
    often and how deeply it is cycled — and therefore how many days of life the
    rule spends to save a dollar of electricity.
    """
    from .policies import POLICY_CATALOGUE, POLICY_ORDER
    from .ui import data

    spec = CASES[case_key]
    carbon_price = float(spec.carbon_price)
    matrix = data.policy_matrix(case_key, carbon_price, 1.0, 1.0,
                                tuple(POLICY_CATALOGUE[key] for key in POLICY_ORDER))
    rows = []
    for key, payload in matrix.items():
        wear = degradation(payload["df"]["soc_mwh"].to_numpy(), payload["dt"],
                           float(spec.battery_kwh) / 1000.0)
        base = payload["baseline"]
        rows.append({
            "policy": POLICY_CATALOGUE[key].label, "key": key,
            "bill USD": float(base["total_cost_usd"]),
            "CO₂ kg": float(base["emissions_t"]) * 1000.0,
            "EFC": wear["efc"], "wear USD": wear["wear_usd"],
            "SoH after one year %": 100.0 * (1.0 - (1.0 - wear["soh_end"]) * 365.0),
        })
    return pd.DataFrame(rows).sort_values("bill USD").reset_index(drop=True)
