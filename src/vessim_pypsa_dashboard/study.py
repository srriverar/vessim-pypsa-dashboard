"""Studies: many Vessim runs, indexed like the Vessim viewer indexes them.

The official viewer (`python -m vessim view results/study`) opens a directory of
recorded runs and gives three pages — *Results*, *Time series* and *Run details*
— over two kinds of entity:

* a **configuration**: one parameter set (an Optuna trial, a policy, a sizing),
* a **run**: one execution of a configuration (a seed, a demand realisation).

This module produces exactly that structure from real simulations, so the
dashboard can offer the same options — tables, distributions, scatter plots with
a Pareto front, overlays and run details — instead of a static screenshot of
somebody else's study.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

import vessim as vs

from .policies import POLICY_CATALOGUE, POLICY_ORDER, MeritOrderPolicy
from .vessim_runner import CASES, run_case

#: Metrics the viewer can plot, with the units and the direction of "better".
METRICS: dict[str, dict] = {
    "bill_usd": {"label": "Operating cost", "unit": "USD", "better": "min"},
    "co2_kg": {"label": "Emissions", "unit": "kg", "better": "min"},
    "gas_mwh": {"label": "Gas", "unit": "MWh", "better": "min"},
    "grid_mwh": {"label": "Grid import", "unit": "MWh", "better": "min"},
    "solar_mwh": {"label": "Solar used", "unit": "MWh", "better": "max"},
    "battery_mwh": {"label": "Battery throughput", "unit": "MWh", "better": "min"},
    "peak_grid_mw": {"label": "Peak import", "unit": "MW", "better": "min"},
    "self_sufficiency": {"label": "Self-sufficiency", "unit": "%", "better": "max"},
    "curtailed_mwh": {"label": "Curtailed", "unit": "MWh", "better": "min"},
}


def _metrics(run) -> dict:
    dt = run.spec.step_s / 3600.0
    df = run.df
    load = float(df["load_mw"].sum() * dt)
    grid = float(df["grid_mw"].clip(lower=0).sum() * dt)
    solar_used = float(df["solar_mw"].sum() * dt)
    co2 = float(df["emissions_kg"].sum()) / 1000.0
    price = df["price_usd_mwh"].to_numpy(float)
    gas_cost = run.spec.gas_cost_usd_kwh * 1000.0
    tau = float(run.spec.carbon_price)
    energy = float((df["grid_mw"].clip(lower=0) * price * dt).sum()
                   - ((-df["grid_mw"]).clip(lower=0) * price * 0.35 * dt).sum()
                   + (df.get("gas_mw", 0) * gas_cost * dt).sum())
    bill = energy + co2 * 1000.0 * tau / 1000.0
    return {
        "bill_usd": bill,
        "co2_kg": co2 * 1000.0,
        "gas_mwh": float(df.get("gas_mw", pd.Series(0.0)).sum() * dt),
        "grid_mwh": grid,
        "solar_mwh": solar_used,
        "battery_mwh": float(df["battery_mw"].abs().sum() * dt / 2),
        "peak_grid_mw": float(df["grid_mw"].max()),
        "self_sufficiency": 100.0 * (1.0 - grid / load) if load else 0.0,
        "curtailed_mwh": float(df.get("curtailed_mw", pd.Series(0.0)).sum() * dt),
    }


class _Recorder:
    """Collect configurations and runs; the viewer's data model, in Python."""

    def __init__(self, study_id: str, name: str, description: str):
        self.study = {"study_id": study_id, "name": name, "description": description,
                      "configurations": [], "runs": [], "traces": {}}
        self._seen: dict[str, dict] = {}

    def add(self, params: dict, label: str, run_key: str, run, extra_name: str = "") -> dict:
        config_key = f"c{len(self._seen) + 1:02d}" if label not in self._seen else self._seen[label]["key"]
        if config_key not in self._seen:
            config = {"key": config_key, "name": label, "parameters": dict(params)}
            self._seen[config_key] = config
            self.study["configurations"].append({**config, "runs": []})
        config = next(c for c in self.study["configurations"] if c["key"] == config_key)
        metrics = _metrics(run)
        record = {
            "key": run_key,
            "name": run_key if not extra_name else extra_name,
            "configuration": config_key,
            "configuration_name": config["name"],
            "parameters": {**params, **({"realisation": extra_name} if extra_name else {})},
            **metrics,
        }
        self.study["runs"].append(record)
        config["runs"].append(run_key)
        self.study["traces"][run_key] = run.df
        return record


def policy_study(case_key: str = "ecopetrol", n_trials: int = 22,
                 seeds: Iterable[int] = (1, 2), noise: float = 0.08,
                 carbon_price: float | None = None) -> dict:
    """An Optuna search over the merit-order policy, then two runs per trial.

    The objective is multi-objective — cost and CO₂ — so the study has a Pareto
    front, which is exactly what the viewer's scatter plot is for.  The two
    seeds are the "runs per configuration": same parameters, different signal
    noise, so the distribution panel has something to show.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:                                  # pragma: no cover - fallback
        optuna = None

    spec = CASES[case_key]
    tau_default = float(spec.carbon_price if carbon_price is None else carbon_price)
    recorder = _Recorder("optuna-search", f"Optuna search · {spec.title}",
                         "Multi-objective search over the dispatch policy on the same day. "
                         "Each trial is a policy configuration; each run is one noise realisation.")

    def evaluate(params: dict) -> list[tuple]:
        scores = []
        for seed in seeds:
            policy = MeritOrderPolicy(
                carbon_price_usd_per_t=params["carbon_price"],
                strategy=params["strategy"],
                degradation_cost_usd_per_kwh=params["degradation"],
                export_price_factor=params["export_factor"],
                soc_target=params["soc_target"],
                charge_bonus_usd_per_kwh=params["charge_bonus"],
                default_price_usd_per_kwh=spec.price_base,
                default_carbon_kg_per_kwh=spec.carbon_base)
            run = run_case(case_key, carbon_price=params["carbon_price"],
                           policy_override=policy, signal_noise=noise, noise_seed=int(seed))
            metrics = _metrics(run)
            label = (f"τ={params['carbon_price']:.0f} · {params['strategy']} · "
                     f"wear={params['degradation']:.3f}")
            recorder.add({**params}, label, f"trial-{len(recorder.study['configurations']) + 1:02d}-s{seed}",
                         run, extra_name=f"seed {seed}")
            scores.append((metrics["bill_usd"], metrics["co2_kg"]))
        return scores

    if optuna is not None:
        study = optuna.create_study(directions=["minimize", "minimize"],
                                    sampler=TPESampler(seed=7, multivariate=True))
        for _ in range(int(n_trials)):
            study.optimize(_objective(study_space(optuna, tau_default), evaluate), n_trials=1)
        best = study.best_trials[0] if study.best_trials else None
        recorder.study["best"] = ({"bill_usd": best.values[0], "co2_kg": best.values[1],
                                   "trial": best.number, "params": dict(best.params)}
                                  if best is not None else None)
    else:                                              # pragma: no cover - fallback
        rng = np.random.default_rng(7)
        for _ in range(int(n_trials)):
            evaluate(sample_space(rng, tau_default))
        recorder.study["best"] = None

    study_payload = _finish(recorder)
    study_payload["search"] = "optuna" if optuna is not None else "random"
    return study_payload


def _objective(space, evaluate):
    def objective(trial):
        params = space(trial)
        scores = evaluate(params)
        bills = np.array([s[0] for s in scores])
        co2 = np.array([s[1] for s in scores])
        trial.set_user_attr("bill_std", float(bills.std()))
        return float(bills.mean()), float(co2.mean())
    return objective


def study_space(optuna, tau_default: float):
    def space(trial) -> dict:
        return {
            "strategy": trial.suggest_categorical(
                "strategy", ["cost", "carbon_first", "battery_first"]),
            "carbon_price": trial.suggest_float("carbon_price", 0.0, 300.0, step=10.0),
            "degradation": trial.suggest_float("degradation", 0.0, 0.08, step=0.005),
            "export_factor": trial.suggest_float("export_factor", 0.1, 1.0, step=0.05),
            "soc_target": trial.suggest_float("soc_target", 0.2, 0.8, step=0.05),
            "charge_bonus": trial.suggest_float("charge_bonus", 0.0, 0.08, step=0.005),
        }
    return space


def sample_space(rng, tau_default: float):
    def space() -> dict:
        return {
            "strategy": str(rng.choice(["cost", "carbon_first", "battery_first"])),
            "carbon_price": float(rng.integers(0, 31) * 10.0),
            "degradation": float(rng.integers(0, 17) * 0.005),
            "export_factor": float(rng.integers(2, 21) * 0.05),
            "soc_target": float(rng.integers(4, 17) * 0.05),
            "charge_bonus": float(rng.integers(0, 17) * 0.005),
        }
    return space


def dispatch_study(case_key: str = "ecopetrol",
                   scales: Iterable[float] = (0.9, 1.0, 1.1)) -> dict:
    """The catalogue policies across demand realisations — the sampling study."""
    spec = CASES[case_key]
    recorder = _Recorder("dispatch-comparison", f"Dispatch strategies · {spec.title}",
                         "Seven policies evaluated on three demand realisations: "
                         "the same rule, different days.")
    for key in POLICY_ORDER:
        if key == "peak_shaving" and case_key != "solar_battery":
            continue
        policy_spec = POLICY_CATALOGUE[key]
        for scale in scales:
            run = run_case(case_key, carbon_price=spec.carbon_price,
                           policy_spec=policy_spec, load_scale=float(scale),
                           signal_noise=0.02, noise_seed=int(scale * 10))
            recorder.add({"policy": policy_spec.label, "load_scale": float(scale)},
                         policy_spec.label, f"{key}-x{scale:.2f}", run,
                         extra_name=f"load × {scale:.2f}")
    return _finish(recorder)


def _finish(recorder: _Recorder) -> dict:
    study = recorder.study
    parameters = sorted({k for c in study["configurations"] for k in c["parameters"]})
    study["parameters"] = parameters
    study["metrics"] = list(METRICS)
    study["runs"] = sorted(study["runs"], key=lambda r: r["key"])
    return study


# --------------------------------------------------------------------------- #
# The viewer's analysis primitives
# --------------------------------------------------------------------------- #

def pareto_front(rows: list[dict], x: str, y: str,
                 x_sense: str = "min", y_sense: str = "min") -> list[str]:
    """Keys on the Pareto front of two objectives (the viewer's 'Pareto front')."""
    sign_x = -1.0 if x_sense == "min" else 1.0
    sign_y = -1.0 if y_sense == "min" else 1.0
    points = [(row["key"], sign_x * float(row[x]), sign_y * float(row[y])) for row in rows]
    front = []
    for key, px, py in points:
        dominated = any(
            (qx >= px and qy >= py) and (qx > px or qy > py)
            for other, qx, qy in points if other != key
        )
        if not dominated:
            front.append(key)
    return front


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    data = np.sort(np.asarray([v for v in values if v is not None], dtype=float))
    if not len(data):
        return np.array([]), np.array([])
    return data, np.arange(1, len(data) + 1) / len(data)


def aggregate(runs: list[dict], config: str, metric: str, stat: str) -> float:
    values = [float(r[metric]) for r in runs if r["configuration"] == config and r.get(metric) is not None]
    if not values:
        return float("nan")
    if stat == "min":
        return float(np.min(values))
    if stat == "max":
        return float(np.max(values))
    if stat == "mean":
        return float(np.mean(values))
    return float(np.median(values))
