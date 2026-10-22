"""Test systems from the papers that use Vessim, reduced to what runs in a page.

Every study here is a *reduced* reproduction: the power models, the parameters
and the signals come from the paper, the size of the deployment is chosen so it
fits in a Streamlit session, and the page says so next to each number.  Nothing
is quoted from a figure without saying which figure.

Papers
------
* Wiesner et al. (2024), *Vessim: A Testbed for Carbon-Aware Applications and
  Systems*, ACM SIGENERGY EIR 4(5) — the reference microgrid.
* Wiesner et al. (2023), *Software-in-the-Loop Simulation for Developing and
  Testing Carbon-Aware Applications*, SPE 53(12) — the CPU→power model.
* Kilian, Wiesner, Kao (2024), *Choosing the Right Battery Model for Data Center
  Simulations*, LOCO — ideal vs linear vs physics-based storage.
* Özcan, Wiesner, Weiß, Kao (2025), *Quantifying the Energy Consumption and
  Carbon Emissions of LLM Inference via Simulations*, PECS@Euro-PAR — the GPU
  power model and the carbon-aware inference study.
* Wiesner et al. (2026), *Distributed LLM Pretraining During Renewable
  Curtailment Windows*, technical report — three geo-distributed clusters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1 · Vessim (2024): the reference microgrid
# --------------------------------------------------------------------------- #

def shifting_value(load_kw: np.ndarray, carbon_kg_per_kwh: np.ndarray,
                   carbon_price: float, flexible: float = 0.35) -> dict:
    """What moving a flexible share out of the dirty hours is worth.

    The paper's demonstration is exactly this: the same energy, a different hour,
    a different carbon intensity.  The flexible share is packed into the cleanest
    steps, at most doubling there; whatever does not fit stays where it was, so
    the daily energy is identical by construction.
    """
    load = np.asarray(load_kw, dtype=float)
    carbon = np.asarray(carbon_kg_per_kwh, dtype=float)
    n = len(load)
    base = load * (1.0 - flexible)
    flex = load * flexible
    total = float(flex.sum())
    moved = np.zeros(n)
    left = total
    for index in np.argsort(carbon):               # cleanest first
        take = min(left, 2.0 * flex[index])
        moved[index] = take
        left -= take
        if left <= 1e-12:
            break
    if left > 1e-12:                               # no room left: it stays flat
        moved = moved + left / n
    new = base + moved
    before = float((carbon * load).sum())
    after = float((carbon * new).sum())
    return {
        "energy_kwh": float(load.sum()), "carbon_before_kg": before, "carbon_after_kg": after,
        "avoided_kg": before - after, "avoided_usd": (before - after) / 1000.0 * carbon_price,
        "share_moved": float(min(1.0, moved.sum() / max(total, 1e-12))),
        "profile_before": load, "profile_after": new, "carbon": carbon,
        "left_unmoved_kwh": max(left, 0.0),
    }


# --------------------------------------------------------------------------- #
# 2 · Software in the loop (SPE 2023): CPU to watts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PowerModel:
    """The linear model the paper fits to a real machine: P = a + b·u."""

    idle_w: float = 62.0
    slope_w: float = 118.0      # watts at 100 % CPU above idle
    r2: float = 0.95            # the paper's mean coefficient of determination

    def watts(self, utilisation: np.ndarray) -> np.ndarray:
        return self.idle_w + self.slope_w * np.clip(np.asarray(utilisation, float), 0.0, 1.0)


def sil_energy(seconds: np.ndarray, cpu: np.ndarray, model: PowerModel) -> dict:
    """Energy of a job from its measured CPU utilisation, at one sample per second."""
    cpu = np.asarray(cpu, float)
    power = model.watts(cpu)
    energy_wh = float(np.sum(power) * 1.0 / 3600.0)
    return {"power_w": power, "energy_wh": energy_wh, "mean_w": float(power.mean()),
            "mean_cpu": float(cpu.mean()), "peak_w": float(power.max())}


# --------------------------------------------------------------------------- #
# 3 · Battery models (LOCO 2024)
# --------------------------------------------------------------------------- #

def battery_day(load_kw: np.ndarray, solar_kw: np.ndarray, capacity_kwh: float,
                power_kw: float, model: str = "linear", backup: float = 0.1) -> dict:
    """One day of storage under one rule, four ways of believing in it.

    Every model runs the same controller — charge from surplus, discharge into
    the deficit, keep a ``backup`` reserve — and the same energy bookkeeping.
    What changes is what the model will let the controller *do*:

    ``ideal``     no losses; the inverter rating is the only limit.
    ``linear``    one-way efficiency ``eta``, same power limit.  The LOCO paper
                  recommends this for short runs: it is cheap and its error
                  stays inside the noise of the experiment.
    ``kibam``     two tanks (available ``y``, bound ``x``).  Current is drawn
                  from ``y`` only and the pump moves charge from ``x`` to ``y``
                  at rate ``k``, so the deliverable power collapses late in the
                  discharge even though energy is still in the battery.
    ``peukert``   usable energy shrinks with the discharge rate:
                  ``E_usable = E · (P/P_rated)^(1-k_p)`` with ``k_p = 1.15``.
    """
    load = np.asarray(load_kw, float)
    solar = np.asarray(solar_kw, float)
    n = len(load)
    dt = 24.0 / n
    eta = 1.0 if model == "ideal" else 0.95
    floor = backup * capacity_kwh
    soc = 0.5 * capacity_kwh                # total stored energy, both tanks
    c, k_pump, k_peukert = 0.5, 0.4, 1.15   # KiBaM ratio, pump rate (1/h), Peukert exp.
    y = c * soc                             # available charge (kWh)
    x = soc - y
    served = np.zeros(n)
    unmet = np.zeros(n)
    grid = np.zeros(n)
    track = np.zeros(n + 1)
    track[0] = soc
    headroom = 0.0

    for t in range(n):
        net = load[t] - solar[t]            # > 0 → deficit
        if net > 0:                         # discharge
            if model == "kibam":
                headroom = min(power_kw, max(y, 0.0) / dt)
            elif model == "peukert":
                rate = max(net / max(power_kw, 1e-9), 1e-9)
                factor = min(1.0, rate ** (1.0 - k_peukert))
                headroom = min(power_kw, max(soc - floor, 0.0) / dt * factor)
            else:
                headroom = min(power_kw, max(soc - floor, 0.0) / dt)
            give = max(min(net, headroom), 0.0)
            stored = give * dt / eta        # what leaves the pack
            soc = max(soc - stored, 0.0)
            if model == "kibam":
                y = max(y - stored, 0.0)
            served[t] = give
            unmet[t] = net - give
            grid[t] = net - give
        else:                               # charge
            take = max(min(-net, power_kw, max(capacity_kwh - soc, 0.0) / dt), 0.0)
            soc = min(soc + take * dt * eta, capacity_kwh)
            if model == "kibam":
                y = min(y + take * dt * eta * c, capacity_kwh * c)
            served[t] = -take
            grid[t] = -net - take
        track[t + 1] = soc
        if model == "kibam":                # pump between the tanks, energy conserved
            x = soc - y
            flow = k_pump * (c * x - (1.0 - c) * y) * dt
            y = float(np.clip(y + flow, 0.0, soc))
            x = soc - y
        else:
            y, x = soc * c, soc * (1.0 - c)

    return {"served_kw": served, "unmet_kw": unmet, "grid_kw": grid,
            "unmet_kwh": float(unmet.sum() * dt),
            "throughput_kwh": float(np.abs(served).sum() * dt),
            "soc_end_kwh": float(soc),
            "grid_import_kwh": float(np.clip(grid, 0, None).sum() * dt),
            "grid_export_kwh": float(-np.clip(grid, None, 0).sum() * dt),
            "served_kwh": float(np.clip(served, 0, None).sum() * dt),
            "soc_kwh": track, "model": model}


# --------------------------------------------------------------------------- #
# 4 · LLM inference (PECS 2025)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GpuModel:
    """The paper's GPU power model, calibrated on published A100/H100/A40 data."""

    name: str
    idle_w: float
    max_w: float
    gamma: float = 0.7
    mfu_sat: float = 0.9

    def watts(self, mfu: np.ndarray) -> np.ndarray:
        mfu = np.clip(np.asarray(mfu, float), 0.0, 1.0)
        return self.idle_w + (self.max_w - self.idle_w) * (mfu / self.mfu_sat) ** self.gamma


GPUS = {
    "A100 SXM4": GpuModel("A100 SXM4", idle_w=100.0, max_w=400.0),
    "H100 SXM5": GpuModel("H100 SXM5", idle_w=60.0, max_w=700.0),
    "A40 PCIe": GpuModel("A40 PCIe", idle_w=30.0, max_w=300.0),
}

#: Wh per request measured in the paper's Fig. 2 for a Llama-3-8B on an A100.
WH_PER_REQUEST = 0.176


def inference_day(requests: np.ndarray, carbon_kg_per_kwh: np.ndarray, gpu: GpuModel,
                  gpus: int = 64, pue: float = 1.2, shiftable: float = 0.45,
                  psu_eff: float = 0.9) -> dict:
    """A serving fleet, then the same work moved out of the dirty hours.

    Power comes from the paper's model applied to an MFU trace that follows the
    request volume; the energy per request is the paper's measurement.  What we
    add is the *energy system*: the same carbon signal Vessim uses.
    """
    requests = np.asarray(requests, float)
    carbon = np.asarray(carbon_kg_per_kwh, float)
    dt = 24.0 / len(requests)
    # MFU rises with batch utilisation: decode-heavy hours sit near 0.25, saturated near 0.6
    saturation = requests / max(requests.max(), 1e-9)
    mfu = 0.22 + 0.38 * saturation
    gpu_w = gpu.watts(mfu) * gpus / psu_eff
    it_kw = gpu_w / 1000.0
    site_kw = it_kw * pue
    energy = site_kw * dt
    emissions = float((energy * carbon).sum())
    # the shiftable share moves to the cleanest steps, up to 2× the local power
    base = site_kw * (1.0 - shiftable)
    flex = site_kw * shiftable
    order = np.argsort(carbon)
    left = float(flex.sum())
    moved = np.zeros_like(flex)
    for index in order:
        take = min(left, 2.0 * flex[index])
        moved[index] = take
        left -= take
        if left <= 1e-9:
            break
    shifted_kw = base + moved
    emissions_after = float((shifted_kw * dt * carbon).sum())
    return {
        "site_kw": site_kw, "shifted_kw": shifted_kw, "carbon": carbon, "dt": dt,
        "energy_kwh": float(energy.sum()), "emissions_kg": emissions,
        "emissions_after_kg": emissions_after,
        "offset": (emissions - emissions_after) / max(emissions, 1e-12),
        "mean_gpu_w": float(gpu_w.mean()), "peak_gpu_w": float(gpu_w.max()),
        "wh_per_request": WH_PER_REQUEST,
    }


# --------------------------------------------------------------------------- #
# 5 · Curtailment-aware pretraining (2026)
# --------------------------------------------------------------------------- #

def curtailment_windows(carbon_kg_per_kwh: np.ndarray, threshold: float) -> np.ndarray:
    """A curtailment window is when clean generation exceeds what the grid absorbs."""
    return np.asarray(carbon_kg_per_kwh, float) <= threshold


def pretraining_day(carbons: list[np.ndarray], threshold: float, cluster_kw: float = 100.0,
                    it_share: float = 0.85) -> dict:
    """Three sites, three grids: train only when at least one is curtailed.

    The baseline is a single site training around the clock — the paper's
    reference.  The result of interest is the ratio of operational emissions.
    """
    carbons = [np.asarray(c, float) for c in carbons]
    n = len(carbons[0])
    dt = 24.0 / n
    windows = [curtailment_windows(c, threshold) for c in carbons]
    any_window = np.logical_or.reduce(windows)
    site_kw = cluster_kw * it_share                # IT load; the rest is facilities
    energy_policy = float(sum((windows[i] * site_kw * dt).sum() for i in range(len(carbons))))
    energy_baseline = float(n * cluster_kw * dt)               # one site, around the clock
    emissions_policy = sum(float((windows[i] * site_kw * dt * carbons[i]).sum())
                           for i in range(len(carbons)))
    baseline_carbon = float(np.mean([c.mean() for c in carbons]))
    emissions_baseline = energy_baseline * baseline_carbon
    return {
        "windows": windows, "any_window": any_window, "dt": dt,
        "hours_available": float(any_window.sum() * dt),
        "energy_policy_kwh": energy_policy, "energy_baseline_kwh": energy_baseline,
        "emissions_policy_kg": emissions_policy, "emissions_baseline_kg": emissions_baseline,
        "ratio": emissions_policy / max(emissions_baseline, 1e-12),
        "ratio_energy": energy_policy / max(energy_baseline, 1e-12),
        "g_per_kwh_policy": 1000.0 * emissions_policy / max(energy_policy, 1e-12),
        "g_per_kwh_baseline": 1000.0 * emissions_baseline / max(energy_baseline, 1e-12),
        "baseline_carbon": baseline_carbon,
    }
