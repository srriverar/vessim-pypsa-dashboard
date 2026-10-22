"""Vessim extensions: a fuel generator and a price/carbon-aware dispatch policy.

Vessim ships ``SimpleBattery``, ``GridConnection`` and
``DefaultDispatchPolicy``, which allocates power in list order.  For the
economically interesting cases this module adds:

* :class:`FuelGenerator` — an on-site gas turbine as a first-class
  ``Dispatchable``, with fuel cost, CO2 intensity and an energy/emissions
  counter in ``state()``.
* :class:`MeritOrderPolicy` — a ``DispatchPolicy`` that dispatches by *effective
  marginal cost*: wholesale price plus a carbon price for the grid, fuel cost
  plus carbon for the turbine, a degradation/opportunity cost for the battery.
  This is what makes the Ecopetrol case meaningful: Vessim then reacts to price
  and carbon signals, and PyPSA can be asked whether the reaction was optimal.
* :class:`DeferrableLoad` — a ``Signal`` that shifts a share of a workload away
  from high-carbon hours while conserving total energy (carbon-aware computing).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

import vessim as vs
from dataclasses import dataclass

from .computing import water_fill


# --------------------------------------------------------------------------- #
# A dispatchable generator
# --------------------------------------------------------------------------- #

class FuelGenerator(vs.Dispatchable):
    """An on-site fuel generator (positive power = consumption, as in Vessim).

    Parameters
    ----------
    p_max: electrical rating in W.
    fuel_cost: USD per kWh generated.
    co2_kg_per_kwh: CO2 intensity of the fuel, kg per kWh generated.
    min_load: minimum stable output as a fraction of ``p_max``.
    """

    def __init__(self, name: str, p_max: float, fuel_cost: float = 0.055,
                 co2_kg_per_kwh: float = 0.35, min_load: float = 0.0):
        super().__init__(name)
        if p_max <= 0:
            raise ValueError("p_max must be positive")
        self.p_max = float(p_max)
        self.fuel_cost = float(fuel_cost)
        self.co2_kg_per_kwh = float(co2_kg_per_kwh)
        self.min_load = float(min_load)
        self.energy_wh = 0.0
        self.co2_kg = 0.0
        self.fuel_usd = 0.0

    def feasible_range(self, duration: float) -> tuple[float, float]:
        # negative = generation
        return (-self.p_max, 0.0)

    def step(self, duration: float) -> None:
        generated_w = max(-self.current_power, 0.0)
        kwh = generated_w * duration / 3600.0 / 1000.0
        self.energy_wh += kwh * 1000.0
        self.co2_kg += kwh * self.co2_kg_per_kwh
        self.fuel_usd += kwh * self.fuel_cost

    def config(self) -> dict:
        return {"p_max": self.p_max, "fuel_cost": self.fuel_cost,
                "co2_kg_per_kwh": self.co2_kg_per_kwh, "min_load": self.min_load}

    def state(self) -> dict:
        return {"power": self.current_power, "energy_wh": self.energy_wh,
                "co2_kg": self.co2_kg, "fuel_usd": self.fuel_usd}


class CurtailmentSink(vs.Dispatchable):
    """Absorbs the surplus that the grid and the battery cannot take.

    Vessim requires every microgrid to balance exactly, so a solar farm on a
    sunny day with a capped export connection has nowhere to put its energy
    unless the model can spill it.  This dispatchable is the spillway: the
    dispatch policy uses it only after the battery is full and the export
    limit is reached, and it records the spilled energy so the dashboard can
    compare it with PyPSA's ``curtailment``.
    """

    def __init__(self, name: str = "curtailment", p_max: float = 1e12):
        super().__init__(name)
        self.p_max = float(p_max)
        self.curtailed_kwh = 0.0

    def feasible_range(self, duration: float) -> tuple[float, float]:
        return (0.0, self.p_max)

    def step(self, duration: float) -> None:
        power = max(float(self.current_power), 0.0)
        self.curtailed_kwh += power * duration / 3600.0 / 1000.0

    def config(self) -> dict:
        return {"p_max": self.p_max}

    def state(self) -> dict:
        return {"power": self.current_power, "curtailed_kwh": self.curtailed_kwh}


# --------------------------------------------------------------------------- #
# Merit-order / carbon-aware dispatch policy
# --------------------------------------------------------------------------- #

class MeritOrderPolicy(vs.DispatchPolicy):
    """Dispatch by effective marginal cost instead of by list order.

    Every dispatchable gets two scores, one for supplying power and one for
    absorbing it, computed from the current grid signals:

    ===============  ==============================  ==========================
    resource         supplying (USD/kWh)             absorbing (USD/kWh)
    ===============  ==============================  ==========================
    GridConnection   price + carbon_price x carbon    -price x export_factor
    FuelGenerator    fuel_cost + carbon_price x CO2   (cannot absorb)
    SimpleBattery    degradation + SOC penalty        degradation - charge bonus
    ===============  ==============================  ==========================

    The cheapest supplier covers the deficit first, the cheapest absorber takes
    the surplus, and the last resource absorbs the exact remainder so that
    Vessim's balance check always passes.
    """

    def __init__(
        self,
        carbon_price_usd_per_t: float = 60.0,
        export_price_factor: float = 0.35,
        degradation_cost_usd_per_kwh: float = 0.02,
        soc_target: float = 0.5,
        reserve_soc: float = 0.1,
        default_price_usd_per_kwh: float = 0.18,
        default_carbon_kg_per_kwh: float = 0.40,
        charge_bonus_usd_per_kwh: float = 0.04,
        strategy: str = "cost",
    ):
        self.carbon_price_usd_per_t = float(carbon_price_usd_per_t)
        self.export_price_factor = float(export_price_factor)
        self.degradation_cost = float(degradation_cost_usd_per_kwh)
        self.soc_target = float(soc_target)
        self.reserve_soc = float(reserve_soc)
        self.default_price = float(default_price_usd_per_kwh)
        self.default_carbon = float(default_carbon_kg_per_kwh)
        self.charge_bonus = float(charge_bonus_usd_per_kwh)
        self.strategy = strategy
        self.log: list[dict] = []

    # -- scoring ----------------------------------------------------------- #
    def _signals(self, grid_signals: Optional[dict]) -> tuple[float, float]:
        signals = grid_signals or {}
        price = _as_float(signals.get("price"), self.default_price)
        carbon = _as_float(signals.get("carbon"), self.default_carbon)
        return price, carbon

    def _carbon_cost(self, carbon_kg_per_kwh: float) -> float:
        weight = 8.0 if self.strategy == "carbon_first" else 1.0
        return weight * self.carbon_price_usd_per_t / 1000.0 * carbon_kg_per_kwh

    def supply_score(self, d, price: float, carbon: float) -> float:
        if isinstance(d, CurtailmentSink):
            return 1e7  # a spillway never supplies
        carbon_cost = self._carbon_cost(carbon)
        if isinstance(d, vs.GridConnection):
            return price + carbon_cost
        if isinstance(d, FuelGenerator):
            return d.fuel_cost + self._carbon_cost(d.co2_kg_per_kwh)
        if isinstance(d, vs.Storage):
            soc = _soc_of(d)
            urgency = self.soc_target - soc  # positive when the battery is low
            if self.strategy == "battery_first":
                return self.degradation_cost + 0.5 * max(urgency, 0.0)
            return self.degradation_cost + 0.6 * urgency
        return price + carbon_cost

    def absorb_score(self, d, price: float) -> float:
        if isinstance(d, CurtailmentSink):
            return 1e7  # spill only after the battery and the export link
        if isinstance(d, vs.GridConnection):
            return -price * self.export_price_factor
        if isinstance(d, vs.Storage):
            soc = _soc_of(d)
            if soc >= 0.98:
                return self.degradation_cost + 1.0
            bonus = self.charge_bonus if self.strategy != "battery_first" else 2 * self.charge_bonus
            return self.degradation_cost - bonus * (1.0 - soc)
        return 1e6

    # -- the policy -------------------------------------------------------- #
    def apply(self, p_delta: float, duration: float, dispatchables, grid_signals=None) -> None:
        price, carbon = self._signals(grid_signals)
        ranges = []
        for d in dispatchables:
            lo, hi = d.feasible_range(duration)
            lo = -1e12 if not math.isfinite(lo) else lo
            hi = 1e12 if not math.isfinite(hi) else hi
            if isinstance(d, vs.Storage):
                # never plan below the reserve SOC
                capacity = float(getattr(d, "capacity", 0.0))
                soc = _soc_of(d)
                room = max(soc - self.reserve_soc, 0.0) * capacity
                lo = max(lo, -room / (duration / 3600.0))
            ranges.append((lo, hi))

        powers = [0.0] * len(dispatchables)
        remaining = float(p_delta)

        if remaining < 0:  # deficit: suppliers
            order = sorted(range(len(dispatchables)),
                           key=lambda i: self.supply_score(dispatchables[i], price, carbon))
            for i in order:
                lo, _ = ranges[i]
                if lo >= 0:
                    continue
                take = min(-remaining, -lo)
                if take <= 0:
                    continue
                powers[i] = -take
                remaining += take
                if remaining >= -1e-9:
                    break
        else:  # surplus: absorbers
            order = sorted(range(len(dispatchables)),
                           key=lambda i: self.absorb_score(dispatchables[i], price))
            for i in order:
                _, hi = ranges[i]
                if hi <= 0:
                    continue
                take = min(remaining, hi)
                if take <= 0:
                    continue
                powers[i] = take
                remaining -= take
                if remaining <= 1e-9:
                    break

        if abs(remaining) > 1e-6:
            raise RuntimeError(
                f"MeritOrderPolicy cannot balance {p_delta:.3f} W "
                f"({remaining:.3f} W left): relax the grid or generator limits."
            )
        powers = exact_balance(p_delta, powers, ranges)
        for d, power in zip(dispatchables, powers):
            d.set_power(power, duration)
        self.log.append({
            "p_delta": p_delta, "price": price, "carbon": carbon,
            "powers": {d.name: p for d, p in zip(dispatchables, powers)},
        })

    def state(self) -> dict:
        return {"carbon_price_usd_per_t": self.carbon_price_usd_per_t,
                "strategy": self.strategy, "soc_target": self.soc_target}


class OptimalPlanPolicy(vs.DispatchPolicy):
    """Execute a schedule instead of deciding.

    The plan is the PyPSA optimum of the same trace, converted to Vessim's sign
    convention (positive = consumption).  Every step, each dispatchable is asked
    to take the planned setpoint, clipped to what it can physically do; whatever
    is left over — because the device's efficiency or state of charge differs
    from the model's — is taken by the grid and recorded as a tracking error.

    ``run 03 of the dashboard`` uses it to answer the only question an LP cannot
    answer about itself: *is the plan executable by the plant?*
    """

    def __init__(self, plan: dict, grid_name: str = "grid", label: str = "PyPSA plan"):
        self.plan = {name: np.asarray(values, dtype=float) for name, values in plan.items()}
        self.grid_name = grid_name
        self.label = label
        self.index = 0
        self.tracking_wh = 0.0
        self.max_error_w = 0.0
        self.log: list[dict] = []

    @staticmethod
    def _remainder(p_delta: float, powers) -> float:
        """Exactly what Vessim will compute: subtract in dispatch order."""
        remaining = float(p_delta)
        for power in powers:
            remaining -= power
        return remaining

    def apply(self, p_delta: float, duration: float, dispatchables, grid_signals=None) -> None:
        powers: list[float] = []
        limits: list[tuple[float, float]] = []
        for d in dispatchables:
            series = self.plan.get(d.name)
            target = 0.0 if series is None or len(series) == 0 else float(
                series[min(self.index, len(series) - 1)])
            lo, hi = d.feasible_range(duration)
            limits.append((lo, hi))
            powers.append(min(max(target, lo), hi))

        names = [d.name for d in dispatchables]
        grid = names.index(self.grid_name) if self.grid_name in names else len(powers) - 1

        # 1) whatever the plan could not deliver is absorbed by the grid and
        #    counted as tracking error; if the grid is already at its limit the
        #    next resource with room takes it (the plant still has to balance).
        mismatch = float(p_delta) - float(sum(powers))
        if mismatch:
            for index in [grid] + [i for i in range(len(powers)) if i != grid]:
                if mismatch == 0.0:
                    break
                lo, hi = limits[index]
                candidate = min(max(powers[index] + mismatch, lo), hi)
                moved = candidate - powers[index]
                if moved == 0.0:
                    continue
                powers[index] = candidate
                mismatch -= moved
            moved_total = abs(float(p_delta) - float(sum(powers)))
            self.tracking_wh += moved_total * duration / 3600.0
            self.max_error_w = max(self.max_error_w, moved_total)

        # 2) exact balance: the spillway closes the chain and any ulp left by the
        #    plan is nudged onto a resource with room (see ``exact_balance``).
        lo, hi = limits[-1]
        powers[-1] = min(max(self._remainder(p_delta, powers[:-1]), lo), hi)
        powers = exact_balance(p_delta, powers, limits)

        for d, power in zip(dispatchables, powers):
            d.set_power(power, duration)

        self.log.append({"step": self.index, "p_delta": p_delta,
                         "residual_w": self._remainder(p_delta, powers)})
        self.index += 1

    def state(self) -> dict:
        return {"label": self.label, "steps_executed": self.index,
                "tracking_error_wh": self.tracking_wh, "max_error_w": self.max_error_w}


def exact_balance(p_delta: float, powers: list[float], ranges: list[tuple[float, float]],
                  tolerance: float = 1e-6) -> list[float]:
    """Make ``sum(powers) == p_delta`` hold **exactly**, in Vessim's order.

    Vessim subtracts every setpoint from the delta and requires the remainder to
    be ``== 0`` — not ``≈ 0``.  An LP plan (or a merit order that lands on a
    limit) can leave a few ulps behind, which is a rounding artefact, not an
    infeasibility.  This nudges one resource that still has room by exactly the
    remainder, verifying with the same left-to-right subtraction Vessim uses.
    A gap larger than ``tolerance`` is a real imbalance and is raised.
    """

    def remainder(values) -> float:
        left = float(p_delta)
        for value in values:
            left -= value
        return left

    powers = list(powers)
    if remainder(powers) == 0.0:
        return powers
    for _ in range(10):
        left = remainder(powers)
        if left == 0.0:
            return powers
        if abs(left) > tolerance:
            break
        repaired = False
        for index in sorted(range(len(powers)), key=lambda i: abs(powers[i])):
            low, high = ranges[index]
            candidate = powers[index] + left
            if candidate == powers[index] or not low <= candidate <= high:
                continue
            trial = list(powers)
            trial[index] = candidate
            if remainder(trial) == 0.0:
                return trial
        break
    raise RuntimeError(
        f"cannot balance {p_delta:.6f} W exactly ({remainder(powers):.3e} W left): "
        "relax the grid or generator limits.")


def _soc_of(d) -> float:
    """State of charge as a float (Vessim exposes it as a method)."""
    value = getattr(d, "soc", 0.0)
    return float(value() if callable(value) else value)


def _as_float(value, default: float) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, dict):  # a recorded grid_signals cell
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


# --------------------------------------------------------------------------- #
# Carbon-aware deferrable workload
# --------------------------------------------------------------------------- #

class DeferrableLoad(vs.Signal):
    """A workload that moves its flexible share out of the dirty hours.

    The rule: a flexible share ``flexible_w`` of the load may run at most at
    double power, and only when the grid is clean (carbon below ``threshold``);
    the rest of the time it waits.  If there are not enough clean hours, what
    does not fit stays in the dirty hours rather than disappearing.

    The schedule is built **once, as a pure function of the day**, so that the
    total energy is exactly the same as a flat workload — the property that
    makes the comparison with PyPSA meaningful.  (A stateful ledger, evaluated
    twice per step by the simulation and by the plot, silently lost 18 % of the
    workload; that is the kind of bug a `p_set` can never reveal.)
    """

    def __init__(self, base_w: float, flexible_w: float, carbon, threshold: float = 0.35,
                 max_defer_hours: float = 6.0, steps_per_day: int = 96,
                 max_w: Optional[float] = None):
        self.base_w = float(base_w)
        self.flexible_w = float(flexible_w)
        self.carbon = carbon
        self.threshold = float(threshold)
        self.max_defer_hours = float(max_defer_hours)
        #: absolute ceiling for the actor, W — a workload that needs more than the
        #: site can import is not a scheduling problem, it is an outage.
        self.max_w = None if max_w is None else float(max_w)
        self.schedule = self._build(steps_per_day)

    # -- the day, once ------------------------------------------------------ #
    def _build(self, n: int) -> np.ndarray:
        hours = np.arange(n) * 24.0 / n
        dirty = np.array([self.carbon(float(h)) > self.threshold for h in hours])
        cap = 2.0
        if self.max_w is not None and self.flexible_w > 0:
            cap = min(cap, max(self.max_w - self.base_w, 0.0) / self.flexible_w)
        weights = water_fill(dirty, cap=max(cap, 1.0))
        return self.base_w + self.flexible_w * weights

    @staticmethod
    def _hour(elapsed) -> float:
        if elapsed is None:
            return 0.0
        if hasattr(elapsed, "total_seconds"):
            return elapsed.total_seconds() / 3600.0
        return float(elapsed) / 3600.0

    def at(self, elapsed=None) -> float:
        hour = self._hour(elapsed) % 24.0
        index = int(round(hour / 24.0 * len(self.schedule))) % len(self.schedule)
        return float(self.schedule[index])

    def state(self) -> dict:
        flexible = self.schedule - self.base_w
        return {"flexible_kwh": float(flexible.sum() * 24.0 / len(self.schedule) / 1000.0),
                "flat_kwh": float(self.flexible_w * 24.0 / 1000.0)}

    def __repr__(self) -> str:
        return f"DeferrableLoad(base={self.base_w}W, flexible={self.flexible_w}W)"
class SocFloorPolicy(MeritOrderPolicy):
    """A resilience rule: the battery is a reserve, not a trader.

    Below ``floor`` the battery may only charge; above ``ceiling`` it may only
    discharge; in between it follows the merit order.  Twenty lines, and it is
    a *hard operational constraint* that no price can express — which is the
    point: an LP respects only the constraints you remember to write, while the
    controller's author writes them straight into the code.
    """

    def __init__(self, floor: float = 0.4, ceiling: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.floor = float(floor)
        self.ceiling = float(ceiling)

    def supply_score(self, d, price: float, carbon: float) -> float:
        if isinstance(d, vs.Storage) and _soc_of(d) <= self.floor:
            return 1e6                     # forbidden to discharge: a reserve
        return super().supply_score(d, price, carbon)

    def absorb_score(self, d, price: float) -> float:
        if isinstance(d, vs.Storage) and _soc_of(d) >= self.ceiling:
            return 1e6                     # full enough: stop charging
        return super().absorb_score(d, price)


# --------------------------------------------------------------------------- #
# A catalogue of policies, so the dashboard can compare scenarios
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PolicySpec:
    """One dispatch policy, described well enough to reproduce it.

    ``kind`` picks the class; the remaining fields are its parameters.  Keeping
    them in a data object (instead of a lambda) means the UI can show exactly
    what it ran, and ``st.cache_data`` can hash it.
    """

    key: str
    label: str
    kind: str = "merit"                    # default | merit | peak_shaving
    description: str = ""
    strategy: str = "cost"                 # cost | carbon_first | battery_first
    carbon_price: float = 60.0             # USD/tCO2
    degradation_cost_usd_per_kwh: float = 0.02
    export_price_factor: float = 0.35
    soc_target: float = 0.5
    charge_bonus_usd_per_kwh: float = 0.04
    peak_cap_kw: float = 1.2               # only for peak_shaving


#: The scenarios the dashboard offers.  The first one of each case is the
#: policy the official example actually uses; the rest are deliberate
#: counterfactuals.
POLICY_CATALOGUE: dict[str, PolicySpec] = {
    "official": PolicySpec(
        "official", "Official example (as shipped)",
        description="Whatever the case's own spec says: Vessim's DefaultDispatchPolicy in cases "
                    "01/02/04, our MeritOrderPolicy in 03/05."),
    "default": PolicySpec(
        "default", "DefaultDispatchPolicy (list order)",
        kind="default",
        description="Vessim's built-in policy: allocate to the dispatchables in the order they were "
                    "declared, battery first, then the grid. It ignores price and carbon entirely."),
    "merit_cost": PolicySpec(
        "merit_cost", "Merit order by cost", kind="merit", strategy="cost",
        description="Cheapest effective USD/MWh first: wholesale price + carbon price for the grid, "
                    "fuel + carbon for the turbine, degradation + SoC urgency for the battery."),
    "merit_carbon": PolicySpec(
        "merit_carbon", "Carbon first (×8 weight)", kind="merit", strategy="carbon_first",
        description="Same policy with the carbon term multiplied by eight: it minimises kgCO2 per "
                    "kWh even where that costs money."),
    "merit_battery": PolicySpec(
        "merit_battery", "Battery first (protect the tank)", kind="merit", strategy="battery_first",
        description="The battery is discharged whenever it can be, and charged back aggressively: a "
                    "resilience policy, not an economic one."),
    "merit_free_cycle": PolicySpec(
        "merit_free_cycle", "Merit order, free cycling", kind="merit", strategy="cost",
        degradation_cost_usd_per_kwh=0.0, charge_bonus_usd_per_kwh=0.0,
        description="The economic policy with the battery's wear priced at zero — the mistake a "
                    "naive optimisation comparison makes on the PyPSA side."),
    "merit_no_export": PolicySpec(
        "merit_no_export", "Merit order, no export revenue", kind="merit", strategy="cost",
        export_price_factor=0.0,
        description="Selling back to the grid pays nothing, so surplus sun must be stored or spilled."),
    "peak_shaving": PolicySpec(
        "peak_shaving", "Peak shaving (1.2 kW cap + overflow)", kind="peak_shaving",
        description="A contract-style policy: a hard import cap with a second, unlimited connection "
                    "behind it. The battery covers the peak until it is empty."),
}

#: Order in which the matrix is presented (cheap first, exotic last).
POLICY_ORDER = ("default", "merit_cost", "merit_carbon", "merit_battery",
                "merit_free_cycle", "merit_no_export", "peak_shaving")


def build_policy(spec: PolicySpec, case, carbon_price: float | None = None,
                 default_price: float = 0.18, default_carbon: float = 0.40):
    """Instantiate the dispatch policy described by ``spec``."""
    if spec.kind == "default":
        return vs.DefaultDispatchPolicy()
    if spec.kind == "peak_shaving":
        return vs.DefaultDispatchPolicy()
    tau = spec.carbon_price if carbon_price is None else float(carbon_price)
    return MeritOrderPolicy(
        carbon_price_usd_per_t=tau,
        export_price_factor=spec.export_price_factor,
        degradation_cost_usd_per_kwh=spec.degradation_cost_usd_per_kwh,
        soc_target=spec.soc_target,
        charge_bonus_usd_per_kwh=spec.charge_bonus_usd_per_kwh,
        default_price_usd_per_kwh=default_price,
        default_carbon_kg_per_kwh=default_carbon,
        strategy=spec.strategy,
    )


def score_table(policy, dispatchables, price: float, carbon: float,
                duration: float = 900.0) -> list[dict]:
    """The numbers the policy compares at one step — no simulation required.

    For :class:`MeritOrderPolicy` the two scores are what it sorts on: a
    dispatchable that can supply is ranked by ``supply_score`` (USD/MWh, carbon
    included), one that can absorb by ``absorb_score``.  For Vessim's
    ``DefaultDispatchPolicy`` there is nothing to compare: the order in the list
    *is* the decision, so the score columns stay empty and ``order`` says it.
    """
    rows = []
    for index, d in enumerate(dispatchables):
        row = {"order": index + 1, "name": d.name, "kind": type(d).__name__}
        try:
            lo, hi = d.feasible_range(duration)
        except Exception:                       # pragma: no cover - defensive
            lo, hi = float("nan"), float("nan")
        row["can_supply_mw"] = -lo / 1e6 if math.isfinite(lo) and lo < 0 else 0.0
        row["can_absorb_mw"] = hi / 1e6 if math.isfinite(hi) and hi > 0 else 0.0
        if isinstance(policy, MeritOrderPolicy):
            # Vessim works in USD/kWh; the dashboard talks in USD/MWh
            row["supply_score"] = 1000.0 * policy.supply_score(d, price, carbon)
            row["absorb_score"] = 1000.0 * policy.absorb_score(d, price)
        else:
            row["supply_score"] = float("nan")
            row["absorb_score"] = float("nan")
        if isinstance(d, vs.Storage):
            row["soc"] = _soc_of(d)
        else:
            row["soc"] = float("nan")
        rows.append(row)
    return rows
