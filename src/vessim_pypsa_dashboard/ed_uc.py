"""Economic dispatch and unit commitment, written out for teaching.

The ED/UC Simulator by Adjie Brotot (https://tool.adjiebrotots.com/ed-uc-simulator/)
solves both problems with PSA.js + HiGHS, i.e. with the same mixed-integer
program that PyPSA builds.  Solving a MILP teaches very little about *how* the
answer is found, so this module implements the two classical textbook
algorithms instead:

* **Economic dispatch by lambda iteration.**  For convex cost curves
  ``C_i(P) = b_i P + c_i P^2`` the optimality condition is equal incremental
  cost ``dC_i/dP = lambda`` for every unconstrained unit.  We find the system
  lambda by bisection on the monotone aggregate supply curve, then repair the
  last few kW on the marginal unit so that generation matches demand exactly.

* **Unit commitment by priority list.**  Units are ranked by their average cost
  at full output and committed cheapest-first until capacity covers the load
  plus reserve, honouring minimum up/down times and ramp limits.

Both are heuristics (the priority list is *not* guaranteed to be optimal), and
that is the point: the dashboard puts their answer next to PyPSA's MILP and
shows the gap.  Everything is in MW and USD; time steps are one hour by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Cost curves
# --------------------------------------------------------------------------- #

def pmin_of(unit: pd.Series) -> float:
    return float(unit.get("p_min_pu", 0.0) or 0.0) * float(unit.get("p_nom", 0.0) or 0.0)


def pmax_of(unit: pd.Series) -> float:
    return float(unit.get("p_nom", 0.0) or 0.0)


def marginal_cost_at(unit: pd.Series, p: float) -> float:
    """dC/dP = b + 2 c P  (USD/MWh)."""
    b = float(unit.get("marginal_cost", 0.0) or 0.0)
    c = float(unit.get("marginal_cost_quadratic", 0.0) or 0.0)
    return b + 2 * c * p


def energy_cost(unit: pd.Series, p: float) -> float:
    """C(P) = b P + c P^2  (USD/h); slopes are non-negative."""
    b = float(unit.get("marginal_cost", 0.0) or 0.0)
    c = float(unit.get("marginal_cost_quadratic", 0.0) or 0.0)
    return b * p + c * p * p


def average_cost(unit: pd.Series) -> float:
    pmax = pmax_of(unit)
    if pmax <= 0:
        return float("inf")
    return energy_cost(unit, pmax) / pmax


# --------------------------------------------------------------------------- #
# Economic dispatch: lambda iteration
# --------------------------------------------------------------------------- #

@dataclass
class DispatchResult:
    units: pd.DataFrame          # p_mw, cost_usd, marginal_usd, on, pmin, pmax
    demand: float
    lam: float
    cost: float
    served: float
    shed: float
    spill: float
    converged: bool
    history: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def cost_no_shed(self) -> float:
        return float(self.units["cost_usd"].sum())


def _supply_at(units: pd.DataFrame, lam: float, bounds: pd.DataFrame) -> np.ndarray:
    """P_i(lambda) clipped to each unit's bounds (the aggregate supply curve)."""
    b = units["marginal_cost"].to_numpy(dtype=float)
    c = units["marginal_cost_quadratic"].to_numpy(dtype=float)
    lo = bounds["lo"].to_numpy(dtype=float)
    hi = bounds["hi"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(c > 0, (lam - b) / (2 * c), np.where(lam > b, hi, lo))
    return np.clip(p, lo, hi)


def economic_dispatch(
    units: pd.DataFrame,
    demand: float,
    bounds: pd.DataFrame | None = None,
    reserve_mw: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 100,
    voll: float = 10_000.0,
) -> DispatchResult:
    """Lambda iteration on a copper-plate system.

    Parameters
    ----------
    units:
        Generator table; uses ``p_nom``, ``p_min_pu``, ``marginal_cost`` and
        ``marginal_cost_quadratic``.
    demand:
        Load in MW (the reserve is added on top of it).
    bounds:
        Optional per-unit ``lo``/``hi`` limits (used by the UC heuristic to
        enforce ramp rates).  Defaults to ``[pmin, pmax]``.
    """
    units = units.copy()
    if units.empty:
        raise ValueError("Economic dispatch needs at least one generator.")
    if bounds is None:
        bounds = pd.DataFrame({
            "lo": [pmin_of(u) for _, u in units.iterrows()],
            "hi": [pmax_of(u) for _, u in units.iterrows()],
        }, index=units.index)
    else:
        bounds = bounds.reindex(units.index).fillna(0.0)

    target = float(demand) + float(reserve_mw)
    lo_bound = float(bounds["lo"].sum())
    hi_bound = float(bounds["hi"].sum())

    feasible = lo_bound - tol <= target <= hi_bound + tol
    if target > hi_bound:
        served_target = hi_bound
    elif target < lo_bound:
        served_target = lo_bound
    else:
        served_target = target

    lam_lo = float((units["marginal_cost"]).min()) - 1.0
    lam_hi = float((units["marginal_cost"] + 2 * units["marginal_cost_quadratic"]
                    * bounds["hi"]).max()) + 1.0
    history = []
    if served_target <= 0:
        p = np.zeros(len(units))
        lam = lam_lo
    else:
        for it in range(max_iter):
            lam = 0.5 * (lam_lo + lam_hi)
            p = _supply_at(units, lam, bounds)
            total = float(p.sum())
            history.append({"iteration": it, "lambda": lam, "total_mw": total,
                            "error_mw": total - served_target})
            if abs(total - served_target) < tol:
                break
            if total < served_target:
                lam_lo = lam
            else:
                lam_hi = lam
        p = _supply_at(units, lam, bounds)

    # --- exact repair: put the residual on the cheapest unit with headroom ----
    residual = served_target - float(p.sum())
    if abs(residual) > 1e-9:
        order = np.argsort(units["marginal_cost"].to_numpy(dtype=float))
        if residual > 0:
            for idx in order:
                room = bounds["hi"].iloc[idx] - p[idx]
                if room <= 0:
                    continue
                add = min(room, residual)
                p[idx] += add
                residual -= add
                if residual <= 1e-9:
                    break
        else:
            for idx in order[::-1]:
                room = p[idx] - bounds["lo"].iloc[idx]
                if room <= 0:
                    continue
                cut = min(room, -residual)
                p[idx] -= cut
                residual += cut
                if residual >= -1e-9:
                    break

    costs = np.array([energy_cost(u, pi) for (_, u), pi in zip(units.iterrows(), p)])
    out = pd.DataFrame({
        "p_mw": p,
        "pmin": bounds["lo"].to_numpy(dtype=float),
        "pmax": bounds["hi"].to_numpy(dtype=float),
        "cost_usd": costs,
        "marginal_usd": [marginal_cost_at(u, pi) for (_, u), pi in zip(units.iterrows(), p)],
        "on": p > 1e-9,
    }, index=units.index)

    served = float(min(p.sum(), served_target))
    shed = float(max(served_target - p.sum(), 0.0))
    spill = float(max(p.sum() - served_target, 0.0))
    total_cost = float(costs.sum()) + shed * voll
    # System lambda: the incremental cost of the marginal MWh.  A unit pinned
    # at its minimum is must-run and does not set the price, so interior units
    # (the one that absorbed the repair) win over the most expensive runner.
    dispatched = out.loc[out["p_mw"] > 1e-9]
    interior = dispatched[(dispatched["p_mw"] > dispatched["pmin"] + 1e-6)
                          & (dispatched["p_mw"] < dispatched["pmax"] - 1e-6)]
    if len(interior):
        lam_out = float(interior["marginal_usd"].max())
    elif len(dispatched):
        lam_out = float(dispatched["marginal_usd"].max())
    else:
        lam_out = float(lam)
    return DispatchResult(
        units=out, demand=float(demand), lam=lam_out, cost=total_cost,
        served=served, shed=shed, spill=spill, converged=feasible,
        history=pd.DataFrame(history),
    )


# --------------------------------------------------------------------------- #
# Unit commitment: priority list + lambda iteration
# --------------------------------------------------------------------------- #

@dataclass
class CommitmentResult:
    hourly: pd.DataFrame        # one row per hour
    dispatch: pd.DataFrame      # hours x units, MW
    status: pd.DataFrame        # hours x units, 1/0
    dispatch_details: list[DispatchResult]
    total_cost: float
    energy_cost: float
    startup_cost: float
    shed_mwh: float
    spill_mwh: float
    voll: float

    @property
    def committed_hours(self) -> int:
        return int(self.status.to_numpy().sum())


def unit_commitment_priority(
    units: pd.DataFrame,
    load: list[float] | np.ndarray,
    reserve_frac: float = 0.10,
    voll: float = 10_000.0,
    dt: float = 1.0,
) -> CommitmentResult:
    """Priority-list unit commitment, hour by hour.

    The heuristic commits the cheapest units (by average cost at full output)
    until the committed capacity covers load plus reserve.  Minimum up/down
    times are enforced by "sticky" states, ramp limits by narrowing each unit's
    bounds around its previous output.  Where the heuristic cannot respect a
    constraint it reports load shedding or spill instead of pretending.
    """
    units = units.copy()
    names = list(units.index)
    hours = len(load)
    pmin = {n: pmin_of(units.loc[n]) for n in names}
    pmax = {n: pmax_of(units.loc[n]) for n in names}
    min_up = {n: int(units.at[n, "min_up_time"] or 0) for n in names}
    min_down = {n: int(units.at[n, "min_down_time"] or 0) for n in names}
    ramp_up = {n: float(units.at[n, "ramp_limit_up"] or 1.0) for n in names}
    ramp_down = {n: float(units.at[n, "ramp_limit_down"] or 1.0) for n in names}
    startup_cost = {n: float(units.at[n, "start_up_cost"] or 0.0) for n in names}

    order = sorted(names, key=lambda n: (average_cost(units.loc[n]), marginal_cost_at(units.loc[n], pmin[n])))
    init = {n: int(units.at[n, "init"] or 0) if "init" in units.columns else 1 for n in names}
    status = {n: init[n] for n in names}
    hours_on = {n: (min_up[n] if init[n] else 0) for n in names}
    hours_off = {n: (0 if init[n] else min_down[n]) for n in names}
    prev_p = {n: 0.0 for n in names}

    dispatch_rows, status_rows, hourly = [], [], []
    details: list[DispatchResult] = []
    total_energy, total_startup, total_shed, total_spill = 0.0, 0.0, 0.0, 0.0

    for hour, load_mw in enumerate(load):
        load_mw = float(load_mw)
        required = load_mw * (1 + reserve_frac)

        # 1) sticky units: minimum up time not yet satisfied, or minimum down time
        for n in names:
            if status[n] == 1 and hours_on[n] < max(min_up[n], 1):
                pass  # must stay on
        must_on = {n for n in names if status[n] == 1 and hours_on[n] < min_up[n]}
        blocked_off = {n for n in names if status[n] == 0 and hours_off[n] < min_down[n]}

        committed = set(must_on)
        capacity = sum(pmax[n] for n in committed)
        # 2) add the cheapest available units until reserve is covered
        for n in order:
            if capacity >= required:
                break
            if n in committed or n in blocked_off:
                continue
            committed.add(n)
            capacity += pmax[n]

        # 3) drop units that are not needed any more (most expensive first)
        for n in sorted(committed, key=lambda x: -average_cost(units.loc[x])):
            if n in must_on:
                continue
            # only drop if the remaining units can still cover the load and
            # their minimum output does not exceed it
            remaining_capacity = capacity - pmax[n]
            remaining_min = sum(pmin[m] for m in committed if m != n)
            if remaining_capacity >= max(required, load_mw) and remaining_min <= load_mw:
                committed.discard(n)
                capacity = remaining_capacity

        # 4) dispatch among committed units within ramp and minimum limits.
        #    The first hour, and any start-up, ramps from zero up to full output
        #    (PyPSA's ramp_limit_start_up defaults to 100 % of the rating).
        on_names = [n for n in names if n in committed]
        if on_names:
            rows = []
            for n in on_names:
                starting = (hour == 0) or (status[n] == 0)
                if starting:
                    lo, hi = pmin[n], pmax[n]
                else:
                    lo = max(pmin[n], prev_p[n] - ramp_down[n] * pmax[n], 0.0)
                    hi = min(pmax[n], prev_p[n] + ramp_up[n] * pmax[n])
                hi = max(hi, lo)
                rows.append({"lo": lo, "hi": hi})
            bounds = pd.DataFrame(rows, index=on_names)
            sub = units.loc[on_names]
            res = economic_dispatch(sub, load_mw, bounds=bounds, voll=voll)
        else:
            res = DispatchResult(
                units=pd.DataFrame(columns=["p_mw", "pmin", "pmax", "cost_usd", "marginal_usd", "on"]),
                demand=load_mw, lam=float("nan"), cost=load_mw * voll, served=0.0,
                shed=load_mw, spill=0.0, converged=False,
            )

        p_row = {n: 0.0 for n in names}
        for n in on_names:
            p_row[n] = float(res.units.at[n, "p_mw"])

        # 5) bookkeeping: start-ups, min up/down counters
        start_cost = 0.0
        for n in names:
            now_on = 1 if n in committed else 0
            if now_on and not status[n]:
                start_cost += startup_cost[n]
            if now_on:
                hours_on[n] += 1
                hours_off[n] = 0
            else:
                hours_on[n] = 0
                hours_off[n] += 1
            status[n] = now_on
            prev_p[n] = p_row[n]

        served = sum(p_row.values())
        shed = max(load_mw - served, 0.0)
        spill = max(served - load_mw, 0.0)
        total_energy += res.units["cost_usd"].sum() * dt if len(res.units) else 0.0
        total_startup += start_cost
        total_shed += shed * dt
        total_spill += spill * dt

        dispatch_rows.append(p_row)
        status_rows.append({n: status[n] for n in names})
        hourly.append({
            "hour": hour,
            "load_mw": load_mw,
            "served_mw": served,
            "shed_mw": shed,
            "spill_mw": spill,
            "on_count": len(committed),
            "committed": ", ".join(sorted(committed)) if committed else "—",
            "energy_cost_usd": float(res.units["cost_usd"].sum()) if len(res.units) else 0.0,
            "startup_cost_usd": start_cost,
            "price_usd_mwh": float(res.lam) if np.isfinite(res.lam) else voll,
        })
        details.append(res)

    hourly_df = pd.DataFrame(hourly)
    hourly_df["cost_usd"] = hourly_df["energy_cost_usd"] + hourly_df["startup_cost_usd"]
    return CommitmentResult(
        hourly=hourly_df,
        dispatch=pd.DataFrame(dispatch_rows, columns=names),
        status=pd.DataFrame(status_rows, columns=names),
        dispatch_details=details,
        total_cost=float(total_energy + total_startup + total_shed * voll),
        energy_cost=float(total_energy),
        startup_cost=float(total_startup),
        shed_mwh=float(total_shed),
        spill_mwh=float(total_spill),
        voll=float(voll),
    )
